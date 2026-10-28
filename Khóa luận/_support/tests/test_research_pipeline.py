"""Research invariants and complete synthetic OOF workflow; no pretrained claims."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from clindistill.artifacts import file_hash, read_json, write_json
from clindistill.research.data import FeatureStore, VisitDataset, clinical_input, model_assets_hash
from clindistill.research.losses import per_case_kl, reliability_weights, weighted_kd
from clindistill.research.models import GatedMIL, VisitClassifier
from clindistill.research.training import grouped_folds, fit, run_study, study_plan
from clindistill.research.reporting import evaluate_study, export_student, predict_student

PROJECT = Path(__file__).resolve().parents[2]


def config():
    cfg = read_json(PROJECT / 'configs/code/research.json')
    cfg.update(debug=True, seeds=[2026], folds=2, epochs=4, patience=4, feature_dim=16,
               learning_rate=0.025, batch_size=8, dropout=0.1, bootstrap_repetitions=10)
    return cfg


def tiny_encoders(root):
    from transformers import DINOv3ViTConfig, DINOv3ViTModel, XLMRobertaConfig, XLMRobertaModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    root = Path(root); paths = {}
    for role, dim in [('student', 32), ('teacher', 48)]:
        target = root / role; target.mkdir(parents=True)
        cfg = DINOv3ViTConfig(hidden_size=dim, intermediate_size=dim*2, num_hidden_layers=1,
            num_attention_heads=4, patch_size=16, num_register_tokens=2)
        DINOv3ViTModel(cfg).save_pretrained(target)
        write_json(target / 'preprocessor_config.json', {'image_mean': [.485,.456,.406], 'image_std': [.229,.224,.225]})
        paths[role] = target
    target = root / 'clinical'; target.mkdir()
    vocab = {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, 'chief': 4, 'complaint': 5, 'pain': 6, ':': 7}
    backend = Tokenizer(WordLevel(vocab, unk_token='<unk>')); backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='<s>', eos_token='</s>',
        pad_token='<pad>', unk_token='<unk>', model_input_names=['input_ids', 'attention_mask']).save_pretrained(target)
    XLMRobertaModel(XLMRobertaConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=32, max_position_embeddings=514)).save_pretrained(target, safe_serialization=False)
    paths['clinical'] = target
    return paths


def synthetic_features(root, *, encoders=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(99)
    meta = {'schema_version': 4, 'debug': True, 'release_stage': 'experimental',
        'class_names': ['A','B'], 'dimensions': {'student':64, 'teacher':96,'resnet50':12,'swin_b':16,'clinical':16},
        'request': {'release_id':'synthetic-only', 'image_size':224,'max_photos':8,'encoder_training':'frozen_features',
            'builder_sha256':file_hash(PROJECT/'clindistill/research/data.py'),
            'model_ids': {k:config()[k+'_model'] for k in ('student','teacher','clinical')},
            'policy': {'fields':['age','gender','chief_complaint'],
            'language':'en','temporal_verified':False,'status':'screened_exploratory'},
            'model_fingerprints': {'student': model_assets_hash(encoders['student']) if encoders else 'synthetic'}},
        'model_locations': {k:str(v) for k,v in (encoders or {}).items()}, 'splits': {}}
    for split, n in [('train',24), ('validation',8)]:
        y = (np.arange(n)//2 % 2).astype(np.int64)
        data = {'ids':np.asarray([f'{split}-{i}' for i in range(n)]),
            'patients':np.asarray([f'{split}-p{i//2}' for i in range(n)]),
            'groups':np.asarray([f'{split}-g{i//2}' for i in range(n)]),
            'bags':np.asarray([f'{split}-b{i}' for i in range(n)]),
            'labels':y, 'offsets':np.arange(n+1)*2, 'image_indices':np.arange(n*2),
            'image_paths':np.asarray([f'{split}-{i}.jpg' for i in range(n*2)]),
            'clinical':np.eye(2,dtype=np.float32)[y] @ rng.normal(size=(2,16)).astype(np.float32),
            'demographics':np.zeros((n,5),np.float32), 'clinical_present':np.ones(n,bool)}
        for key in ('student','teacher','resnet50','swin_b'):
            vectors = np.eye(2)[np.repeat(y,2)] @ rng.normal(size=(2,meta['dimensions'][key]))
            data[key] = (vectors + rng.normal(scale=.01,size=vectors.shape)).astype(np.float16)
        np.savez_compressed(root / f'{split}.npz', **data)
        meta['splits'][split] = {'sha256':file_hash(root/f'{split}.npz'),'cases':n}
    write_json(root/'feature_manifest.json',meta)
    return root


class ResearchInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_primary_plan_and_clinical_label_independence(self):
        cfg = read_json(PROJECT/'configs/code/research.json')
        self.assertEqual(study_plan(cfg)['total_fits'],57)
        row = {'age_raw':'32','gender_raw':'Female','clinical_raw_allowlist':{
            'chief_complaint':'pain for two days','diagnosis':'secret target','examination':'diagnosis label'}}
        original = clinical_input(row,cfg['clinical_fields'])
        altered = copy.deepcopy(row); altered['labels']=[1]; altered['clinical_raw_allowlist']['diagnosis']='OTHER'
        revised = clinical_input(altered,cfg['clinical_fields'])
        self.assertEqual(original[0],revised[0]); np.testing.assert_array_equal(original[1],revised[1])
        self.assertNotIn('secret',original[0])
        with self.assertRaisesRegex(ValueError,'forbidden'):
            clinical_input(row,['diagnosis'])
        row['clinical_raw_allowlist']['chief_complaint']='diagnosed with caries'
        self.assertEqual(clinical_input(row,cfg['clinical_fields'])[3],['chief_complaint'])

    def test_mil_padding_permutation_and_student_boundary(self):
        torch.manual_seed(1); model=VisitClassifier(8,2,width=8,dropout=0).eval()
        x=torch.randn(2,3,8); mask=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool)
        a=model(x,mask)['logits']; x[0,2]=1e6
        torch.testing.assert_close(a,model(x,mask)['logits'])
        torch.testing.assert_close(a,model(x[:,[2,0,1]],mask[:,[2,0,1]])['logits'])
        with self.assertRaisesRegex(ValueError,'clinical'):
            model(x,mask,clinical=torch.zeros(2,4))

    def test_kl_direction_detach_temperature_and_batch_one_weights(self):
        s=torch.tensor([[2.,-1.],[-.4,.9]],requires_grad=True)
        t=torch.tensor([[.1,.7],[1.,-.2]],requires_grad=True)
        expected=F.kl_div(F.log_softmax(s/4,-1),F.softmax(t.detach()/4,-1),reduction='none').sum(-1)*16
        torch.testing.assert_close(per_case_kl(s,t,4),expected)
        w=torch.tensor([.1,.9]); loss=weighted_kd(s,t,w,float(w.mean()),4)
        gradient=torch.autograd.grad(loss,s,retain_graph=True)[0]
        micro=torch.cat([torch.autograd.grad(weighted_kd(s[i:i+1],t[i:i+1],w[i:i+1],float(w.mean()),4)/2,
                           s,retain_graph=True)[0][i:i+1] for i in range(2)])
        torch.testing.assert_close(gradient,micro)
        loss.backward(); self.assertIsNone(t.grad)
        wrong=reliability_weights(torch.zeros(2,2),torch.tensor([[0.,5.],[5.,0.]]),torch.tensor([0,1]))
        self.assertTrue(torch.all(wrong['reliability']==0))
        with self.assertRaisesRegex(ValueError,'collapsed'):
            weighted_kd(s,t,w,0,4)

    def test_group_folds_and_feature_checksums(self):
        with tempfile.TemporaryDirectory() as d:
            path=synthetic_features(Path(d)/'cache'); store=FeatureStore(path,allow_debug=True)
            train=store.data['train']; coverage=np.zeros(24,int)
            for fit_ix,heldout in grouped_folds(train['labels'],train['groups'],2,2026):
                self.assertFalse(set(train['groups'][fit_ix]) & set(train['groups'][heldout])); coverage[heldout]+=1
            np.testing.assert_array_equal(coverage,np.ones(24))
            item=VisitDataset(store,'train','student')[0]
            self.assertNotIn('clinical',item); self.assertNotIn('demographics',item)
            with self.assertRaisesRegex(ValueError,'clinical vector'):
                VisitDataset(store,'train','clinical')
            (path/'train.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                FeatureStore(path,allow_debug=True)

    def test_complete_oof_study_resume_evaluation_and_export(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); encoders=tiny_encoders(root/'encoders')
            cache=synthetic_features(root/'features',encoders=encoders); cfg=config(); study=root/'study'
            summary=run_study(cache,study,cfg,require_gpu=False)
            self.assertEqual(len(summary),9)
            proof=read_json(study/'_oof/seed_2026/targets.json')
            self.assertEqual(proof['weight_normalization'],'fixed_global_OOF_train_mean')
            for fold in proof['folds']:
                self.assertFalse(set(fold['train_groups']) & set(fold['heldout_groups']))
                self.assertFalse(set(fold['train_ids']) & set(fold['heldout_ids']))
            hashes={p:file_hash(p) for p in study.rglob('*.pt')}
            run_study(cache,study,cfg,require_gpu=False)
            self.assertEqual(hashes,{p:file_hash(p) for p in hashes})
            table,_=evaluate_study(cache,study)
            self.assertEqual(len(table),9)
            with self.assertRaisesRegex(ValueError,'Locked test'):
                evaluate_study(cache,study,split='test')
            export=export_student(cache,study,encoders['student'],root/'export')
            self.assertEqual(export['input'],'photos only; no clinical data or labels')
            from PIL import Image
            Image.new('RGB',(32,32),(80,20,40)).save(root/'photo.jpg')
            result=predict_student(root/'export',['photo.jpg'],root,root/'inference')
            self.assertEqual(set(result['probabilities']),{'A','B'})
            self.assertAlmostEqual(sum(result['probabilities'].values()),1,places=5)
            # Changed target bytes must be rejected even after completed training.
            target=study/'_oof/seed_2026/targets.npz'; target.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'OOF targets'):
                run_study(cache,study,cfg,require_gpu=False)

    def test_epoch_resume_matches_uninterrupted_fit(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); cache=synthetic_features(root/'features'); store=FeatureStore(cache,allow_debug=True); cfg=config()
            fit(store,root/'full',cfg,'student_no_kd',3)
            fit(store,root/'resume',cfg,'student_no_kd',3,stop_after_epoch=1)
            fit(store,root/'resume',cfg,'student_no_kd',3)
            a=torch.load(root/'full/last.pt',weights_only=False); b=torch.load(root/'resume/last.pt',weights_only=False)
            for name in a['model']: torch.testing.assert_close(a['model'][name],b['model'][name],rtol=0,atol=0)


class FoundationPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_real_transformers_backends_prepare_features_and_resume(self):
        from clindistill.research.data import prepare_features
        from PIL import Image
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); encoders=tiny_encoders(root/'encoders'); prepared=root/'prepared'; prepared.mkdir()
            checks={}; cfg=config(); cfg.update(methods=['student_no_kd'],image_size=32,image_batch_size=2)
            for split in ('train','validation','test'):
                records=[]; contexts=[]
                for i in range(4):
                    name=f'{split}-{i}'; photo=f'{name}.jpg'; Image.new('RGB',(24+i,32),(30*i,40,80)).save(root/photo)
                    records.append({'case_id':name,'patient_id':name,'split_group':name,'bag_signature':name,
                        'split':split,'eligible_single':True,'target_labels':['A' if i%2==0 else 'B'],
                        'labels_multihot':[int(i%2==0),int(i%2==1)],'single_label_index':i%2,'photo_paths':[photo]})
                    contexts.append({'case_id':name,'patient_id':name,'split':split,'age_raw':'30','gender_raw':'Male',
                                     'clinical_raw_allowlist':{'chief_complaint':'pain'}})
                for kind,rows in [('student_single',records),('teacher_context',contexts)]:
                    path=prepared/f'{kind}_{split}.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
                    checks[path.name]=file_hash(path)
            write_json(prepared/'release.json',{'schema_version':1,'dataset':'COde','stage':'experimental',
                'release_id':'tiny-encoder-test','class_names':['A','B'],'file_checksums':checks})
            first=prepare_features(prepared,root,encoders,root/'cache',cfg)
            store=FeatureStore(root/'cache',allow_debug=True)
            self.assertEqual(first['dimensions'],{'student':64,'teacher':96,'clinical':16})
            self.assertEqual(store.data['train']['student'].shape,(4,64))
            original=file_hash(root/'cache/train.npz')
            prepare_features(prepared,root,encoders,root/'cache',cfg)
            self.assertEqual(original,file_hash(root/'cache/train.npz'))
            before=store.identity
            prepare_features(prepared,root,encoders,root/'cache',cfg,splits=('test',))
            self.assertEqual(before,FeatureStore(root/'cache',allow_debug=True).identity)
            self.assertEqual(original,file_hash(root/'cache/train.npz'))
            self.assertEqual(len(FeatureStore(root/'cache',splits=('train','validation','test'),allow_debug=True).data),3)
            changed=copy.deepcopy(cfg); changed['clinical_fields']=['age','examination']
            with self.assertRaisesRegex(ValueError,'Forbidden'):
                prepare_features(prepared,root,encoders,root/'cache',changed)
