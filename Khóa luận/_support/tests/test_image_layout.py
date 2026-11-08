"""Real filesystem aliases must retain canonical identities and exact photo bytes."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from clindistill.artifacts import file_hash
from clindistill.research.data import resolve_image_sources
from clindistill.research.data import encode_images
from test_research_pipeline import tiny_encoders

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('qwen_image_layout', ROOT/'_support/image_layout.py')
layout = importlib.util.module_from_spec(spec); spec.loader.exec_module(layout)


class ImageLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root, canonical, content=b'original photo', *, supplement=False):
        data = root/'input'/('supplement' if supplement else 'qwen')/'data'
        folder = data/'images'; folder.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha1(canonical.encode()).hexdigest()[:10] + '_' + Path(canonical).name
        photo = folder/name; photo.write_bytes(content)
        return data, photo

    def test_sha1_prefix_is_disambiguated_from_same_basename_radiograph(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); canonical='Images/Photographs/0127-001-01.jpg'
            data, photo=self.fixture(root,canonical)
            _, radio=self.fixture(root,'Images/Radiographs/0127-001-01.jpg',b'X-ray must not be selected')
            self.assertEqual(photo.name,'b42b6cfb7a_0127-001-01.jpg')
            self.assertEqual(radio.name,'f3f5a4e7c9_0127-001-01.jpg')
            index={canonical:file_hash(photo)}
            view=layout.verified_image_view([data],index,root/'working',required_paths=[canonical])
            actual=resolve_image_sources([canonical],view,index)
            self.assertEqual(file_hash(actual[canonical]),file_hash(photo))
            self.assertNotEqual(file_hash(actual[canonical]),file_hash(radio))
            self.assertEqual(layout.verified_image_view([data],index,root/'working',required_paths=[canonical]),view)
            report=json.loads((root/'working/image_layout_report.json').read_text())
            self.assertEqual(report['status'],'verified')
            self.assertEqual(report['missing_required'],[])

    def test_supplement_completes_cohort_and_unneeded_missing_photos_do_not_filter_cases(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);a='Images/Photographs/a.jpg';b='Images/Photographs/b.jpg';unneeded='Images/Photographs/test-only.jpg'
            data,pa=self.fixture(root,a);extra,pb=self.fixture(root,b,b'additional original',supplement=True)
            index={a:file_hash(pa),b:file_hash(pb),unneeded:'0'*64}
            with self.assertRaisesRegex(FileNotFoundError,'1 required photos'):
                layout.verified_image_view([data],index,root/'working',required_paths=[a,b])
            self.assertFalse((root/'Images').exists())
            report=json.loads((root/'working/image_layout_report.json').read_text())
            self.assertEqual(report['missing_required'],[b])
            view=layout.verified_image_view([data,extra],index,root/'working',required_paths=[a,b])
            self.assertEqual(set(resolve_image_sources([a,b],view,index)),{a,b})
            self.assertEqual(json.loads((root/'working/image_layout_report.json').read_text())['unavailable_indexed'],[unneeded])

    def test_hashed_filename_alone_is_not_a_byte_identity_proof(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);canonical='Images/Photographs/a.jpg';data,photo=self.fixture(root,canonical)
            original=file_hash(photo);photo.write_bytes(b'changed or resized image')
            with self.assertRaisesRegex(FileNotFoundError,'differ from original bytes'):
                layout.verified_image_view([data],{canonical:original},root/'working',required_paths=[canonical])
            self.assertFalse((root/'Images').exists())

    def test_original_tree_is_used_without_aliases(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'input';p=data/'Images/Photographs/a.jpg';p.parent.mkdir(parents=True);p.write_bytes(b'original')
            index={'Images/Photographs/a.jpg':file_hash(p)}
            self.assertEqual(layout.verified_image_view([data],index,root/'working',required_paths=list(index)),data)
            self.assertFalse((root/'Images').exists())

    def test_installed_encoder_gets_identical_pixels_and_features_through_view(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);canonical='Images/Photographs/a.jpg';original=root/'input/original'
            photo=original/canonical;photo.parent.mkdir(parents=True)
            Image.new('RGB',(25,31),(40,80,120)).save(photo)
            data,packaged=self.fixture(root,canonical,photo.read_bytes())
            models=tiny_encoders(root/'models')
            index={canonical:file_hash(photo)}
            view=layout.verified_image_view([data],index,root/'working',required_paths=[canonical])
            expected=encode_images([canonical],original,models['student'],'student',32,1,root/'expected',allow_debug=True)
            actual=encode_images([canonical],view,models['student'],'student',32,1,root/'actual',allow_debug=True)
            np.testing.assert_array_equal(expected,actual)

    def test_invalid_paths_and_existing_user_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);canonical='Images/Photographs/a.jpg';data,photo=self.fixture(root,canonical)
            with self.assertRaisesRegex(ValueError,'Invalid canonical'):
                layout.verified_image_view([data],{'../escape.jpg':file_hash(photo)},root/'working')
            folder=root/'Images';folder.mkdir();sentinel=folder/'user-owned.txt';sentinel.write_text('preserve')
            with self.assertRaisesRegex(FileExistsError,'not managed'):
                layout.verified_image_view([data],{canonical:file_hash(photo)},root/'working',required_paths=[canonical])
            self.assertEqual(sentinel.read_text(),'preserve')


if __name__=='__main__':
    unittest.main()
