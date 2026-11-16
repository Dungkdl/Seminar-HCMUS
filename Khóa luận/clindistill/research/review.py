"""Read-only research readiness; completed software never certifies expert review."""
import csv
from pathlib import Path

from ..artifacts import read_json

TASK = {
    'unit': 'visit with photographs and exactly one recorded mapped target',
    'classes': 'six recorded dental target groups; no healthy or out-of-scope class',
    'negative_labels': 'unrecorded conditions are not verified negatives',
    'inference_scope': 'selected-cohort classification, not unrestricted dental diagnosis',
    'encoder_training': 'frozen features; MIL/projection/fusion/head only',
    'reliability': 'OOF model-based sample weighting; not field selection or clinical visual sufficiency',
}


def readiness(metadata, config):
    pending = metadata.get('pending_reviews', ['Review status not supplied.'])
    policy = metadata.get('request', {}).get('policy', {})
    # Complete negative annotations are not required for the registered single-target task.
    out_of_scope = [p for p in pending if isinstance(p, dict) and
                    p.get('check') == 'label completeness for six-target multi-label training']
    relevant = [p for p in pending if p not in out_of_scope]
    blockers = []
    if relevant:
        blockers.append('Prepared release retains unresolved expert/data reviews.')
    if metadata.get('release_stage') != 'reviewed':
        blockers.append('Prepared release has not been independently reviewed and promoted.')
    if metadata.get('debug'):
        blockers.append('Synthetic/debug features cannot support research results.')
    uses_clinical = ('teacher_multimodal' in config.get('teachers', ['teacher_visual', 'teacher_multimodal'])
                     or any(m.startswith('teacher_') and m != 'teacher_visual' for m in config['methods']))
    if uses_clinical and not (policy.get('temporal_verified') and policy.get('review_record')):
        blockers.append('Clinical timing/content remains exploratory.')
    return {'study_mode': config.get('study_mode', 'exploratory'), 'task': TASK,
            'pending_reviews': pending, 'out_of_scope_reviews': out_of_scope,
            'confirmatory_ready': not blockers,
            'blockers': blockers, 'publication_ready': False,
            'note': 'Readiness does not establish model quality, novelty, clinical safety or external validity.'}


def review_status(prepared_root, audit_root=None):
    """Aggregate decisions only; never include case IDs or clinical text in output."""
    root = Path(prepared_root)
    release = read_json(root / 'release.json')
    result = {'release_id': release['release_id'], 'release_stage': release.get('stage'),
              'pending_reviews': release.get('pending_reviews', ['Review status not supplied.']),
              'task': TASK, 'review_files': {}, 'expert_reviews_verified': False}
    audit = Path(audit_root) if audit_root else root
    specs = {
        'clinical_label_review_train.csv': ('reviewer', 'label_mapping_correct',
            'annotation_complete_for_six_targets', 'post_diagnosis_leakage', 'visual_grounding_review'),
        'cross_split_near_duplicate_review.csv': ('review_decision',),
    }
    for filename, columns in specs.items():
        path = audit / filename
        if not path.is_file():
            result['review_files'][filename] = {'status': 'not_supplied', 'rows': None}
            continue
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not set(columns) <= set(reader.fieldnames or []):
                raise ValueError(f'Review CSV is missing required decision columns: {filename}')
            rows = list(reader)
        filled = {c: sum(bool(r[c].strip()) for r in rows) for c in columns}
        complete = sum(all(r[c].strip() for c in columns) for r in rows)
        result['review_files'][filename] = {'rows': len(rows), 'completed_rows': complete,
            'pending_rows': len(rows) - complete, 'filled_by_column': filled,
            'status': 'decisions_recorded_need_adjudication' if complete == len(rows) else 'review_required'}
    result['next_actions'] = [
        'Have qualified reviewers complete and adjudicate annotation/clinical decisions.',
        'Adjudicate duplicate candidates; feed confirmed links into a NEW prepared split version.',
        'Record provenance and regenerate the release after corrections; never clear flags by editing a report.',
        'Use exploratory runs while reviews remain open; lock model/config before any test evaluation.']
    return result
