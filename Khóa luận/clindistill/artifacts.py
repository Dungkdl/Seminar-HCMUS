"""Versioned artifact contracts; no torch dependency in preparation."""
import hashlib
import json
import shutil
from pathlib import Path


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def code_hash():
    root = Path(__file__).parent
    return object_hash({p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*.py'))})


def publish_release(audit_dir, destination):
    """Copy audited manifests into an immutable, experimental data release.

    Repeating the same export is safe; modified inputs require a new destination.
    This does not certify annotation quality or clear manual review flags.
    """
    source, dest = Path(audit_dir), Path(destination)
    audit = read_json(source / 'run_summary.json')
    config = read_json(source / 'experiment_config.json')
    assert audit['split_fingerprint'] == config['split_fingerprint']
    names = ['experiment_config.json', 'label_policy.json', 'split_lock.json', 'split_assignments.csv',
             'split_summary.csv', 'label_support_by_split.csv', 'run_summary.json', 'REPORT.md']
    names += [f'{kind}_{split}.jsonl' for kind in ['student_single', 'student_multilabel_candidate', 'teacher_context']
              for split in ['train', 'validation', 'test']]
    # run_summary has a timestamp: identity is based on semantic protocol + training inputs.
    hashes = {name: file_hash(source / name) for name in names}
    identity = {k: v for k, v in hashes.items() if k not in {'run_summary.json', 'REPORT.md'}}
    release_id = object_hash(identity)
    if (dest / 'release.json').exists():
        previous = verify_release(dest)
        if previous['release_id'] != release_id:
            raise ValueError('Prepared inputs changed. Choose a new release directory; existing release is immutable.')
        return previous
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('Destination contains files but no complete release. Choose a fresh release directory.')
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(source / name, dest / name)
    release = {'schema_version': 1, 'dataset': 'COde', 'release_id': release_id,
               'split_fingerprint': config['split_fingerprint'], 'stage': 'experimental',
               'class_names': config['class_names'], 'file_checksums': hashes,
               'pending_reviews': [c for c in audit['checks'] if c['status'] != 'PASS'],
               'annotation_note': 'Single-target recorded conditions; not complete clinical ground truth.'}
    write_json(dest / 'release.json', release)
    return verify_release(dest)


def verify_release(path, splits=None):
    """Verify release metadata and selected partitions without opening test in training."""
    path = Path(path)
    release = read_json(path / 'release.json')
    if release.get('schema_version') != 1 or release.get('dataset') != 'COde':
        raise ValueError('Unsupported prepared-data schema or dataset.')
    for name, expected in release['file_checksums'].items():
        if splits is not None and name.endswith('.jsonl') and not any(name.endswith(f'_{s}.jsonl') for s in splits):
            continue
        if Path(name).name != name or file_hash(path / name) != expected:
            raise ValueError(f'Prepared artifact checksum mismatch: {name}')
    return release
