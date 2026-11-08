"""Map Qwen's SHA1-prefixed filenames to canonical CODe paths, without changing bytes."""


def verified_image_view(image_roots, image_index, workspace, *, required_paths=None):
    """Create audited filesystem aliases usable by the installed 0.5.1 wheel.

    Only byte-identical photos are admitted. The canonical root is the common
    ancestor of inputs and working storage, keeping symlink targets inside the
    existing wheel's path boundary. No installed functions or manifests change.
    """
    from pathlib import Path, PurePosixPath
    import hashlib
    import json
    import os

    roots = [Path(p).resolve() for p in image_roots]
    work = Path(workspace).resolve()
    if not roots or not image_index:
        raise ValueError('Image roots and original image_index.json are required.')
    required = set() if required_paths is None else set(required_paths)
    if not required <= set(image_index):
        raise ValueError('Required photos are absent from the original image index.')
    for name, digest in image_index.items():
        parts = PurePosixPath(name).parts
        if (len(parts) != 3 or parts[:2] != ('Images', 'Photographs')
                or '\\' in name or '..' in parts or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Invalid canonical photo path or SHA256 in image index.')
    if all((roots[0] / name).is_file() for name in (required or image_index)):
        return roots[0]  # Original CODe tree: prepare_features performs the hash verification.

    def sha256(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()

    common = Path(os.path.commonpath([str(p) for p in roots] + [str(work)]))
    if common == Path(common.anchor):
        raise ValueError('Inputs and workspace need a shared parent below the filesystem root, e.g. /kaggle.')
    work.mkdir(parents=True, exist_ok=True)
    report_path = work / 'image_layout_report.json'
    mapping, missing, mismatched, hashes = {}, [], [], {}
    # Map all available indexed photos, so adding a locked test shard needs no new
    # layout policy. Strict coverage applies to the requested train/validation set.
    for number, name in enumerate(sorted(image_index), 1):
        packaged = hashlib.sha1(name.encode('utf-8')).hexdigest()[:10] + '_' + PurePosixPath(name).name
        candidates = []
        for root in roots:
            for candidate in (root / name, root / 'images' / packaged, root / packaged,
                              root / 'images' / PurePosixPath(name).name, root / PurePosixPath(name).name):
                if not candidate.is_file():
                    continue
                source = candidate.resolve()
                if root not in source.parents or common not in source.parents:
                    raise ValueError('Packaged image escapes the selected input root.')
                if source not in candidates:
                    candidates.append(source)
        selected = None
        for source in candidates:
            if source not in hashes:
                hashes[source] = sha256(source)
            if hashes[source] == image_index[name]:
                selected = source
                break
        if selected is None:
            (mismatched if candidates else missing).append(name)
        else:
            mapping[name] = str(selected)
        if number % 2000 == 0:
            print(f'Image layout: verified {number:,}/{len(image_index):,} canonical photos', flush=True)

    absent = sorted(required - set(mapping))
    proof = hashlib.sha256(json.dumps(image_index, sort_keys=True).encode()).hexdigest()
    report = {'schema_version': 1, 'policy': 'sha1(canonical_path)[:10]_basename; original SHA256 required',
              'image_index_identity': proof, 'source_roots': list(map(str, roots)),
              'canonical_root': str(common), 'indexed': len(image_index), 'mapped': len(mapping),
              'required': len(required), 'missing_required': absent,
              'unavailable_indexed': missing, 'mismatched_indexed': mismatched, 'mapping': mapping,
              'status': 'missing_required' if absent else 'coverage_verified'}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if absent:
        raise FileNotFoundError(f'{len(absent)} required photos are missing or differ from original bytes. '
                                f'First: {absent[0]}. Attach the CODe image supplement or original photos. '
                                f'Coverage report: {report_path}')

    anchor = common / 'Images'
    marker = anchor / '.clindistill-image-layout.json'
    if anchor.exists():
        if anchor.is_symlink() or not marker.is_file():
            raise FileExistsError('Canonical Images directory already exists and is not managed by this adapter.')
        info = json.loads(marker.read_text(encoding='utf-8'))
        if info.get('image_index_identity') != proof:
            raise ValueError('Existing canonical image view belongs to another image index. Use a fresh session.')
    else:
        anchor.mkdir()
        marker.write_text(json.dumps({'image_index_identity': proof}) + '\n', encoding='utf-8')

    counts = {'symlink': 0, 'hardlink': 0, 'reused': 0}
    for name, source in mapping.items():
        dest, source = common / name, Path(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.exists():
            if not dest.is_file() or not os.path.samefile(dest, source):
                raise FileExistsError('Existing canonical alias points to another image; use a fresh session.')
            counts['reused'] += 1
            continue
        try:
            dest.symlink_to(source)
            counts['symlink'] += 1
        except OSError:
            # Windows regression checks can use hardlinks on the same volume.
            # On Kaggle, symlinks connect read-only Input files to the canonical view.
            os.link(source, dest)
            counts['hardlink'] += 1
    report['aliases'] = counts
    report['status'] = 'verified'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'IMAGE LAYOUT READY: {len(mapping):,} byte-verified photos; root={common}; report={report_path}', flush=True)
    return common
