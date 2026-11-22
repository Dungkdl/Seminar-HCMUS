"""Package selected Run 5 train/validation radiographs locally for private Kaggle use."""
import argparse
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from clindistill.artifacts import verify_release
from clindistill.research.data import resolve_image_sources
from run5_xray import paired_records, selected_xrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    args = parser.parse_args()
    verify_release(args.prepared_root, splits=('train', 'validation'))
    records = paired_records(args.prepared_root)
    paths = sorted({path for rows in records.values() for row in rows for path in selected_xrays(row)})
    sources = resolve_image_sources(paths, args.image_root)
    output = Path(__file__).resolve().parents[1] / 'dist/run4-run5/run5-radiographs-private.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.tmp')
    with ZipFile(tmp, 'w', compression=ZIP_STORED) as archive:
        for path in paths:
            archive.write(sources[path], path)
    with ZipFile(tmp) as archive:
        if archive.testzip() is not None:
            raise ValueError('Radiograph ZIP integrity check failed.')
    tmp.replace(output)
    print(f'{len(paths):,} selected train/validation X-ray files archived locally at {output}')


if __name__ == '__main__':
    main()
