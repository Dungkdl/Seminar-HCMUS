"""Package original cohort photos absent from original-paper classification references."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package(prepared, originals, index_archive, output, max_photos=8):
    prepared, originals, output = Path(prepared), Path(originals).resolve(), Path(output)
    release = json.loads((prepared/'release.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(index_archive) as z:
        index = json.loads(z.read('image_index.json'))
    if index['release_id'] != release['release_id']:
        raise ValueError('Image index belongs to another prepared release.')
    paper = set()
    for name in ('train-cls.json', 'test_cls.json'):
        for row in json.loads((originals/name).read_text(encoding='utf-8')):
            paper.update(row['images'])
    needed = set()
    for split in ('train', 'validation', 'test'):
        for line in (prepared/f'student_single_{split}.jsonl').read_text(encoding='utf-8').splitlines():
            paths = sorted(json.loads(line)['photo_paths'])
            if len(paths) > max_photos:
                positions = [round(i*(len(paths)-1)/(max_photos-1)) for i in range(max_photos)] if max_photos > 1 else [0]
                paths = [paths[i] for i in positions]
            needed.update(paths)
    missing = sorted(needed-paper)
    manifest = {'schema_version':1, 'release_id':release['release_id'], 'contains_raw_images':True,
                'selection':'Prepared canonical photo policy; absent from union of original paper cls references.',
                'max_photos':max_photos, 'photos':{}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name in missing:
            photo = (originals/name).resolve()
            if originals not in photo.parents:
                raise ValueError('Unsafe original image path.')
            content = photo.read_bytes(); digest = hashlib.sha256(content).hexdigest()
            if digest != index['images'][name]:
                raise ValueError('Original image bytes differ from audited image index.')
            target = 'images/'+hashlib.sha1(name.encode()).hexdigest()[:10]+'_'+Path(name).name
            z.writestr(target,content)
            manifest['photos'][name] = {'path':target,'sha256':digest}
        z.writestr('code_image_supplement.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            raise ValueError('Image supplement ZIP failed validation.')
    print(output, 'photos:',len(missing),'bytes:',output.stat().st_size)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root',type=Path,required=True)
    parser.add_argument('--original-root',type=Path,required=True)
    parser.add_argument('--image-index-archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-photos',type=int,default=8)
    args=parser.parse_args()
    if args.max_photos<1:
        parser.error('--max-photos must be positive')
    package(args.prepared_root,args.original_root,args.image_index_archive,args.output,args.max_photos)
