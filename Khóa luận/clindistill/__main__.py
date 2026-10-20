import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='ClinDistill: preparation, Teachers, KD and evaluation')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('export')
    prepare.add_argument('--audit-dir', required=True)
    prepare.add_argument('--destination', required=True)
    cache = commands.add_parser('clinical-cache')
    cache.add_argument('--release-dir', required=True)
    cache.add_argument('--destination', required=True)
    cache.add_argument('--encoder', default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    cache.add_argument('--revision', default='main')
    cache.add_argument('--include-test', action='store_true')
    cache.add_argument('--local-files-only', action='store_true')
    for name in ['train', 'evaluate', 'suite']:
        p = commands.add_parser(name)
        p.add_argument('--release-dir', required=True)
        p.add_argument('--image-root', required=True)
        p.add_argument('--run-dir', required=True)
        p.add_argument('--device', default=None)
        p.add_argument('--clinical-cache', default=None)
        if name in {'train', 'suite'}:
            p.add_argument('--config', required=True)
            p.add_argument('--resume', action='store_true')
            if name == 'train':
                p.add_argument('--teacher-root', default=None, help='Contains teacher_full/seed_N, teacher_v/seed_N, etc.')
            else:
                p.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
                p.add_argument('--experiments', nargs='+', default=None)
        else:
            p.add_argument('--split', choices=['validation', 'test'], default='validation')
            p.add_argument('--allow-experimental-test', action='store_true')
    args = parser.parse_args()
    if args.command == 'export':
        from .artifacts import publish_release
        result = publish_release(args.audit_dir, args.destination)
    elif args.command == 'clinical-cache':
        from .clinical import prepare_clinical_cache
        result = prepare_clinical_cache(args.release_dir, args.destination, encoder=args.encoder,
            revision=args.revision, splits=('train', 'validation', 'test') if args.include_test else ('train', 'validation'),
            local_files_only=args.local_files_only)
    elif args.command in {'train', 'suite'}:
        from pathlib import Path
        from .config import TrainConfig
        from .engine import fit
        config = TrainConfig.load(args.config)
        if args.command == 'suite':
            from .suite import run_suite
            result = run_suite(args.release_dir, args.image_root, args.run_dir, config,
                clinical_cache=args.clinical_cache, selected=args.experiments,
                seeds=args.seeds, resume=args.resume, device=args.device)
        else:
            teachers = ({g: Path(args.teacher_root) / f'teacher_{g}' / f'seed_{config.seed}'
                         for g in ['clinical', 'full', 'v', 'h']} if args.teacher_root else None)
            result = fit(args.release_dir, args.image_root, args.run_dir, config,
                         resume=args.resume, device=args.device, clinical_cache=args.clinical_cache,
                         teacher_dirs=teachers)
    else:
        from .engine import evaluate_checkpoint
        result = evaluate_checkpoint(args.release_dir, args.image_root, args.run_dir, split=args.split,
                                     allow_experimental_test=args.allow_experimental_test, device=args.device,
                                     clinical_cache=args.clinical_cache)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
