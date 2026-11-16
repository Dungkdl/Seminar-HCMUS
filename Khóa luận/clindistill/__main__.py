"""Current frozen-feature research CLI; archived trainer requires explicit `legacy`."""
import argparse
import contextlib
import json
import sys

from .artifacts import read_json


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'legacy':
        from .legacy_cli import main as legacy_main
        previous = sys.argv
        try:
            sys.argv = [previous[0] + ' legacy', *argv[1:]]
            return legacy_main()
        finally:
            sys.argv = previous
    parser = argparse.ArgumentParser(description='ClinDistill selected-cohort, frozen-feature research')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'train'):
        cmd = commands.add_parser(name)
        cmd.add_argument('--config', required=True, help='research_visual/clinical/transfer.json, or registered full profile')
        if name == 'train':
            cmd.add_argument('--features', required=True)
            cmd.add_argument('--study', required=True)
            cmd.add_argument('--allow-cpu-debug', action='store_true')
    cmd = commands.add_parser('features')
    cmd.add_argument('--config', required=True)
    cmd.add_argument('--prepared', required=True)
    cmd.add_argument('--images', required=True)
    cmd.add_argument('--models-json', required=True, help='JSON mapping student/teacher/clinical and optional CNN roles to local assets')
    cmd.add_argument('--output', required=True)
    cmd.add_argument('--splits', nargs='+', choices=['train', 'validation', 'test'], default=['train', 'validation'])
    cmd.add_argument('--test-protocol')
    for name in ('evaluate', 'lock-test', 'export-student'):
        cmd = commands.add_parser(name)
        cmd.add_argument('--features', required=True)
        cmd.add_argument('--study', required=True)
        if name == 'lock-test':
            cmd.add_argument('--review-record', required=True)
        elif name == 'evaluate':
            cmd.add_argument('--output')
            cmd.add_argument('--split', choices=['validation', 'test'], default='validation')
            cmd.add_argument('--review-record')
        else:
            cmd.add_argument('--encoder', required=True)
            cmd.add_argument('--output', required=True)
            cmd.add_argument('--method')
    cmd = commands.add_parser('review-status')
    cmd.add_argument('--prepared', required=True)
    cmd.add_argument('--audit')
    args = parser.parse_args(argv)
    if args.command in {'plan', 'train'}:
        from .research.training import validate_config, study_plan, run_study
        config = read_json(args.config)
        validate_config(config)
        if args.command == 'plan':
            result = study_plan(config)
        else:
            if args.allow_cpu_debug and not config.get('debug'):
                parser.error('--allow-cpu-debug requires a synthetic/debug configuration.')
            with contextlib.redirect_stdout(sys.stderr):
                table = run_study(args.features, args.study, config, require_gpu=not args.allow_cpu_debug)
            result = table.astype(object).where(table.notna(), None).to_dict('records')
    elif args.command == 'features':
        from .research.data import prepare_features
        with contextlib.redirect_stdout(sys.stderr):
            result = prepare_features(args.prepared, args.images, read_json(args.models_json), args.output,
                read_json(args.config), splits=args.splits,
                test_protocol=read_json(args.test_protocol) if args.test_protocol else None)
    elif args.command == 'review-status':
        from .research.review import review_status
        result = review_status(args.prepared, args.audit)
    else:
        from .research.reporting import evaluate_study, freeze_test_protocol, export_student
        if args.command == 'lock-test':
            result = freeze_test_protocol(args.features, args.study, args.review_record)
        elif args.command == 'export-student':
            result = export_student(args.features, args.study, args.encoder, args.output, method=args.method)
        else:
            table, _ = evaluate_study(args.features, args.study, args.output,
                split=args.split, test_review_record=args.review_record)
            result = table.astype(object).where(table.notna(), None).to_dict('records')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
