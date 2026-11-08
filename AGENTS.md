# Repository workflow

- Resolve a problem through an existing or new GitHub issue, a dedicated branch, implementation and relevant checks, then a PR linked to the issue.
- Push implementation branches. Merge only after the user approves or explicitly authorizes merging. Do not close an umbrella issue when the PR addresses only part of it.
- Keep original-paper Qwen reproduction separate from the ClinDistill research pipeline. Preserve archived baseline notebooks unless the task explicitly includes them.
- Keep CODe images, patient records, clinical vectors, raw predictions, weights and checkpoints outside Git. Local generated bundles belong in ignored `_support/dist/`.
- Distinguish synthetic software checks, actual pretrained experiments, and clinical/expert review. Never mark unresolved annotation or temporal-alignment reviews as passed automatically.
