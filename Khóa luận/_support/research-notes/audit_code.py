"""Read-only structural audit of the local COde release; emits aggregate counts."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
DATA_PARENT = WORKSPACE if (WORKSPACE / "Data").is_dir() else WORKSPACE.parent
ROOT = DATA_PARENT / "Data" / "COde-Dataset (1)"


def main():
    with (ROOT / "complete_dataset.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    patients_by_split = defaultdict(set)
    for row in rows:
        patients_by_split[row["split"]].add(row["patient_id"])
    report = {
        "csv_rows": len(rows),
        "unique_checkups": len({r["checkup_id"] for r in rows}),
        "unique_patients": len({r["patient_id"] for r in rows}),
        "csv_split_rows": dict(Counter(r["split"] for r in rows)),
        "csv_train_test_patient_overlap": len(patients_by_split["train"] & patients_by_split["test"]),
        "empty_fields": {
            k: sum(not r[k].strip() for r in rows)
            for k in ["photographs", "radiographs", "chief_complaint", "present_illness",
                      "past_medical_record", "examination", "radiographs_examination", "anomalies_en"]
        },
    }
    image_stats = {}
    for field, folder in [("photographs", "Photographs"), ("radiographs", "Radiographs")]:
        actual = {p.name for p in (ROOT / "Images" / folder).glob("*.jpg")}
        references = [x.strip() for r in rows for x in r[field].split(",") if x.strip()]
        counts = Counter(references)
        image_stats[folder] = {
            "jpg_files": len(actual), "csv_references": len(references),
            "unique_csv_references": len(counts),
            "missing_referenced_files": len(set(references) - actual),
            "unreferenced_files": len(actual - set(references)),
            "filenames_referenced_more_than_once": sum(n > 1 for n in counts.values()),
        }
    report["images"] = image_stats
    checkups, patients = {}, {}
    for filename in ["train-cls.json", "test_cls.json"]:
        data = json.loads((ROOT / filename).read_text(encoding="utf-8"))
        case_sets = [{Path(x).name.rsplit("-", 1)[0] for x in s["images"]} for s in data]
        checkups[filename] = set().union(*case_sets)
        patients[filename] = {c.split("-")[0] for c in checkups[filename]}
        report[filename] = {
            "rows": len(data), "unique_checkups_from_image_paths": len(checkups[filename]),
            "unique_patients_from_image_paths": len(patients[filename]),
            "rows_without_images": sum(not s for s in case_sets),
            "rows_with_multiple_checkup_ids": sum(len(s) > 1 for s in case_sets),
            "rows_with_multilabel_answer_comma": sum(
                "," in m["content"] for s in data for m in s["messages"] if m["role"] == "assistant"
            ),
        }
    report["json_train_test_checkup_overlap"] = len(checkups["train-cls.json"] & checkups["test_cls.json"])
    report["json_train_test_patient_overlap"] = len(patients["train-cls.json"] & patients["test_cls.json"])
    labels = Counter(x.strip() for r in rows for x in set(r["anomalies_en"].split(",")) if x.strip())
    report["top_20_record_level_labels"] = labels.most_common(20)
    # Illustrative mapping only: it must be reviewed before becoming a task definition.
    mapping = {
        "Dental Caries": "Caries", "Gingivitis": "Gingivitis", "Pulpitis": "Pulpitis",
        "Tooth Loss": "Tooth Loss", "Tooth Structure Loss": "Tooth Structure Loss",
        "Malocclusion": "Malocclusion", "Class I Malocclusion": "Malocclusion",
        "Class II Malocclusion": "Malocclusion", "Class III Malocclusion": "Malocclusion",
    }
    mapped_counts = Counter()
    for row in rows:
        mapped = {mapping[x.strip()] for x in row["anomalies_en"].split(",") if x.strip() in mapping}
        mapped_counts[len(mapped)] += 1
    report["provisional_six_target_count_per_record"] = dict(sorted(mapped_counts.items()))
    report["limitations"] = [
        "Structural audit only: no image decoding, file hash or perceptual duplicate checks.",
        "Empty strings counted; nonempty missing-value placeholders were not normalized.",
        "JSON grouping inferred from filenames, not confirmed by a separate encounter table.",
        "Provisional label mapping is not clinically validated; absent labels are not confirmed negatives.",
    ]
    output = Path(__file__).with_name("code_audit_summary.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
