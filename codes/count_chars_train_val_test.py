from pathlib import Path
from collections import Counter
import csv


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\asus\OneDrive\Desktop\BÁO CÁO ĐỒ ÁN\PhanXuânDương-2386400966\dataset\dataset_done_ocr")

LABEL_FILES = {
    "train": BASE_DIR / "rec_gt_train_balanced_all.txt",
    "val": BASE_DIR / "rec_gt_val_all.txt",
    "test": BASE_DIR / "rec_gt_test_all.txt",
}

OUTPUT_CSV = BASE_DIR / "char_count_train_val_test.csv"


# ============================================================
# FUNCTIONS
# ============================================================

def read_labels(label_file: Path):
    labels = []

    if not label_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {label_file}")

    with open(label_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            if "\t" not in line:
                print(f"[SKIP] Dòng {line_no} sai định dạng: {line}")
                continue

            _, label = line.split("\t", 1)
            label = label.strip().upper()

            if label:
                labels.append(label)

    return labels


def count_chars(labels):
    counter = Counter()

    for label in labels:
        for ch in label:
            counter[ch] += 1

    return counter


def main():
    print("===== THỐNG KÊ KÝ TỰ TRAIN / VAL / TEST =====")
    print(f"BASE_DIR: {BASE_DIR}")

    split_counters = {}
    split_label_counts = {}
    split_char_totals = {}

    all_chars = set()

    for split, label_file in LABEL_FILES.items():
        labels = read_labels(label_file)
        counter = count_chars(labels)

        split_counters[split] = counter
        split_label_counts[split] = len(labels)
        split_char_totals[split] = sum(counter.values())

        all_chars.update(counter.keys())

        print("\n" + "=" * 70)
        print(split.upper())
        print(f"File nhãn hợp lệ : {len(labels)}")
        print(f"Tổng số ký tự    : {sum(counter.values())}")
        print("Chi tiết ký tự:")

        for ch, count in counter.most_common():
            print(f"{ch}: {count}")

    rows = []

    for ch in sorted(all_chars):
        train_count = split_counters["train"].get(ch, 0)
        val_count = split_counters["val"].get(ch, 0)
        test_count = split_counters["test"].get(ch, 0)
        total_count = train_count + val_count + test_count

        rows.append({
            "Ký tự": ch,
            "Train": train_count,
            "Val": val_count,
            "Test": test_count,
            "Tổng": total_count
        })

    rows = sorted(rows, key=lambda x: x["Tổng"], reverse=True)

    print("\n" + "=" * 70)
    print("TỔNG HỢP TRAIN / VAL / TEST")
    print(f"{'Ký tự':<8}{'Train':>10}{'Val':>10}{'Test':>10}{'Tổng':>10}")

    for row in rows:
        print(
            f"{row['Ký tự']:<8}"
            f"{row['Train']:>10}"
            f"{row['Val']:>10}"
            f"{row['Test']:>10}"
            f"{row['Tổng']:>10}"
        )

    print("\n" + "=" * 70)
    print("TỔNG SỐ MẪU VÀ KÝ TỰ")
    print(f"Train - số dòng nhãn: {split_label_counts['train']}, số ký tự: {split_char_totals['train']}")
    print(f"Val   - số dòng nhãn: {split_label_counts['val']}, số ký tự: {split_char_totals['val']}")
    print(f"Test  - số dòng nhãn: {split_label_counts['test']}, số ký tự: {split_char_totals['test']}")

    total_labels = sum(split_label_counts.values())
    total_chars = sum(split_char_totals.values())

    print(f"Tổng số dòng nhãn: {total_labels}")
    print(f"Tổng số ký tự    : {total_chars}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Ký tự", "Train", "Val", "Test", "Tổng"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nĐã lưu file CSV:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()