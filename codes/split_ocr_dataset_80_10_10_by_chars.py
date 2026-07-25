from pathlib import Path
from collections import Counter
import shutil
import random
import csv


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    r"C:\Users\asus\OneDrive\Desktop\bao_cao_co_xo"
)

# Các file nhãn đầu vào sau khi đã gộp data thường + data ký tự hiếm
# Mỗi dòng có dạng:
# images/thu_muc/ten_anh.jpg<TAB>NHAN_BIEN_SO
#
# Sửa tên file ở đây cho đúng dữ liệu của bạn.
INPUT_LABEL_FILES = [
    BASE_DIR / "rec_gt_all.txt",
    BASE_DIR / "rec_gt_rare_all.txt",
]

# Nếu bạn chỉ có 1 file nhãn đã gộp sẵn, dùng như này:
# INPUT_LABEL_FILES = [
#     BASE_DIR / "rec_gt_merged_all.txt",
# ]

OUTPUT_TRAIN_DIR = BASE_DIR / "train_balanced_all"
OUTPUT_VAL_DIR = BASE_DIR / "val_all"
OUTPUT_TEST_DIR = BASE_DIR / "test_all"

OUT_TRAIN_LABEL = BASE_DIR / "rec_gt_train_balanced_all.txt"
OUT_VAL_LABEL = BASE_DIR / "rec_gt_val_all.txt"
OUT_TEST_LABEL = BASE_DIR / "rec_gt_test_all.txt"

OUT_CHAR_COUNT_CSV = BASE_DIR / "char_count_train_val_test.csv"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42


# ============================================================
# UTILS
# ============================================================

def clean_label(label: str) -> str:
    label = str(label).strip().upper()
    label = label.replace(" ", "")
    label = label.replace("-", "")
    label = label.replace(".", "")
    label = label.replace("_", "")
    label = label.replace("/", "")
    label = label.replace("\\", "")
    return label


def resolve_image_path(img_rel: str) -> Path:
    """
    Tìm ảnh theo đường dẫn trong file nhãn.
    Hỗ trợ cả đường dẫn tương đối và tuyệt đối.
    """

    img_rel = img_rel.strip()

    p = Path(img_rel)

    if p.is_absolute() and p.exists():
        return p

    candidates = [
        BASE_DIR / img_rel,
        BASE_DIR / "images" / img_rel,
        BASE_DIR.parent / img_rel,
    ]

    for c in candidates:
        if c.exists():
            return c

    return BASE_DIR / img_rel


def read_all_samples():
    samples = []
    seen = set()

    total_lines = 0
    invalid_lines = 0
    missing_images = 0
    empty_labels = 0
    duplicate_count = 0

    for label_file in INPUT_LABEL_FILES:
        print("\n" + "=" * 80)
        print(f"Đọc file nhãn: {label_file}")

        if not label_file.exists():
            print(f"[WARNING] Không tìm thấy file: {label_file}")
            continue

        with open(label_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                total_lines += 1

                if "\t" not in line:
                    invalid_lines += 1
                    print(f"[SKIP] Dòng sai định dạng {line_no}: {line}")
                    continue

                img_rel, label = line.split("\t", 1)
                img_rel = img_rel.strip()
                label = clean_label(label)

                if not label:
                    empty_labels += 1
                    continue

                img_abs = resolve_image_path(img_rel)

                if not img_abs.exists():
                    missing_images += 1
                    print(f"[MISSING] {img_abs}")
                    continue

                key = (str(img_abs.resolve()).lower(), label)

                if key in seen:
                    duplicate_count += 1
                    continue

                seen.add(key)

                samples.append({
                    "img_abs": img_abs,
                    "label": label
                })

    print("\n" + "=" * 80)
    print("KẾT QUẢ ĐỌC DỮ LIỆU")
    print(f"Tổng dòng đọc được       : {total_lines}")
    print(f"Số mẫu hợp lệ            : {len(samples)}")
    print(f"Dòng sai định dạng       : {invalid_lines}")
    print(f"Ảnh bị thiếu             : {missing_images}")
    print(f"Nhãn rỗng                : {empty_labels}")
    print(f"Mẫu trùng bị bỏ qua      : {duplicate_count}")

    return samples


def count_chars(samples):
    counter = Counter()

    for s in samples:
        counter.update(s["label"])

    return counter


def label_counter(label: str):
    return Counter(label)


# ============================================================
# SPLIT THEO PHÂN BỐ KÝ TỰ
# ============================================================

def split_by_char_distribution(samples):
    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    total_samples = len(samples)

    target_counts = {
        "train": round(total_samples * TRAIN_RATIO),
        "val": round(total_samples * VAL_RATIO),
    }
    target_counts["test"] = total_samples - target_counts["train"] - target_counts["val"]

    total_char_counter = count_chars(samples)

    target_char_counts = {
        split: {
            ch: total_char_counter[ch] * ratio
            for ch in total_char_counter
        }
        for split, ratio in {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO
        }.items()
    }

    # Ưu tiên xếp các mẫu có ký tự hiếm trước
    def rarity_score(sample):
        label = sample["label"]
        freqs = [total_char_counter[ch] for ch in label]
        return min(freqs), len(label)

    samples_sorted = sorted(samples, key=rarity_score)

    split_samples = {
        "train": [],
        "val": [],
        "test": [],
    }

    split_char_counters = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    def score_split(split_name, sample):
        current_n = len(split_samples[split_name])
        target_n = target_counts[split_name]

        # Nếu tập đã quá số lượng mục tiêu thì phạt mạnh
        count_over = max(0, current_n - target_n)
        count_deficit = max(0, target_n - current_n)

        sample_chars = label_counter(sample["label"])

        char_need_score = 0.0
        char_over_penalty = 0.0

        for ch, n in sample_chars.items():
            current_ch = split_char_counters[split_name][ch]
            target_ch = target_char_counts[split_name][ch]

            deficit = target_ch - current_ch

            if deficit > 0:
                char_need_score += min(deficit, n)
            else:
                char_over_penalty += abs(deficit) * 0.05

        score = 0.0
        score += char_need_score * 10.0
        score += count_deficit * 0.05
        score -= count_over * 100.0
        score -= char_over_penalty

        return score

    for sample in samples_sorted:
        scores = {
            split: score_split(split, sample)
            for split in ["train", "val", "test"]
        }

        best_split = max(scores, key=scores.get)

        split_samples[best_split].append(sample)
        split_char_counters[best_split].update(sample["label"])

    print("\n" + "=" * 80)
    print("KẾT QUẢ CHIA DỮ LIỆU")
    for split in ["train", "val", "test"]:
        print(
            f"{split.upper():<6}: "
            f"{len(split_samples[split])} ảnh, "
            f"{sum(split_char_counters[split].values())} ký tự"
        )

    return split_samples


# ============================================================
# COPY ẢNH VÀ GHI FILE NHÃN
# ============================================================

def safe_copy_samples(samples, output_dir: Path, output_label_file: Path, rel_prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    output_lines = []
    used_names = set()
    copied_count = 0
    duplicate_count = 0

    for idx, sample in enumerate(samples, start=1):
        old_path = sample["img_abs"]
        label = sample["label"]

        suffix = old_path.suffix.lower()
        if suffix not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            suffix = ".jpg"

        new_name = old_path.name

        if new_name in used_names:
            duplicate_count += 1
            new_name = f"{old_path.stem}_dup{duplicate_count:05d}{suffix}"

        used_names.add(new_name)

        new_path = output_dir / new_name
        shutil.copy2(old_path, new_path)

        new_rel = f"{rel_prefix}/{new_name}"
        output_lines.append(f"{new_rel}\t{label}\n")

        copied_count += 1

    with open(output_label_file, "w", encoding="utf-8") as f:
        f.writelines(output_lines)

    print(f"[OK] {output_label_file.name}: {copied_count} dòng")
    print(f"[OK] Thư mục ảnh: {output_dir}")


def write_outputs(split_samples):
    print("\n" + "=" * 80)
    print("GHI FILE ẢNH VÀ FILE NHÃN")

    safe_copy_samples(
        split_samples["train"],
        OUTPUT_TRAIN_DIR,
        OUT_TRAIN_LABEL,
        "train_balanced_all"
    )

    safe_copy_samples(
        split_samples["val"],
        OUTPUT_VAL_DIR,
        OUT_VAL_LABEL,
        "val_all"
    )

    safe_copy_samples(
        split_samples["test"],
        OUTPUT_TEST_DIR,
        OUT_TEST_LABEL,
        "test_all"
    )


# ============================================================
# THỐNG KÊ KÝ TỰ
# ============================================================

def write_char_count_csv(split_samples):
    counters = {
        split: count_chars(split_samples[split])
        for split in ["train", "val", "test"]
    }

    all_chars = set()
    for c in counters.values():
        all_chars.update(c.keys())

    rows = []

    for ch in sorted(all_chars):
        train_count = counters["train"].get(ch, 0)
        val_count = counters["val"].get(ch, 0)
        test_count = counters["test"].get(ch, 0)
        total = train_count + val_count + test_count

        rows.append({
            "char": ch,
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "total": total
        })

    rows = sorted(rows, key=lambda x: x["total"], reverse=True)

    with open(OUT_CHAR_COUNT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["char", "train", "val", "test", "total"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 80)
    print("THỐNG KÊ KÝ TỰ")
    print(f"{'Ký tự':<8}{'Train':>10}{'Val':>10}{'Test':>10}{'Tổng':>10}")

    for r in rows:
        print(
            f"{r['char']:<8}"
            f"{r['train']:>10}"
            f"{r['val']:>10}"
            f"{r['test']:>10}"
            f"{r['total']:>10}"
        )

    print("\n[OK] Đã lưu thống kê:")
    print(OUT_CHAR_COUNT_CSV)


# ============================================================
# MAIN
# ============================================================

def main():
    print("===== GỘP DATA + DATA KÝ TỰ HIẾM VÀ CHIA 80/10/10 =====")
    print(f"BASE_DIR: {BASE_DIR}")

    samples = read_all_samples()

    if not samples:
        print("Không có mẫu hợp lệ.")
        return

    print("\nTổng số ảnh hợp lệ trước khi chia:", len(samples))
    print("Tổng số ký tự trước khi chia:", sum(count_chars(samples).values()))

    split_samples = split_by_char_distribution(samples)

    write_outputs(split_samples)

    write_char_count_csv(split_samples)

    print("\n===== DONE =====")
    print("Kết quả gồm:")
    print(OUTPUT_TRAIN_DIR)
    print(OUTPUT_VAL_DIR)
    print(OUTPUT_TEST_DIR)
    print(OUT_TRAIN_LABEL)
    print(OUT_VAL_LABEL)
    print(OUT_TEST_LABEL)
    print(OUT_CHAR_COUNT_CSV)


if __name__ == "__main__":
    main()