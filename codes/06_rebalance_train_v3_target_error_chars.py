from pathlib import Path
from collections import Counter
import random
import csv
import shutil

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\asus\OneDrive\Desktop\bao_cao_co_xo")
PADDLE_DIR = BASE_DIR / "data_test 2" / "paddle_rec_new"

# Dùng file gốc sau khi gộp rare, KHÔNG dùng file cân bằng cũ
INPUT_TRAIN_FILE = PADDLE_DIR / "rec_gt_train_final.txt"

# Output train mới
OUTPUT_TRAIN_FILE = PADDLE_DIR / "rec_gt_train_balanced_all.txt"
OUTPUT_IMAGE_DIR = PADDLE_DIR / "images" / "train_balanced_all"

# Ảnh augment raw
AUG_RAW_DIR = PADDLE_DIR / "images" / "train_rebalanced_v3_target_aug_raw"

# Reports
REPORT_DIR = PADDLE_DIR / "reports"
REPORT_CHAR_COUNT = REPORT_DIR / "char_count_rebalanced_v3_target.csv"
REPORT_SAMPLE = REPORT_DIR / "rebalance_v3_target_sample_report.csv"
REPORT_AUG = REPORT_DIR / "rebalance_v3_target_augmented.csv"

RANDOM_SEED = 42

# Nhóm ký tự cần tăng theo lỗi test bạn vừa thống kê
TARGET_CHARS = set(["U", "V", "Z", "N", "K", "B", "E", "G", "M"])

# Target đề xuất
# M rất lỗi cao 28%, G/E cũng cao, nên kéo cao hơn
TARGET_COUNTS = {
    "M": 300,
    "G": 400,
    "E": 400,
    "B": 900,
    "K": 950,
    "U": 250,
    "V": 250,
    "Z": 250,
    "N": 250,
}

# Không augment quá nhiều từ cùng một ảnh để giảm overfit
MAX_AUG_PER_ORIGINAL = 10

# Resize padding
TARGET_W = 320
TARGET_H = 48
PAD_COLOR = 255

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def clean_label(label: str) -> str:
    label = str(label).upper().strip()
    for ch in [" ", "-", ".", "_", "/", "\\"]:
        label = label.replace(ch, "")
    return label


def read_label_file(path: Path):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()

            if not line or "\t" not in line:
                continue

            img_rel, label = line.split("\t", 1)
            label = clean_label(label)

            samples.append({
                "line_id": line_id,
                "img_rel": img_rel,
                "label": label,
                "type": "original"
            })

    return samples


def count_chars(samples):
    counter = Counter()

    for s in samples:
        counter.update(s["label"])

    return counter


def get_target_chars(label: str):
    return sorted(set(label) & TARGET_CHARS)


def has_target_char(label: str):
    return len(get_target_chars(label)) > 0


def image_abs(img_rel: str):
    return PADDLE_DIR / img_rel


def target_for_char(ch: str):
    return TARGET_COUNTS.get(ch, 0)


def print_target_summary(title, counter):
    print("\n" + title)
    print("=" * 70)

    for ch in sorted(TARGET_CHARS):
        print(f"{ch}: {counter.get(ch, 0)} / target={TARGET_COUNTS.get(ch, 0)}")

    print("\nMột số ký tự phổ biến để theo dõi:")
    for ch in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "H"]:
        print(f"{ch}: {counter.get(ch, 0)}")


# ============================================================
# AUGMENTATION FUNCTIONS
# ============================================================

def aug_hsv(img):
    """
    Color space augmentation nhẹ.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    h_shift = random.randint(-5, 5)
    s_scale = random.uniform(0.80, 1.20)
    v_scale = random.uniform(0.80, 1.20)

    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_brightness_contrast(img):
    alpha = random.uniform(0.82, 1.22)
    beta = random.randint(-18, 18)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def aug_blur(img):
    if random.random() < 0.30:
        return cv2.GaussianBlur(img, (3, 3), 0)
    return img


def aug_noise(img):
    if random.random() < 0.30:
        sigma = random.uniform(2, 7)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)
    return img


def aug_rotate(img):
    h, w = img.shape[:2]
    angle = random.uniform(-2.5, 2.5)

    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


def aug_perspective_light(img):
    """
    Perspective rất nhẹ, tránh làm biến dạng quá mạnh.
    """
    if random.random() > 0.35:
        return img

    h, w = img.shape[:2]

    max_shift_x = max(1, int(w * 0.025))
    max_shift_y = max(1, int(h * 0.035))

    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ])

    dst = np.float32([
        [random.randint(0, max_shift_x), random.randint(0, max_shift_y)],
        [w - 1 - random.randint(0, max_shift_x), random.randint(0, max_shift_y)],
        [w - 1 - random.randint(0, max_shift_x), h - 1 - random.randint(0, max_shift_y)],
        [random.randint(0, max_shift_x), h - 1 - random.randint(0, max_shift_y)],
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


def augment_image(img):
    """
    Augmentation nhẹ, vì mục tiêu là tăng đa dạng chứ không làm ảnh quá khác thật.
    """
    aug = img.copy()

    ops = [
        aug_hsv,
        aug_brightness_contrast,
        aug_rotate,
        aug_blur,
        aug_noise,
        aug_perspective_light,
    ]

    random.shuffle(ops)

    n_ops = random.randint(2, 4)

    for op in ops[:n_ops]:
        aug = op(aug)

    return aug


# ============================================================
# TARGETED AUGMENTATION
# ============================================================

def needs_more(counter):
    for ch, target in TARGET_COUNTS.items():
        if counter.get(ch, 0) < target:
            return True
    return False


def choose_aug_source(samples, counter, aug_count_by_line):
    deficits = {}

    for ch, target in TARGET_COUNTS.items():
        cur = counter.get(ch, 0)
        if cur < target:
            deficits[ch] = target - cur

    if not deficits:
        return None

    # Ký tự thiếu nhiều nhất so với target
    target_char = max(deficits, key=deficits.get)

    candidates = [
        s for s in samples
        if target_char in s["label"]
        and aug_count_by_line[s["line_id"]] < MAX_AUG_PER_ORIGINAL
    ]

    if candidates:
        return random.choice(candidates), target_char

    # Nếu ký tự thiếu nhất hết ảnh nguồn, thử các ký tự thiếu khác
    for ch, _ in sorted(deficits.items(), key=lambda x: x[1], reverse=True):
        candidates = [
            s for s in samples
            if ch in s["label"]
            and aug_count_by_line[s["line_id"]] < MAX_AUG_PER_ORIGINAL
        ]

        if candidates:
            return random.choice(candidates), ch

    return None


def augment_target_chars(samples):
    AUG_RAW_DIR.mkdir(parents=True, exist_ok=True)

    counter = count_chars(samples)
    source_pool = [s for s in samples if has_target_char(s["label"])]

    print(f"\nSource pool chứa ký tự cần tăng: {len(source_pool)}")

    augmented_samples = []
    augmented_report = []
    aug_count_by_line = Counter()

    aug_idx = 0
    max_total_aug = len(source_pool) * MAX_AUG_PER_ORIGINAL

    while needs_more(counter) and aug_idx < max_total_aug:
        chosen = choose_aug_source(source_pool, counter, aug_count_by_line)

        if chosen is None:
            print("[WARNING] Không còn ảnh nguồn phù hợp để augment.")
            break

        source_sample, target_char = chosen
        img_path = image_abs(source_sample["img_rel"])

        img = cv2.imread(str(img_path))

        if img is None:
            aug_count_by_line[source_sample["line_id"]] += 1
            continue

        aug_img = augment_image(img)

        src_name = Path(source_sample["img_rel"]).name
        ext = Path(src_name).suffix.lower()

        if ext not in IMAGE_EXTS:
            ext = ".jpg"

        new_name = f"aug_v3_{aug_idx:06d}_{target_char}_{Path(src_name).stem}{ext}"
        new_rel = f"images/train_rebalanced_v3_target_aug_raw/{new_name}"
        new_abs = PADDLE_DIR / new_rel

        cv2.imwrite(str(new_abs), aug_img)

        new_sample = {
            "line_id": 10000000 + aug_idx,
            "img_rel": new_rel,
            "label": source_sample["label"],
            "type": "augment",
            "source_img_rel": source_sample["img_rel"],
            "target_char": target_char,
        }

        augmented_samples.append(new_sample)
        counter.update(source_sample["label"])
        aug_count_by_line[source_sample["line_id"]] += 1

        augmented_report.append({
            "aug_image": new_rel,
            "source_image": source_sample["img_rel"],
            "label": source_sample["label"],
            "target_char": target_char,
            "target_chars_in_label": "".join(get_target_chars(source_sample["label"])),
        })

        aug_idx += 1

    print(f"Số ảnh augment tạo thêm: {len(augmented_samples)}")

    return augmented_samples, augmented_report


# ============================================================
# RESIZE + PADDING
# ============================================================

def resize_with_padding(img):
    h, w = img.shape[:2]

    scale = min(TARGET_W / w, TARGET_H / h)

    new_w = max(1, min(TARGET_W, int(round(w * scale))))
    new_h = max(1, min(TARGET_H, int(round(h * scale))))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC

    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((TARGET_H, TARGET_W, 3), PAD_COLOR, dtype=np.uint8)

    x = (TARGET_W - new_w) // 2
    y = (TARGET_H - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def write_padded_train(samples):
    if OUTPUT_IMAGE_DIR.exists():
        shutil.rmtree(OUTPUT_IMAGE_DIR)

    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    failed = 0

    for idx, s in enumerate(samples):
        img_path = image_abs(s["img_rel"])
        img = cv2.imread(str(img_path))

        if img is None:
            failed += 1
            continue

        padded = resize_with_padding(img)

        new_name = f"rebalance_v3_target_pad_{idx:06d}.jpg"
        new_abs = OUTPUT_IMAGE_DIR / new_name

        cv2.imwrite(str(new_abs), padded)

        new_rel = f"images/pad_train_rebalanced_v3_target/{new_name}"
        lines.append(f"{new_rel}\t{s['label']}\n")

    with open(OUTPUT_TRAIN_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n[OK] Train file mới: {OUTPUT_TRAIN_FILE}")
    print(f"[OK] Ảnh train mới : {OUTPUT_IMAGE_DIR}")
    print(f"Số dòng train mới  : {len(lines)}")
    print(f"Ảnh lỗi            : {failed}")

    return lines


# ============================================================
# REPORTS
# ============================================================

def write_reports(before_counter, after_counter, original_n, aug_n, aug_report):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_chars = sorted(set(before_counter.keys()) | set(after_counter.keys()))

    with open(REPORT_CHAR_COUNT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["char", "before", "after", "change", "target_if_any"])

        for ch in all_chars:
            writer.writerow([
                ch,
                before_counter.get(ch, 0),
                after_counter.get(ch, 0),
                after_counter.get(ch, 0) - before_counter.get(ch, 0),
                TARGET_COUNTS.get(ch, "")
            ])

    with open(REPORT_SAMPLE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "count"])
        writer.writerow(["original_train_final", original_n])
        writer.writerow(["augmented", aug_n])
        writer.writerow(["final_before_padding", original_n + aug_n])

    with open(REPORT_AUG, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "aug_image",
            "source_image",
            "label",
            "target_char",
            "target_chars_in_label"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aug_report)


# ============================================================
# MAIN
# ============================================================

def main():
    print("===== REBALANCE TRAIN V3 - TARGET ERROR CHARS =====")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not INPUT_TRAIN_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy: {INPUT_TRAIN_FILE}")

    if AUG_RAW_DIR.exists():
        shutil.rmtree(AUG_RAW_DIR)
    AUG_RAW_DIR.mkdir(parents=True, exist_ok=True)

    samples = read_label_file(INPUT_TRAIN_FILE)
    before_counter = count_chars(samples)

    print(f"Input train_final samples: {len(samples)}")
    print_target_summary("BEFORE", before_counter)

    augmented_samples, aug_report = augment_target_chars(samples)

    final_samples = samples + augmented_samples
    random.shuffle(final_samples)

    after_counter = count_chars(final_samples)
    print_target_summary("AFTER TARGETED AUGMENTATION", after_counter)

    write_padded_train(final_samples)

    write_reports(
        before_counter=before_counter,
        after_counter=after_counter,
        original_n=len(samples),
        aug_n=len(augmented_samples),
        aug_report=aug_report
    )

    print("\n===== DONE =====")
    print(f"Char report: {REPORT_CHAR_COUNT}")
    print(f"Aug report : {REPORT_AUG}")
    print("\nTrain bằng file:")
    print("rec_gt_train_rebalanced_v3_target_pad.txt")
    print("Val giữ nguyên:")
    print("rec_gt_val_final_pad.txt")
    print("Test giữ nguyên:")
    print("rec_gt_test_final_pad.txt")


if __name__ == "__main__":
    main()