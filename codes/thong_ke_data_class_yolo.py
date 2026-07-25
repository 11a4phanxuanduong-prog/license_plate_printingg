from pathlib import Path
from collections import Counter
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\asus\OneDrive\Desktop\bao_cao_co_xo")

LABEL_ROOT = BASE_DIR / "dataset_yolo_fixed" / "labels"

SPLITS = ["train", "val", "test"]

OUTPUT_CSV = BASE_DIR / "yolo_object_count_summary.csv"


# ============================================================
# MAIN
# ============================================================

def count_objects_in_label_file(txt_path: Path) -> int:
    count = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            # Nhãn YOLO hợp lệ có ít nhất 5 giá trị:
            # class_id x_center y_center width height
            if len(parts) >= 5:
                count += 1

    return count


def main():
    print("===== THỐNG KÊ SỐ ĐỐI TƯỢNG BIỂN SỐ TRÊN MỖI ẢNH =====")
    print("Label root:", LABEL_ROOT)

    if not LABEL_ROOT.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục label: {LABEL_ROOT}")

    rows = []

    for split in SPLITS:
        label_dir = LABEL_ROOT / split

        if not label_dir.exists():
            print(f"Bỏ qua vì không có thư mục: {label_dir}")
            continue

        counter = Counter()

        txt_files = sorted(label_dir.glob("*.txt"))

        for txt_file in txt_files:
            obj_count = count_objects_in_label_file(txt_file)

            if obj_count == 0:
                group = "0 biển số"
            elif obj_count == 1:
                group = "1 biển số"
            elif obj_count == 2:
                group = "2 biển số"
            else:
                group = "Trên 2 biển số"

            counter[group] += 1

        total_images = sum(counter.values())

        for group in ["0 biển số", "1 biển số", "2 biển số", "Trên 2 biển số"]:
            count = counter[group]
            ratio = count / total_images * 100 if total_images > 0 else 0

            rows.append({
                "Tập dữ liệu": split,
                "Số đối tượng biển số trong ảnh": group,
                "Số ảnh": count,
                "Tỷ lệ (%)": round(ratio, 2)
            })

        print(f"\n--- {split.upper()} ---")
        print(f"Tổng số ảnh có label: {total_images}")
        for group in ["0 biển số", "1 biển số", "2 biển số", "Trên 2 biển số"]:
            count = counter[group]
            ratio = count / total_images * 100 if total_images > 0 else 0
            print(f"{group}: {count} ảnh = {ratio:.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n===== DONE =====")
    print("Đã lưu file:", OUTPUT_CSV)


if __name__ == "__main__":
    main()