from pathlib import Path
from PIL import Image, ImageOps
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\asus\OneDrive\Desktop\bao_cao_co_xo")

MODEL_PATH = BASE_DIR / "runs" / "detect" / "plate_model" / "weights" / "best.pt"
DATA_YAML = BASE_DIR / "data.yaml"

OUTPUT_PROJECT = BASE_DIR / "runs" / "detect"
OUTPUT_NAME = "val_test_fixed_473"

COMBINED_OUTPUT = OUTPUT_PROJECT / OUTPUT_NAME / "yolo_test_curves_combined.png"


# ============================================================
# RUN YOLO VALIDATION AGAIN
# ============================================================

def run_yolo_val():
    print("===== REDRAW YOLO TEST PLOTS =====")
    print("Model:", MODEL_PATH)
    print("Data :", DATA_YAML)

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        conf=0.25,
        iou=0.7,
        plots=True,
        save=True,
        project=str(OUTPUT_PROJECT),
        name=OUTPUT_NAME,
        exist_ok=True
    )

    print("\n===== METRICS =====")
    print("Precision:", metrics.box.mp)
    print("Recall   :", metrics.box.mr)
    print("mAP50    :", metrics.box.map50)
    print("mAP50-95 :", metrics.box.map)

    return OUTPUT_PROJECT / OUTPUT_NAME


# ============================================================
# COMBINE CURVES
# ============================================================

def resize_with_padding(img, target_size):
    img = ImageOps.contain(img, target_size)

    canvas = Image.new("RGB", target_size, "white")
    x = (target_size[0] - img.width) // 2
    y = (target_size[1] - img.height) // 2
    canvas.paste(img, (x, y))

    return canvas


def combine_curve_images(result_dir: Path):
    image_files = [
        result_dir / "BoxPR_curve.png",
        result_dir / "BoxP_curve.png",
        result_dir / "BoxR_curve.png",
        result_dir / "BoxF1_curve.png",
    ]

    missing = [p for p in image_files if not p.exists()]
    if missing:
        print("\nThiếu các file biểu đồ:")
        for p in missing:
            print("-", p)
        return

    images = [Image.open(p).convert("RGB") for p in image_files]

    single_size = (900, 700)
    images = [resize_with_padding(img, single_size) for img in images]

    margin = 40
    gap = 30

    combined_width = single_size[0] * 2 + gap + margin * 2
    combined_height = single_size[1] * 2 + gap + margin * 2

    combined = Image.new("RGB", (combined_width, combined_height), "white")

    positions = [
        (margin, margin),
        (margin + single_size[0] + gap, margin),
        (margin, margin + single_size[1] + gap),
        (margin + single_size[0] + gap, margin + single_size[1] + gap),
    ]

    for img, pos in zip(images, positions):
        combined.paste(img, pos)

    combined.save(COMBINED_OUTPUT, quality=95)

    print("\n===== COMBINED IMAGE =====")
    print("Đã lưu ảnh gộp:", COMBINED_OUTPUT)


# ============================================================
# MAIN
# ============================================================

def main():
    result_dir = run_yolo_val()
    combine_curve_images(result_dir)

    print("\n===== DONE =====")
    print("Thư mục kết quả:", result_dir)


if __name__ == "__main__":
    main()