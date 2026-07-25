import re
import math
import uuid
import sqlite3
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, date

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\asus\OneDrive\Desktop\bao_cao_co_xo")

PYTHON_EXE = Path(r"C:\Users\asus\miniconda3\envs\myenv\python.exe")

PADDLEOCR_DIR = BASE_DIR / "models" / "PaddleOCR"

YOLO_MODEL_PATH = BASE_DIR / "models" / "yolo_plate" / "best.pt"

OCR_MODEL_DIR = BASE_DIR / "models" / "vn_plate_rec_inference"
OCR_DICT_PATH = OCR_MODEL_DIR / "vn_plate_dict.txt"

DATA_DIR = BASE_DIR / "parking_data"
IMAGE_DIR = DATA_DIR / "images"
CROP_DIR = DATA_DIR / "crops"
DB_PATH = DATA_DIR / "parking.db"

YOLO_CONF = 0.25
PADDING = 5

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "123456"


# ============================================================
# INIT
# ============================================================

def init_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS parking_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT NOT NULL,
            vehicle_type TEXT NOT NULL,
            time_in TEXT NOT NULL,
            time_out TEXT,
            hours REAL,
            charged_hours INTEGER,
            fee INTEGER,
            status TEXT NOT NULL,
            image_in TEXT,
            image_out TEXT,
            created_date TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS price_settings (
            vehicle_type TEXT PRIMARY KEY,
            first_hours INTEGER NOT NULL,
            first_price INTEGER NOT NULL,
            after_price_per_hour INTEGER NOT NULL
        )
    """)

    # Giá mặc định
    default_prices = [
        ("Xe máy", 4, 5000, 6000),
        ("Xe hơi", 4, 10000, 15000),
    ]

    for vehicle_type, first_hours, first_price, after_price in default_prices:
        cur.execute("""
            INSERT OR IGNORE INTO price_settings (
                vehicle_type, first_hours, first_price, after_price_per_hour
            )
            VALUES (?, ?, ?, ?)
        """, (vehicle_type, first_hours, first_price, after_price))

    conn.commit()
    conn.close()


# ============================================================
# ADMIN LOGIN
# ============================================================

def login_page():
    st.set_page_config(
        page_title="Đăng nhập hệ thống",
        layout="centered"
    )

    st.markdown(
        """
        <div style="text-align:center; padding-top: 80px;">
            <h1> Hệ thống quản lý xe</h1>
            <h3>Đăng nhập quản trị</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")

        if st.button("Đăng nhập", use_container_width=True):
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state["is_logged_in"] = True
                st.success("Đăng nhập thành công.")
                st.rerun()
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu.")

def require_login():
    if "is_logged_in" not in st.session_state:
        st.session_state["is_logged_in"] = False

    if not st.session_state["is_logged_in"]:
        login_page()
        st.stop()


# ============================================================
# PRICE SETTINGS
# ============================================================

def get_price_settings(vehicle_type):
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT *
        FROM price_settings
        WHERE vehicle_type = ?
    """, conn, params=(vehicle_type,))
    conn.close()

    if df.empty:
        if vehicle_type == "Xe máy":
            return 4, 5000, 6000
        return 4, 10000, 15000

    row = df.iloc[0]
    return int(row["first_hours"]), int(row["first_price"]), int(row["after_price_per_hour"])


def update_price_settings(vehicle_type, first_hours, first_price, after_price):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO price_settings (
            vehicle_type, first_hours, first_price, after_price_per_hour
        )
        VALUES (?, ?, ?, ?)
    """, (vehicle_type, first_hours, first_price, after_price))

    conn.commit()
    conn.close()


def get_all_price_settings():
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT vehicle_type, first_hours, first_price, after_price_per_hour
        FROM price_settings
    """, conn)
    conn.close()
    return df


# ============================================================
# OCR PIPELINE
# ============================================================

@st.cache_resource
def load_yolo_model():
    return YOLO(str(YOLO_MODEL_PATH))


def clean_plate_text(text: str) -> str:
    text = str(text).upper().strip()
    text = text.replace(" ", "")
    text = text.replace("-", "")
    text = text.replace(".", "")
    text = text.replace("_", "")
    text = text.replace("/", "")
    text = text.replace("\\", "")
    return text

def guess_vehicle_type_from_plate(plate: str) -> str:
    """
    Phân loại tương đối xe máy / xe hơi dựa trên chuỗi biển số.

    Quy tắc này chỉ dùng cho demo, không phải quy tắc pháp lý chính thức.
    Kết quả vẫn cho phép người dùng sửa lại trên giao diện.

    Quy tắc demo:
    - Biển ô tô thường có dạng: 2 số tỉnh + 1 chữ cái + 5 số.
      Ví dụ: 51A96141.
    - Biển xe máy thường có dạng: 2 số tỉnh + 1 chữ cái + 1 số sê-ri + 5 số.
      Ví dụ: 61D209902, 93G135529.
    - Một số biển có 2 chữ cái sau mã tỉnh như LD, NN, NG, CD, KT...
      được ưu tiên gợi ý là xe hơi.
    """

    plate = clean_plate_text(plate)

    if len(plate) < 5:
        return "Xe máy"

    # Biển thường bắt đầu bằng 2 số mã tỉnh
    if not re.match(r"^\d{2}", plate):
        return "Xe máy"

    body = plate[2:]

    # Một số nhóm biển 2 chữ cái thường gặp ở ô tô hoặc nhóm đặc biệt
    car_like_prefixes = {
        "AA", "AB", "AC", "AD", "AE", "AF",
        "LD", "DA", "KT", "CD", "NN", "NG",
        "HC", "MK", "MD"
    }

    # Dạng 2 chữ cái + số, ví dụ 51LD12345
    match_two_letters = re.match(r"^([A-Z]{2})(\d+)$", body)
    if match_two_letters:
        letters = match_two_letters.group(1)

        if letters in car_like_prefixes:
            return "Xe hơi"

        return "Xe hơi"

    # Dạng 1 chữ cái + toàn số
    match_one_letter = re.match(r"^([A-Z])(\d+)$", body)
    if match_one_letter:
        digits = match_one_letter.group(2)

        # Ví dụ ô tô: 51A96141 = A + 5 số
        if len(digits) == 5:
            return "Xe hơi"

        # Ví dụ xe máy: 61D209902 = D + 6 số
        # 93G135529 = G + 6 số
        if len(digits) >= 6:
            return "Xe máy"

    # Mặc định cho demo
    return "Xe máy"

def expand_box(x1, y1, x2, y2, image_width, image_height, padding=5):
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width - 1, x2 + padding)
    y2 = min(image_height - 1, y2 + padding)
    return x1, y1, x2, y2


def save_image_file(file_obj, prefix):
    suffix = ".jpg"

    if hasattr(file_obj, "name"):
        input_suffix = Path(file_obj.name).suffix.lower()
        if input_suffix in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            suffix = input_suffix

    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
    save_path = IMAGE_DIR / filename

    with open(save_path, "wb") as f:
        f.write(file_obj.getbuffer())

    return save_path


def run_paddleocr_recognition(crop_path: Path):
    temp_dir = CROP_DIR / f"ocr_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_crop_path = temp_dir / crop_path.name
    shutil.copy2(crop_path, temp_crop_path)

    predict_script = PADDLEOCR_DIR / "tools" / "infer" / "predict_rec.py"

    command = [
        str(PYTHON_EXE),
        str(predict_script),
        "--image_dir", str(temp_crop_path),
        "--rec_model_dir", str(OCR_MODEL_DIR),
        "--rec_char_dict_path", str(OCR_DICT_PATH),
        "--rec_image_shape", "3,48,320",
        "--rec_algorithm", "SVTR_LCNet",
        "--use_gpu", "False",
        "--show_log", "True",
    ]

    result = subprocess.run(
        command,
        cwd=str(PADDLEOCR_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        return "", 0.0, stdout + "\n" + stderr

    pattern = re.compile(r"Predicts of (.*?):\('([^']*)',\s*([0-9.eE+-]+)\)")
    match = pattern.search(stdout)

    if not match:
        return "", 0.0, stdout

    plate = clean_plate_text(match.group(2))
    score = float(match.group(3))

    return plate, score, stdout


def detect_and_read_plate(image_path: Path):
    yolo_model = load_yolo_model()

    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "plate": "",
            "ocr_score": 0.0,
            "yolo_conf": 0.0,
            "crop_path": None,
            "annotated_path": None,
            "message": "Không đọc được ảnh."
        }

    image_h, image_w = image.shape[:2]

    results = yolo_model.predict(
        source=str(image_path),
        conf=YOLO_CONF,
        save=False,
        verbose=False
    )

    best_box = None
    best_conf = 0.0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            conf = float(box.conf[0].cpu().numpy())

            if conf > best_conf:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                best_box = (x1, y1, x2, y2)
                best_conf = conf

    if best_box is None:
        return {
            "plate": "",
            "ocr_score": 0.0,
            "yolo_conf": 0.0,
            "crop_path": None,
            "annotated_path": None,
            "message": "YOLO không phát hiện được biển số."
        }

    x1, y1, x2, y2 = best_box
    x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, image_w, image_h, PADDING)

    crop = image[y1:y2, x1:x2]

    crop_filename = f"crop_{image_path.stem}_{uuid.uuid4().hex[:8]}.jpg"
    crop_path = CROP_DIR / crop_filename
    cv2.imwrite(str(crop_path), crop)

    plate, ocr_score, raw_output = run_paddleocr_recognition(crop_path)

    annotated = image.copy()

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = plate if plate else "UNKNOWN"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3

    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_w, text_h = text_size

    label_y = max(30, y1 - 10)

    cv2.rectangle(
        annotated,
        (x1, label_y - text_h - 12),
        (x1 + text_w + 16, label_y + 6),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        annotated,
        label,
        (x1 + 8, label_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

    annotated_filename = f"annotated_{image_path.stem}_{uuid.uuid4().hex[:8]}.jpg"
    annotated_path = IMAGE_DIR / annotated_filename
    cv2.imwrite(str(annotated_path), annotated)

    return {
        "plate": plate,
        "ocr_score": ocr_score,
        "yolo_conf": best_conf,
        "crop_path": crop_path,
        "annotated_path": annotated_path,
        "message": raw_output
    }


# ============================================================
# DATABASE
# ============================================================

def insert_vehicle_in(plate, vehicle_type, image_in):
    conn = get_connection()
    cur = conn.cursor()

    now = datetime.now()
    time_in = now.strftime("%Y-%m-%d %H:%M:%S")
    created_date = now.strftime("%Y-%m-%d")

    cur.execute("""
        INSERT INTO parking_records (
            plate, vehicle_type, time_in, status, image_in, created_date
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        plate,
        vehicle_type,
        time_in,
        "IN",
        str(image_in) if image_in else "",
        created_date
    ))

    conn.commit()
    conn.close()


def find_active_vehicle(plate):
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT *
        FROM parking_records
        WHERE plate = ?
          AND status = 'IN'
        ORDER BY time_in DESC
        LIMIT 1
    """, conn, params=(plate,))
    conn.close()
    return df


def calculate_fee(time_in_str, vehicle_type):
    time_in = datetime.strptime(time_in_str, "%Y-%m-%d %H:%M:%S")
    time_out = datetime.now()

    duration_seconds = (time_out - time_in).total_seconds()
    duration_hours = duration_seconds / 3600

    charged_hours = max(1, math.ceil(duration_hours))

    first_hours, first_price, after_price = get_price_settings(vehicle_type)

    if charged_hours <= first_hours:
        fee = first_price
    else:
        extra_hours = charged_hours - first_hours
        fee = first_price + extra_hours * after_price

    return time_out, duration_hours, charged_hours, fee


def update_vehicle_out(record_id, time_out, duration_hours, charged_hours, fee, image_out):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE parking_records
        SET time_out = ?,
            hours = ?,
            charged_hours = ?,
            fee = ?,
            status = 'OUT',
            image_out = ?
        WHERE id = ?
    """, (
        time_out.strftime("%Y-%m-%d %H:%M:%S"),
        duration_hours,
        charged_hours,
        fee,
        str(image_out) if image_out else "",
        int(record_id)
    ))

    conn.commit()
    conn.close()


def get_active_records():
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT id, plate, vehicle_type, time_in, status
        FROM parking_records
        WHERE status = 'IN'
        ORDER BY time_in DESC
    """, conn)
    conn.close()
    return df


def get_all_records():
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT id, plate, vehicle_type, time_in, time_out, hours, charged_hours, fee, status
        FROM parking_records
        ORDER BY id DESC
    """, conn)
    conn.close()
    return df


def get_daily_summary(selected_date):
    conn = get_connection()
    date_str = selected_date.strftime("%Y-%m-%d")

    df = pd.read_sql_query("""
        SELECT *
        FROM parking_records
        WHERE created_date = ?
        ORDER BY id DESC
    """, conn, params=(date_str,))

    conn.close()
    return df


# ============================================================
# CAMERA / IMAGE INPUT
# ============================================================

def camera_or_upload_input_compact(label, key_prefix):
    st.markdown(f"### {label}")

    input_method = st.radio(
        "Nguồn ảnh",
        ["Chụp từ camera", "Tải ảnh lên"],
        horizontal=True,
        key=f"{key_prefix}_method"
    )

    file_obj = None

    left_col, right_col = st.columns([0.9, 1.1])

    with left_col:
        if input_method == "Chụp từ camera":
            file_obj = st.camera_input(
                "Máy ảnh",
                key=f"{key_prefix}_camera"
            )
        else:
            file_obj = st.file_uploader(
                "Tải ảnh",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key=f"{key_prefix}_upload"
            )

    with right_col:
        st.info(
            "Sau khi chụp hoặc tải ảnh, hệ thống sẽ tự động phát hiện biển số "
            "bằng YOLO và đọc ký tự bằng PaddleOCR."
        )

    return file_obj, input_method


# ============================================================
# UI
# ============================================================

def ui_vehicle_in():
    st.subheader("Lưu xe vào")

    st.markdown("### 1. Chọn loại xe và nguồn ảnh")

    col_top1, col_top2 = st.columns([1, 1])

    with col_top1:
        type_mode = st.radio(
            "Cách xác định loại xe",
            ["Tự động theo biển số", "Chọn thủ công"],
            horizontal=True,
            key="vehicle_type_mode_in"
        )

    with col_top2:
        if type_mode == "Chọn thủ công":
            manual_vehicle_type = st.selectbox(
                "Loại xe",
                ["Xe máy", "Xe hơi"],
                key="vehicle_in_type_manual"
            )
        else:
            manual_vehicle_type = "Xe máy"
            st.info("Hệ thống sẽ tự gợi ý loại xe sau khi đọc biển số.")

    st.markdown("### 2. Chụp hoặc tải ảnh xe vào")

    left_col, right_col = st.columns([0.8, 1.2])

    with left_col:
        input_method = st.radio(
            "Nguồn ảnh",
            ["Chụp từ camera", "Tải ảnh lên"],
            horizontal=True,
            key="vehicle_in_source"
        )

        if input_method == "Chụp từ camera":
            file_obj = st.camera_input(
                "Camera",
                key="vehicle_in_camera"
            )
        else:
            file_obj = st.file_uploader(
                "Tải ảnh xe vào",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="vehicle_in_upload"
            )

        auto_read = st.checkbox(
            "Tự động nhận dạng sau khi chụp/tải ảnh",
            value=True,
            key="auto_read_in"
        )

    with right_col:
        st.markdown("#### Ảnh kết quả")
        result_placeholder = st.empty()
        info_placeholder = st.empty()

    if file_obj is not None:
        image_path = save_image_file(file_obj, "in")

        with right_col:
            st.image(
                str(image_path),
                caption="Ảnh xe vào đã chụp/tải lên",
                use_container_width=True
            )

        should_run = auto_read or st.button("Nhận dạng biển số xe vào")

        if should_run:
            with st.spinner("Đang phát hiện biển số và OCR..."):
                result = detect_and_read_plate(image_path)

            plate_detected = result["plate"]

            if type_mode == "Tự động theo biển số":
                detected_vehicle_type = guess_vehicle_type_from_plate(plate_detected)
            else:
                detected_vehicle_type = manual_vehicle_type

            st.session_state["plate_in_detected"] = plate_detected
            st.session_state["vehicle_in_detected_type"] = detected_vehicle_type
            st.session_state["image_in_path"] = str(image_path)

            with right_col:
                if result["annotated_path"]:
                    st.image(
                        str(result["annotated_path"]),
                        caption="Ảnh đã nhận dạng biển số",
                        use_container_width=True
                    )

                if plate_detected:
                    st.success(
                        f"Biển số: {plate_detected} | "
                        f"Loại xe: {detected_vehicle_type} | "
                        f"OCR: {result['ocr_score']:.4f} | "
                        f"YOLO: {result['yolo_conf']:.4f}"
                    )
                else:
                    st.error(result["message"])

    st.markdown("### 3. Xác nhận thông tin xe vào")

    form_col1, form_col2, form_col3 = st.columns([1.2, 1, 0.8])

    with form_col1:
        plate_value = st.text_input(
            "Biển số xe",
            value=st.session_state.get("plate_in_detected", "")
        )

    auto_type_value = st.session_state.get("vehicle_in_detected_type", "Xe máy")

    with form_col2:
        vehicle_type_final = st.selectbox(
            "Loại xe",
            ["Xe máy", "Xe hơi"],
            index=0 if auto_type_value == "Xe máy" else 1,
            key="vehicle_in_type_final"
        )

    with form_col3:
        st.write("")
        st.write("")
        save_button = st.button(
            "Lưu xe vào bãi",
            use_container_width=True
        )

    if save_button:
        plate_value = clean_plate_text(plate_value)

        if not plate_value:
            st.error("Chưa có biển số.")
            return

        active = find_active_vehicle(plate_value)

        if not active.empty:
            st.warning("Xe này đang có trạng thái IN trong bãi.")
            return

        image_in = st.session_state.get("image_in_path", "")

        insert_vehicle_in(
            plate=plate_value,
            vehicle_type=vehicle_type_final,
            image_in=image_in
        )

        st.success(f"Đã lưu xe vào: {plate_value} - {vehicle_type_final}")

def ui_vehicle_out():
    st.subheader("Lưu xe ra")

    input_type = st.radio(
        "Cách nhập biển số",
        ["Nhập tay", "Chụp/upload ảnh"],
        horizontal=True
    )

    plate_out = ""
    image_out_path = ""

    if input_type == "Nhập tay":
        plate_out = st.text_input("Nhập biển số xe ra")
    else:
        file_obj, _ = camera_or_upload_input_compact(
            "Ảnh xe ra",
            "vehicle_out"
                    )

        auto_read = st.checkbox(
            "Tự động nhận dạng sau khi chụp/upload",
            value=True,
            key="auto_read_out"
        )

        if file_obj is not None:
            image_path = save_image_file(file_obj, "out")
            image_out_path = str(image_path)
            st.image(str(image_path), caption="Ảnh xe ra đã chụp/upload", use_container_width=True)

            should_run = auto_read or st.button("Nhận dạng biển số xe ra")

            if should_run:
                with st.spinner("Đang detect biển số và OCR..."):
                    result = detect_and_read_plate(image_path)

                if result["annotated_path"]:
                    st.image(
                        str(result["annotated_path"]),
                        caption="Ảnh kết quả nhận dạng xe ra",
                        use_container_width=True
                    )

                st.session_state["plate_out_detected"] = result["plate"]
                st.session_state["image_out_path"] = image_out_path

                if result["plate"]:
                    st.success(
                        f"Biển số: {result['plate']} | "
                        f"OCR: {result['ocr_score']:.4f} | "
                        f"YOLO: {result['yolo_conf']:.4f}"
                    )
                else:
                    st.error(result["message"])

        plate_out = st.text_input(
            "Biển số xe ra",
            value=st.session_state.get("plate_out_detected", "")
        )

    if st.button("Tìm xe đang gửi"):
        plate_out = clean_plate_text(plate_out)

        if not plate_out:
            st.error("Chưa nhập biển số.")
            return

        df = find_active_vehicle(plate_out)

        if df.empty:
            st.error("Không tìm thấy xe đang gửi trong bãi.")
            return

        record = df.iloc[0]

        time_out, duration_hours, charged_hours, fee = calculate_fee(
            record["time_in"],
            record["vehicle_type"]
        )

        st.session_state["out_record_id"] = int(record["id"])
        st.session_state["out_plate"] = record["plate"]
        st.session_state["out_vehicle_type"] = record["vehicle_type"]
        st.session_state["out_time_in"] = record["time_in"]
        st.session_state["out_duration_hours"] = duration_hours
        st.session_state["out_charged_hours"] = charged_hours
        st.session_state["out_fee"] = fee
        st.session_state["out_time_out"] = time_out.strftime("%Y-%m-%d %H:%M:%S")

        st.info(
            f"Biển số: {record['plate']} | "
            f"Loại xe: {record['vehicle_type']} | "
            f"Giờ vào: {record['time_in']} | "
            f"Số giờ tính tiền: {charged_hours} | "
            f"Tiền vé: {fee:,} VNĐ"
        )

    if "out_record_id" in st.session_state:
        st.markdown("### Xác nhận xe ra")

        st.write(f"Biển số: **{st.session_state['out_plate']}**")
        st.write(f"Loại xe: **{st.session_state['out_vehicle_type']}**")
        st.write(f"Giờ vào: **{st.session_state['out_time_in']}**")
        st.write(f"Giờ ra: **{st.session_state['out_time_out']}**")
        st.write(f"Số giờ thực tế: **{st.session_state['out_duration_hours']:.2f} giờ**")
        st.write(f"Số giờ tính tiền: **{st.session_state['out_charged_hours']} giờ**")
        st.write(f"Tiền vé: **{st.session_state['out_fee']:,} VNĐ**")

        if st.button("Xác nhận lưu xe ra"):
            image_out = st.session_state.get("image_out_path", "")

            update_vehicle_out(
                record_id=st.session_state["out_record_id"],
                time_out=datetime.strptime(
                    st.session_state["out_time_out"],
                    "%Y-%m-%d %H:%M:%S"
                ),
                duration_hours=st.session_state["out_duration_hours"],
                charged_hours=st.session_state["out_charged_hours"],
                fee=st.session_state["out_fee"],
                image_out=image_out
            )

            st.success("Đã lưu xe ra và tính tiền thành công.")

            keys_to_clear = [
                "out_record_id",
                "out_plate",
                "out_vehicle_type",
                "out_time_in",
                "out_duration_hours",
                "out_charged_hours",
                "out_fee",
                "out_time_out",
                "plate_out_detected",
                "image_out_path"
            ]

            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]


def ui_active_records():
    st.subheader("Xe đang trong bãi")

    df = get_active_records()

    if df.empty:
        st.info("Hiện không có xe nào đang gửi.")
    else:
        st.dataframe(df, use_container_width=True)


def ui_daily_summary():
    st.subheader("Tổng kết ngày")

    selected_date = st.date_input("Chọn ngày", value=date.today())

    df = get_daily_summary(selected_date)

    if df.empty:
        st.info("Không có dữ liệu trong ngày này.")
        return

    total_in = len(df)
    total_out = len(df[df["status"] == "OUT"])
    total_revenue = int(df["fee"].fillna(0).sum())

    motorbike_df = df[df["vehicle_type"] == "Xe máy"]
    car_df = df[df["vehicle_type"] == "Xe hơi"]

    motorbike_revenue = int(motorbike_df["fee"].fillna(0).sum())
    car_revenue = int(car_df["fee"].fillna(0).sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Tổng xe vào", total_in)
    col2.metric("Tổng xe ra", total_out)
    col3.metric("Doanh thu xe máy", f"{motorbike_revenue:,} VNĐ")
    col4.metric("Doanh thu xe hơi", f"{car_revenue:,} VNĐ")

    st.metric("Tổng doanh thu", f"{total_revenue:,} VNĐ")

    st.markdown("### Chi tiết trong ngày")

    show_cols = [
        "id",
        "plate",
        "vehicle_type",
        "time_in",
        "time_out",
        "hours",
        "charged_hours",
        "fee",
        "status"
    ]

    available_cols = [col for col in show_cols if col in df.columns]

    st.dataframe(df[available_cols], use_container_width=True)

    csv_data = df.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        label="Tải báo cáo CSV",
        data=csv_data,
        file_name=f"parking_summary_{selected_date}.csv",
        mime="text/csv"
    )


def ui_history():
    st.subheader("Lịch sử gửi xe")

    df = get_all_records()

    if df.empty:
        st.info("Chưa có dữ liệu.")
    else:
        st.dataframe(df, use_container_width=True)


def ui_settings():
    st.subheader("Cấu hình giá vé")

    st.markdown("### Giá vé hiện tại")

    df = get_all_price_settings()
    st.dataframe(df, use_container_width=True)

    st.markdown("### Chỉnh giá vé")

    vehicle_type = st.selectbox("Loại xe", ["Xe máy", "Xe hơi"])

    first_hours, first_price, after_price = get_price_settings(vehicle_type)

    new_first_hours = st.number_input(
        "Số giờ đầu",
        min_value=1,
        max_value=24,
        value=int(first_hours)
    )

    new_first_price = st.number_input(
        "Giá cho số giờ đầu",
        min_value=0,
        step=1000,
        value=int(first_price)
    )

    new_after_price = st.number_input(
        "Giá mỗi giờ sau đó",
        min_value=0,
        step=1000,
        value=int(after_price)
    )

    if st.button("Cập nhật giá vé"):
        update_price_settings(
            vehicle_type=vehicle_type,
            first_hours=int(new_first_hours),
            first_price=int(new_first_price),
            after_price=int(new_after_price)
        )

        st.success("Đã cập nhật giá vé.")
        st.rerun()


# ============================================================
# APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Hệ thống quản lý xe",
        layout="wide"
    )

    init_dirs()
    init_db()
    require_login()
    st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        max-width: 1300px;
    }

    h1 {
        font-size: 34px !important;
        margin-bottom: 0.5rem !important;
    }

    h2, h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    section[data-testid="stSidebar"] {
        width: 260px !important;
    }

    div[data-testid="stCameraInput"] {
        max-width: 420px !important;
    }

    div[data-testid="stCameraInput"] video {
        max-height: 280px !important;
        object-fit: contain !important;
        border-radius: 10px !important;
    }

    div[data-testid="stCameraInput"] img {
        max-height: 280px !important;
        object-fit: contain !important;
        border-radius: 10px !important;
    }

    .stImage img {
        max-height: 420px;
        object-fit: contain;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    # CSS làm giao diện gọn hơn
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        h1 {
            font-size: 34px !important;
            margin-bottom: 0.5rem !important;
        }

        h2, h3 {
            margin-top: 0.5rem !important;
        }

        section[data-testid="stSidebar"] {
            width: 260px !important;
        }

        div[data-testid="stCameraInput"] video {
            max-height: 320px !important;
            object-fit: contain !important;
        }

        div[data-testid="stCameraInput"] {
            max-width: 480px !important;
        }

        img {
            border-radius: 8px;
        }

        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <div>
                <h1>Hệ thống quản lý xe</h1>
                <p style="font-size:16px; color:#999;">
                    Nhận dạng biển số bằng YOLO + PaddleOCR và quản lý xe vào/ra
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("Menu")

    if st.sidebar.button("Đăng xuất", use_container_width=True):
        st.session_state["is_logged_in"] = False
        st.rerun()

    page = st.sidebar.radio(
        "Chọn chức năng",
        [
            "Lưu xe vào",
            "Lưu xe ra",
            "Xe đang trong bãi",
            "Tổng kết ngày",
            "Lịch sử",
            "Cấu hình giá vé"
        ]
    )

    st.divider()

    if page == "Lưu xe vào":
        ui_vehicle_in()
    elif page == "Lưu xe ra":
        ui_vehicle_out()
    elif page == "Xe đang trong bãi":
        ui_active_records()
    elif page == "Tổng kết ngày":
        ui_daily_summary()
    elif page == "Lịch sử":
        ui_history()
    elif page == "Cấu hình giá vé":
        ui_settings()

if __name__ == "__main__":
    main()