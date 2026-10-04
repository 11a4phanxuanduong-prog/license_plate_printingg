Hệ Thống Quản Lý Xe Ra Vào Tự Động Cho Bãi Giữ Xe Nhỏ
Đồ án cơ sở - Ngành Công nghệ Thông tin (Chuyên ngành Khoa học Dữ liệu)

Trường Đại học Công nghệ TP.HCM (HUTECH)

Giảng viên hướng dẫn: ThS. Nguyễn Quang Phúc

Sinh viên thực hiện: Phan Xuân Dương (MSSV: 2386400966)

📌 Giới Thiệu Đề Tài
Trong các bãi giữ xe quy mô vừa và nhỏ (cửa hàng, quán ăn, chung cư nhỏ...), việc ghi nhận biển số và tính tiền thủ công dễ gây ra sai sót, tốn thời gian tra cứu. Hệ thống này cung cấp giải pháp tự động hóa quy trình kiểm soát phương tiện ra vào bằng các công nghệ thị giác máy tính và học sâu tiên tiến, thiết kế tối ưu để chạy mượt mà trên thiết bị cá nhân hoặc thiết bị biên.

Pipeline Xử Lý
Phát hiện biển số: Sử dụng mô hình YOLOv8n xác định khung bao (bounding box) biển số từ ảnh/camera.

Tiền xử lý ảnh: Cắt (crop) vùng biển số và chuẩn hóa bằng kỹ thuật Resize + Padding giữ nguyên tỉ lệ ký tự.

Nhận dạng ký tự (OCR): Đưa ảnh vào mô hình PaddleOCR (PP-OCRv4 / SVTR_LCNet) để trích xuất chuỗi ký tự.

Quản lý nghiệp vụ: Lưu dữ liệu vào cơ sở dữ liệu SQLite, hỗ trợ lưu xe vào/ra, phân loại phương tiện, tính phí và tổng kết doanh thu.

✨ Tính Năng Chính
Lưu xe vào bãi:

Hỗ trợ tải ảnh hoặc chụp trực tiếp qua Webcam/Camera.

Tự động phát hiện vị trí & đọc chuỗi ký tự biển số.

Tự động đề xuất loại xe (Ô tô / Xe máy) dựa trên ký tự biển số hoặc chọn thủ công.

Lưu xe ra bãi:

Nhận dạng biển số khi xe rời bãi và đối chiếu tự động với cơ sở dữ liệu.

Tự động tính thời gian gửi và tính tiền dựa trên cấu hình giá vé.

Thống kê & Báo cáo:

Xem danh sách xe hiện đang gửi trong bãi theo thời gian thực.

Tổng kết ngày: hiển thị tổng xe vào/ra, doanh thu theo loại xe và tổng doanh thu.

Xuất báo cáo dữ liệu lịch sử ra file .csv.

Cấu hình hệ thống:

Giao diện Đăng nhập / Quản trị bảo mật.

Linh hoạt tùy chỉnh khung giá vé (giá giờ đầu, giá các giờ tiếp theo) cho từng loại xe.

🛠️ Công Nghệ & Thư Viện Sử Dụng
Ngôn ngữ: Python

Computer Vision & Deep Learning: OpenCV, Ultralytics YOLO (YOLOv8n), PaddleOCR (PP-OCRv4), PyTorch.

Web App Framework: Streamlit

Cơ sở dữ liệu: SQLite3

Khác: Matplotlib (phân tích biểu đồ loss/metrics), Pandas.

📊 Kết Quả Thực Nghiệm
1. Mô hình YOLOv8n (Phát hiện biển số)
Tập dữ liệu: 4,578 ảnh biển số xe Việt Nam.

Kết quả đánh giá trên tập Test:

Precision: 0.9558 (95.58%)

Recall: 0.8608 (86.08%)

mAP50: 0.8610

Thời gian suy luận (Inference Time): ~44.0 ms/ảnh (đáp ứng thời gian thực).

2. Mô hình PaddleOCR (Nhận dạng ký tự)
Tập dữ liệu: 12,515 ảnh biển số (sau khi bổ sung ký tự hiếm và cân bằng dữ liệu).

Bộ từ điển (Dictionary): 36 ký tự (0-9 và A-Z).

Kết quả đánh giá trên tập Test:

Accuracy (Độ chính xác tuyệt đối chuỗi): 85.6%

Normalized Edit Distance (NED): 0.952

📂 Cấu Trúc File Báo Cáo
Dưới đây là sơ lược cấu trúc tài liệu báo cáo đính kèm (Báo Cáo (2).pdf):

Chương 1: Tổng quan về đề tài, tính cấp thiết, mục tiêu, đối tượng và phương pháp nghiên cứu.

Chương 2: Cơ sở lý thuyết về bài toán ALPR, mô hình YOLOv8, kiến trúc PaddleOCR và các chỉ số đánh giá (mAP, Precision, Recall, NED).

Chương 3: Chi tiết chuẩn bị dữ liệu, cân bằng ký tự hiếm, quá trình huấn luyện và kết quả thực nghiệm + Giao diện ứng dụng Demo.

Chương 4: Kết luận, các hạn chế còn tồn tại và hướng phát triển tương lai.

🚀 Hướng Dẫn Cài Đặt & Chạy Demo
1. Yêu cầu hệ thống
Python >= 3.9

Môi trường hỗ trợ GPU (khuyên dùng nếu huấn luyện lại) hoặc CPU (cho chạy app demo).

2. Cài đặt môi trường
Bash
# Clone repository
git clone https://github.com/username/QuanLyBaiXeAuto.git
cd QuanLyBaiXeAuto

# Tạo môi trường ảo (tùy chọn)
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
3. Chạy ứng dụng Streamlit
Bash
streamlit run app.py
Sau khi chạy command trên, truy cập đường dẫn http://localhost:8501 trên trình duyệt.

🔮 Hướng Phát Triển Tương Lai
[ ] Thu thập thêm dữ liệu thực tế trong điều kiện thiếu sáng, biển số mờ/nghiêng/2 dòng.

[ ] Bổ sung bước Căn chỉnh góc nghiêng (Perspective Transformation) trước khi đưa ảnh vào OCR.

[ ] Tích hợp mô hình CNN chuyên biệt để phân loại phương tiện (Ô tô / Xe máy) thay vì chỉ dự đoán qua chuỗi ký tự.

[ ] Chuyển đổi Cơ sở dữ liệu sang MySQL/PostgreSQL và triển khai luồng xử lý Video Real-time với Camera IP.

📝 Giấy Phép & Bản Quyền
Đồ án được thực hiện bởi Phan Xuân Dương phục vụ cho mục đích học tập và nghiên cứu tại Trường Đại học Công nghệ TP.HCM (HUTECH). Vui lòng ghi rõ nguồn khi tham khảo hoặc tái sử dụng.