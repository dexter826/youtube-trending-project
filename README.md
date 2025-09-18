# Phân Tích Xu Hướng YouTube

## Tổng Quan

Phân tích video xu hướng YouTube sử dụng Apache Spark, HDFS và Học Máy với giao diện người dùng React.

## Kiến Trúc

```
Dữ liệu CSV → HDFS → Xử lý Spark → MongoDB → FastAPI → Giao diện React
                      ↓
              Huấn luyện Spark MLlib → Lưu trữ Model HDFS
```

## Công Nghệ Sử Dụng

- **Big Data**: Apache Spark, HDFS, MongoDB
- **ML**: Spark MLlib (RandomForest cho hồi quy ngày, KMeans cho phân cụm)
- **Backend**: FastAPI, Python
- **Frontend**: React, TailwindCSS

## Cấu Trúc Dự Án

```
youtube-trending-project/
├── data/                    # Dữ liệu CSV thô (10 quốc gia)
├── spark/                   # Công việc Spark và mô hình ML
├── backend/                 # Ứng dụng FastAPI
│   ├── app/
│   │   ├── main.py          # Điểm vào ứng dụng FastAPI
│   │   ├── ml_service.py    # Logic dịch vụ ML
│   │   ├── routes/          # Các tuyến API
│   │   ├── services/        # Dịch vụ logic nghiệp vụ
│   │   ├── models/          # Mô hình Pydantic
│   │   └── utils/           # Các hàm tiện ích
│   └── requirements.txt     # Phụ thuộc backend
├── frontend/                # Bảng điều khiển React
│   ├── src/
│   │   ├── components/      # Các thành phần React
│   │   ├── pages/           # Các thành phần trang
│   │   ├── hooks/           # Hooks tùy chỉnh
│   │   └── context/         # Context React
│   └── package.json         # Phụ thuộc frontend
├── config/                  # Các tệp cấu hình
├── report/                  # Báo cáo dự án (PDF, DOCX)
├── tools/                   # Các công cụ bổ sung
├── run.py                   # Script chạy chính
├── setup.py                 # Script cài đặt
└── README.md
```

## Bắt Đầu Nhanh

### Yêu Cầu Tiên Quyết

- Python 3.8+, Node.js 16+, Java 8/11
- MongoDB, Apache Spark 3.x, Hadoop/HDFS
- Windows/Linux/Mac

### Cài Đặt & Thiết Lập

```bash
# Sao chép kho lưu trữ
git clone https://github.com/dexter826/youtube-trending-project.git
cd youtube-trending-project

# Thiết lập một lần nhấp (cài đặt tất cả phụ thuộc)
python setup.py
```

### Chạy Dự Án

```bash
# Khởi động backend + frontend chỉ
python run.py app

# Khởi động toàn bộ stack (cơ sở hạ tầng + pipeline + ứng dụng)
python run.py all

# Khởi động cơ sở hạ tầng chỉ (HDFS + MongoDB)
python run.py infrastructure

# Chạy pipeline dữ liệu chỉ
python run.py pipeline

# Kiểm tra trạng thái dịch vụ
python run.py status
```

### Điểm Truy Cập

- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **HDFS**: http://localhost:9870

## Tính Năng

- Phân tích xu hướng đa quốc gia (10 quốc gia)
- Dự đoán ML: phân cụm và chỉ ngày xu hướng
- Dự đoán theo URL YouTube sử dụng YouTube Data API v3 (cung cấp khóa API của bạn tại thời gian chạy)
- Bảng điều khiển tương tác với biểu đồ
- Xử lý dữ liệu thời gian thực với Spark

## Mô Hình ML

- **Hồi Quy Ngày Xu Hướng**: Hồi quy RandomForest (nhãn: days_in_trending)
- **Phân Cụm Nội Dung**: Phân cụm KMeans

## Điểm Cuối API

### Dữ Liệu

- `GET /countries` - Các quốc gia có sẵn
- `GET /trending` - Video xu hướng
- `GET /statistics` - Dữ liệu phân tích

### ML

- `POST /ml/train` - Huấn luyện mô hình (phân cụm + ngày xu hướng)
- `POST /ml/predict/days` - Dự đoán số ngày video có thể ở trên xu hướng (đầu vào có cấu trúc)
- `POST /ml/predict/cluster` - Dự đoán cụm nội dung (đầu vào có cấu trúc)
- `POST /ml/predict/url` - Dự đoán theo URL YouTube (yêu cầu body: { url, api_key })

## Cấu Hình

### Spark

```python
spark_config = {
    "spark.master": "local[*]",
    "spark.hadoop.fs.defaultFS": "hdfs://localhost:9000"
}
```

### MongoDB

```python
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "youtube_trending"
```

## Hiệu Suất

- **Khối Lượng Dữ Liệu**: 375K+ bản ghi
- **Xử Lý**: ~10K bản ghi/giây
- **Phản Hồi API**: < 200ms trung bình
- **Người Dùng Đồng Thời**: Hỗ trợ 100+ người

## Nhà Phát Triển

Trần Công Minh - MSSV: 2001222641

### Thay Đổi Cấu Trúc Tệp

- ✅ **setup.py**: Thiết lập môi trường một lần nhấp
- ✅ **run.py**: Trình chạy linh hoạt cho các chế độ khác nhau
