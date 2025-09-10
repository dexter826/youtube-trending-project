# YouTube Trending Analytics - Big Data Project

> Sinh viên thực hiện: **Trần Công Minh** - MSSV: 2001222641

## 1. Giới thiệu

Dự án phân tích video thịnh hành trên YouTube từ nhiều quốc gia, sử dụng công nghệ Big Data để xử lý, phân tích và trực quan hóa dữ liệu. Hệ thống gồm backend (Spark, FastAPI, MongoDB) và frontend (ReactJS).

## 2. Kiến trúc hệ thống

```
data (CSV) → Spark (PySpark) → MongoDB → FastAPI → React Frontend
```

Các bước chính:

- Thu thập dữ liệu từ Kaggle [Trending YouTube Video Statistics](https://www.kaggle.com/datasets/datasnaek/youtube-new)
- Xử lý, phân tích dữ liệu với Spark
- Huấn luyện mô hình Machine Learning dự đoán trending
- Lưu kết quả vào MongoDB
- Backend FastAPI cung cấp API cho frontend và ML prediction
- Frontend React hiển thị bảng, biểu đồ, wordcloud và giao diện dự đoán trending

## 3. Công nghệ sử dụng

- **Apache Spark**: Xử lý dữ liệu lớn, phân tích video trending
- **Scikit-learn**: Huấn luyện mô hình Machine Learning (Logistic Regression)
- **MongoDB**: Lưu trữ dữ liệu thô và kết quả phân tích
- **FastAPI**: Xây dựng REST API backend và ML prediction service
- **ReactJS + TailwindCSS + Chart.js**: Giao diện web trực quan với tính năng dự đoán trending
- **Docker Compose**: Quản lý, khởi tạo các dịch vụ

## 4. Cấu trúc thư mục

```
youtube-trending-project/
├── data/           # Dữ liệu CSV các quốc gia
├── spark/          # Xử lý dữ liệu với PySpark
│   ├── jobs/       # Script xử lý chính
│   ├── ml_models/  # Huấn luyện mô hình Machine Learning
│   ├── saved_models/ # Lưu trữ mô hình đã huấn luyện
│   └── requirements.txt
├── backend/
│   ├── app/        # FastAPI backend
│   │   └── main.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── services/     # API service
│   │   └── App.jsx
│   ├── public/
│   └── package.json
├── infra/
│   ├── docker-compose.yml
│   └── mongo-init.js
└── start.bat, quick-start.bat
```

## 5. Hướng dẫn cài đặt & chạy hệ thống

### Bước 1: Chuẩn bị môi trường

- Cài đặt **Docker Desktop** ([link](https://www.docker.com/products/docker-desktop))
- Cài đặt **Python 3.8+**
- Cài đặt **Node.js** (>=14)

### Bước 2: Khởi tạo dịch vụ bằng Docker

```powershell
cd infra
docker-compose up --build
```

MongoDB và backend sẽ được khởi tạo tự động.

### Bước 3: Cài đặt & chạy Spark jobs

```powershell
cd spark
pip install -r requirements.txt
python jobs/process_trending.py
```

Dữ liệu sẽ được xử lý và lưu vào MongoDB.

### Bước 4: Huấn luyện mô hình Machine Learning

```powershell
cd spark
python ml_models/trending_predictor.py
```

Mô hình ML sẽ được huấn luyện và lưu vào thư mục `saved_models/`.

### Bước 5: Chạy backend FastAPI

```powershell
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API sẽ chạy ở địa chỉ: `http://localhost:8000`

### Bước 6: Chạy frontend React

```powershell
cd frontend
npm install
npm start
```

Giao diện web tại: `http://localhost:3000`

## 6. Một số API chính

- `GET /videos`: Lấy danh sách video trending
- `GET /statistics`: Thống kê theo quốc gia, thể loại
- `GET /wordcloud`: Sinh wordcloud từ tiêu đề/video
- `POST /ml/predict-trending`: Dự đoán chi tiết khả năng trending của video
- `GET /ml/model-info`: Thông tin model Machine Learning

---

**Lưu ý:**

- Đảm bảo các file dữ liệu CSV đã có trong thư mục `data/` trước khi chạy Spark job.
- Chạy script huấn luyện ML trước khi sử dụng tính năng dự đoán trending.
- Có thể chỉnh sửa cấu hình MongoDB trong `infra/mongo-init.js` hoặc biến môi trường backend.

## 7. Tính năng Machine Learning

Hệ thống tích hợp mô hình Machine Learning để dự đoán khả năng video sẽ trending:

### Mô hình sử dụng:
- **Logistic Regression**: Mô hình chính cho dự đoán
- **Feature Engineering**: Tự động tạo các đặc trưng từ dữ liệu video
- **Preprocessing**: Chuẩn hóa dữ liệu và xử lý missing values

### Các đặc trưng chính:
- Thông tin cơ bản: lượt xem, like, dislike, comment
- Đặc trưng derived: tỷ lệ engagement, độ dài tiêu đề
- Thông tin temporal: giờ đăng, ngày trong tuần
- Thông tin kênh: số video, trung bình views

### Giao diện dự đoán:
- **Dự đoán chi tiết**: Nhập đầy đủ thông tin video để có kết quả chính xác nhất
- **Kết quả**: Hiển thị % khả năng trending, độ tin cậy và gợi ý cải thiện
- **Recommendations**: Đưa ra lời khuyên cụ thể để tăng khả năng trending
