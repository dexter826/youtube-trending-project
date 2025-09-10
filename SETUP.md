# 🚀 Hướng dẫn chạy YouTube Trending Analytics Project

## 📋 Yêu cầu hệ thống

- **Python 3.11+**
- **Node.js 18+** 
- **Docker Desktop**
- **Java 11** (cho Apache Spark)
- **Git**

## ⚡ Cách chạy nhanh (Windows)

### Option 1: Quick Start (sử dụng dữ liệu có sẵn)
```bash
quick-start.bat
```
Hoặc chọn option 1 trong:
```bash
start.bat
```

### Option 2: Full Rebuild (tạo lại toàn bộ dữ liệu)
```bash
start.bat
# Chọn option 2 khi được hỏi
```

**⚠️ Lưu ý:**
- **Quick Start**: Chỉ khởi động services, sử dụng dữ liệu đã có trong MongoDB (nhanh)
- **Full Rebuild**: Chạy lại Spark ETL và tạo lại toàn bộ dữ liệu từ CSV files (mất 5-10 phút)

## 🛠️ Cài đặt từng bước (Manual)

### Bước 1: Clone project

```bash
git clone https://github.com/dexter826/youtube-trending-project.git
cd youtube-trending-project
```

### Bước 2: Khởi động Infrastructure (MongoDB + Spark)

```bash
cd infra
docker compose up -d
```

**Kiểm tra containers đã chạy:**
```bash
docker ps
```

Bạn sẽ thấy 4 containers:
- `youtube-mongodb` (port 27017)
- `youtube-mongo-express` (port 8081) 
- `youtube-spark-master` (port 8080)
- `youtube-spark-worker` (port 8081)

### Bước 3: Chạy Spark ETL Pipeline

```bash
# Chuyển đến thư mục spark
cd ../spark

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy Spark job để xử lý dữ liệu
python run_spark_job.py ../data
```

**Kết quả mong đợi:**
```
🚀 Starting Spark Job for YouTube Trending Analysis
✅ Loaded 40881 records from CAvideos.csv
✅ Loaded 40840 records from DEvideos.csv
✅ Loaded 40724 records from FRvideos.csv
...
🎉 Pipeline completed successfully!
📊 Processed 1967 trending analysis results
☁️ Generated 1967 wordcloud datasets
```

### Bước 4: Khởi động Backend API

```bash
# Chuyển đến thư mục backend
cd ../backend

# Cài đặt dependencies
pip install -r requirements.txt

# Khởi động FastAPI server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**API sẽ chạy tại:** http://localhost:8000/docs

### Bước 5: Khởi động Frontend Dashboard

Mở terminal mới:

```bash
# Chuyển đến thư mục frontend
cd frontend

# Cài đặt dependencies
npm install

# Khởi động React server
npm start
```

**Dashboard sẽ mở tại:** http://localhost:3000

## 🎯 Xác nhận hệ thống hoạt động

### 1. Kiểm tra dữ liệu trong MongoDB:

```bash
python -c "
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')
db = client['youtube_trending']
print(f'Raw videos: {db.raw_videos.count_documents({}):,}')
print(f'Trending results: {db.trending_results.count_documents({}):,}')
print(f'Wordcloud data: {db.wordcloud_data.count_documents({}):,}')
"
```

**Kết quả mong đợi:**
```
Raw videos: 375,940
Trending results: 1,967
Wordcloud data: 1,967
```

### 2. Truy cập các URL:

- **Dashboard chính**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **MongoDB UI**: http://localhost:8081
- **Spark Master UI**: http://localhost:8080

## 🔧 Xử lý sự cố

### Lỗi MongoDB connection:
```bash
cd infra
docker compose restart mongodb
```

### Lỗi Spark memory:
Sửa file `spark/run_spark_job.py`, tăng memory:
```python
os.environ["SPARK_DRIVER_MEMORY"] = "8g"  # Tăng từ 4g lên 8g
```

### Lỗi Frontend build:
```bash
cd frontend
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Port đã được sử dụng:
```bash
# Kiểm tra process đang dùng port
netstat -ano | findstr :3000
netstat -ano | findstr :8000

# Kill process (thay PID)
taskkill /PID <PID> /F
```

## 📊 Dữ liệu được xử lý

Sau khi chạy thành công, hệ thống sẽ có:

- **375,940 video records** từ 10 quốc gia (CA, DE, FR, GB, IN, JP, KR, MX, RU, US)
- **1,967 trending analysis** (phân tích theo ngày và quốc gia)
- **1,967 wordcloud datasets** (từ khóa hot theo ngày/quốc gia)
- **Timespan**: 8 tháng dữ liệu (2017-11-14 đến 2018-06-14)

## 🎮 Sử dụng Dashboard

### Trending Videos Analysis:
- Xem top 10 videos theo views, likes, comments
- Lọc theo quốc gia và khoảng thời gian
- Biểu đồ tương tác với Chart.js

### Word Cloud:
- Từ khóa hot từ video titles
- Lọc theo quốc gia
- Hiển thị tần suất xuất hiện

### Analytics:
- Tổng quan thống kê
- So sánh giữa các quốc gia
- Xu hướng theo thời gian

## 🛑 Tắt hệ thống

```bash
# Tắt containers
cd infra
docker compose down

# Tắt backend (Ctrl+C)
# Tắt frontend (Ctrl+C)
```

## 📈 Performance Tips

1. **Tăng memory cho Spark** nếu xử lý dataset lớn
2. **Sử dụng SSD** cho MongoDB storage
3. **Close unused browser tabs** khi chạy dashboard
4. **Monitor Docker memory usage** trong Docker Desktop

---

**🎉 Chúc bạn demo thành công!**
