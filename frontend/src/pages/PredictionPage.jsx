import React, { useState, useEffect } from "react";
import {
  TrendingUp,
  Eye,
  Target,
  Zap,
  AlertCircle,
  CheckCircle,
  Play,
  BarChart3,
} from "lucide-react";
import { useApi } from "../context/ApiContext";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorMessage from "../components/ErrorMessage";

const PredictionPage = () => {
  const {
    predictTrending,
    predictViews,
    predictCluster,
    checkMLHealth,
    fetchCategories,
    loading,
    error,
  } = useApi();

  const [mlHealth, setMlHealth] = useState(null);
  const [categories, setCategories] = useState([]);
  const [videoData, setVideoData] = useState({
    title: "",
    views: 0,
    likes: 0,
    dislikes: 0,
    comment_count: 0,
    category_id: 0,
    tags: "",
    publish_hour: 12,
    video_age_proxy: 2,
  });
  const [predictions, setPredictions] = useState({
    trending: null,
    views: null,
    cluster: null,
  });
  const [predictionLoading, setPredictionLoading] = useState({
    trending: false,
    views: false,
    cluster: false,
    all: false,
  });
  const [samplePredictions, setSamplePredictions] = useState({
    tech: {
      data: {
        title: "iPhone 15 Pro Max Review - Is It Worth The Hype?",
        views: 50000,
        likes: 2500,
        dislikes: 150,
        comment_count: 800,
        category_id: 28, // Science & Technology
        tags: "iPhone|iPhone 15|review|tech|Apple|smartphone",
        publish_hour: 14,
        video_age_proxy: 2,
      },
      predictions: {
        trending: {
          prediction: {
            trending_probability: 0.014,
            prediction: 0,
            confidence: "high",
            method: "spark_mllib",
          },
        },
        views: {
          prediction: {
            predicted_views: 51633,
            confidence: "high",
            method: "spark_mllib",
          },
        },
        cluster: {
          prediction: {
            cluster: 0,
            cluster_type: "Nội dung Tác động Cao",
            confidence: "high",
            method: "spark_mllib",
          },
        },
      },
    },
    entertainment: {
      data: {
        title: "Top 10 Funniest Cat Videos of 2024!",
        views: 100000,
        likes: 8000,
        dislikes: 200,
        comment_count: 1200,
        category_id: 24, // Entertainment
        tags: "cats|funny|videos|pets|comedy|animals",
        publish_hour: 18,
        video_age_proxy: 1,
      },
      predictions: {
        trending: {
          prediction: {
            trending_probability: 0.025,
            prediction: 0,
            confidence: "high",
            method: "spark_mllib",
          },
        },
        views: {
          prediction: {
            predicted_views: 100601,
            confidence: "high",
            method: "spark_mllib",
          },
        },
        cluster: {
          prediction: {
            cluster: 1,
            cluster_type: "Nội dung Đại chúng",
            confidence: "high",
            method: "spark_mllib",
          },
        },
      },
    },
    trending: {
      data: {
        title: "Breaking News: Major Event Shocks the World!",
        views: 1000000,
        likes: 150000,
        dislikes: 5000,
        comment_count: 25000,
        category_id: 25, // News & Politics
        tags: "news|breaking|viral|trending|world|event",
        publish_hour: 8,
        video_age_proxy: 1,
      },
      predictions: {
        trending: {
          prediction: {
            trending_probability: 0.85,
            prediction: 1,
            confidence: "high",
            method: "spark_mllib",
          },
        },
        views: {
          prediction: {
            predicted_views: 1200000,
            confidence: "high",
            method: "spark_mllib",
          },
        },
        cluster: {
          prediction: {
            cluster: 2,
            cluster_type: "Nội dung Tiềm năng",
            confidence: "high",
            method: "spark_mllib",
          },
        },
      },
    },
  });
  const [selectedSample, setSelectedSample] = useState(null);

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const [healthData, categoriesData] = await Promise.all([
          checkMLHealth(),
          fetchCategories(),
        ]);
        setMlHealth(healthData);
        setCategories(categoriesData.categories || []);
      } catch (err) {
        // Error handled by ApiContext
      }
    };

    loadInitialData();
  }, [checkMLHealth, fetchCategories]);

  const handleInputChange = (field, value) => {
    setVideoData((prev) => ({
      ...prev,
      [field]:
        field.includes("count") ||
        field === "views" ||
        field === "likes" ||
        field === "dislikes" ||
        field === "category_id" ||
        field === "publish_hour" ||
        field === "video_age_proxy"
          ? parseInt(value) || 0
          : value,
    }));
  };

  const handlePredict = async (type) => {
    setPredictionLoading((prev) => ({ ...prev, [type]: true }));
    // Clear previous predictions to avoid showing stale data
    setPredictions((prev) => ({ ...prev, [type]: null }));
    setSelectedSample(null); // Clear selected sample when doing real prediction
    try {
      let result;
      switch (type) {
        case "trending":
          result = await predictTrending(videoData);
          setPredictions((prev) => ({ ...prev, trending: result }));
          break;
        case "views":
          result = await predictViews(videoData);
          setPredictions((prev) => ({ ...prev, views: result }));
          break;
        case "cluster":
          result = await predictCluster(videoData);
          setPredictions((prev) => ({ ...prev, cluster: result }));
          break;
        default:
          break;
      }
    } catch (err) {
      // Error handled by ApiContext
    } finally {
      setPredictionLoading((prev) => ({ ...prev, [type]: false }));
    }
  };

  const handlePredictAll = async () => {
    setPredictionLoading((prev) => ({
      ...prev,
      all: true,
      trending: true,
      views: true,
      cluster: true,
    }));
    // Clear all predictions
    setPredictions({ trending: null, views: null, cluster: null });
    setSelectedSample(null);
    try {
      const [trendingResult, viewsResult, clusterResult] = await Promise.all([
        predictTrending(videoData),
        predictViews(videoData),
        predictCluster(videoData),
      ]);

      setPredictions({
        trending: trendingResult,
        views: viewsResult,
        cluster: clusterResult,
      });
    } catch (err) {
      // Error handled by ApiContext
    } finally {
      setPredictionLoading((prev) => ({
        ...prev,
        all: false,
        trending: false,
        views: false,
        cluster: false,
      }));
    }
  };

  const loadSample = (sampleKey) => {
    const sample = samplePredictions[sampleKey];
    if (sample) {
      setVideoData(sample.data);
      setSelectedSample(sampleKey);
      // Reset predictions to show sample predictions
      setPredictions({
        trending: sample.predictions.trending,
        views: sample.predictions.views,
        cluster: sample.predictions.cluster,
      });
    }
  };

  const formatNumber = (num) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + "M";
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + "K";
    }
    return num?.toLocaleString() || "0";
  };

  const isFormValid = () => {
    return (
      videoData.title.trim() !== "" &&
      videoData.category_id !== 0 &&
      videoData.views >= 0 &&
      videoData.likes >= 0 &&
      videoData.dislikes >= 0 &&
      videoData.comment_count >= 0
    );
  };

  const PredictionCard = ({
    title,
    icon: Icon,
    result,
    color = "blue",
    type,
  }) => (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Icon className={`w-5 h-5 text-${color}-600`} />
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        </div>
        {result?.prediction && !predictionLoading[type] && (
          <div className="flex items-center space-x-1">
            <CheckCircle className="w-4 h-4 text-green-500" />
            <span className="text-xs text-green-600">Hoàn thành</span>
          </div>
        )}
      </div>

      <div className="min-h-[200px] flex items-center justify-center">
        {predictionLoading[type] ? (
          <LoadingSpinner message="Đang thực hiện dự đoán..." />
        ) : result ? (
          <div className="w-full space-y-3">
            {/* Hiển thị kết quả dưới dạng bảng */}
            <div className="overflow-x-auto">
              <table className="min-w-full table-auto border-collapse border border-gray-300">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="border border-gray-300 px-4 py-2 text-left text-sm font-medium text-gray-700">
                      Thuộc tính
                    </th>
                    <th className="border border-gray-300 px-4 py-2 text-left text-sm font-medium text-gray-700">
                      Giá trị
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {/* Sử dụng result.prediction thay vì result trực tiếp */}
                  {result.prediction?.trending_probability !== undefined && (
                    <tr>
                      <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                        Xác suất Trending
                      </td>
                      <td
                        className={`border border-gray-300 px-4 py-2 text-sm font-bold ${
                          result.prediction.trending_probability > 0.7
                            ? "text-green-600"
                            : result.prediction.trending_probability > 0.4
                            ? "text-yellow-600"
                            : "text-red-600"
                        }`}
                      >
                        {(result.prediction.trending_probability * 100).toFixed(
                          1
                        )}
                        %
                      </td>
                    </tr>
                  )}
                  {result.prediction?.prediction !== undefined && (
                    <tr>
                      <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                        Dự đoán
                      </td>
                      <td
                        className={`border border-gray-300 px-4 py-2 text-sm font-bold text-${color}-600`}
                      >
                        {result.prediction.prediction === 1
                          ? "Có khả năng trending"
                          : "Không trending"}
                      </td>
                    </tr>
                  )}
                  {result.prediction?.predicted_views !== undefined && (
                    <tr>
                      <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                        Dự đoán lượt xem
                      </td>
                      <td
                        className={`border border-gray-300 px-4 py-2 text-sm font-bold text-${color}-600`}
                      >
                        {formatNumber(result.prediction.predicted_views)}
                      </td>
                    </tr>
                  )}
                  {result.prediction?.cluster !== undefined && (
                    <tr>
                      <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                        Cluster
                      </td>
                      <td
                        className={`border border-gray-300 px-4 py-2 text-sm font-bold text-${color}-600`}
                      >
                        {result.prediction.cluster}
                      </td>
                    </tr>
                  )}
                  {result.prediction?.cluster_type && (
                    <tr>
                      <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                        Loại nội dung
                      </td>
                      <td
                        className={`border border-gray-300 px-4 py-2 text-sm`}
                      >
                        <span
                          className={`px-2 py-1 bg-${color}-100 text-${color}-700 text-sm rounded-full`}
                        >
                          {result.prediction.cluster_type}
                        </span>
                      </td>
                    </tr>
                  )}
                  {result.prediction?.confidence && (
                    <tr>
                      <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                        Độ tin cậy
                      </td>
                      <td
                        className={`border border-gray-300 px-4 py-2 text-sm font-medium ${
                          result.prediction.confidence === "high"
                            ? "text-green-600"
                            : result.prediction.confidence === "medium"
                            ? "text-yellow-600"
                            : "text-red-600"
                        }`}
                      >
                        {result.prediction.confidence === "high"
                          ? "Cao"
                          : result.prediction.confidence === "medium"
                          ? "Trung bình"
                          : "Thấp"}
                      </td>
                    </tr>
                  )}
                  {result.prediction?.method && (
                    <tr>
                      <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                        Phương pháp
                      </td>
                      <td className="border border-gray-300 px-4 py-2 text-sm text-gray-500">
                        {result.prediction.method}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <Icon className={`w-12 h-12 text-gray-300 mx-auto mb-2`} />
            <p>Chưa có dự đoán</p>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            Dự đoán Video Trending
          </h1>
          <p className="mt-2 text-gray-600">
            Sử dụng Machine Learning để dự đoán hiệu suất video
          </p>
        </div>
      </div>

      {error && <ErrorMessage message={error} />}

      {/* ML Health Status */}
      {mlHealth && (
        <div className="card">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div
                className={`w-3 h-3 rounded-full ${
                  mlHealth.is_trained ? "bg-green-500" : "bg-red-500"
                }`}
              />
              <div>
                <h3 className="text-lg font-semibold text-gray-900">
                  Trạng thái ML Models
                </h3>
                <p className="text-sm text-gray-600">
                  {mlHealth.is_trained
                    ? `${mlHealth.total_models} mô hình đã sẵn sàng`
                    : "Mô hình chưa được huấn luyện"}
                </p>
              </div>
            </div>

            {!mlHealth.is_trained && (
              <div className="flex items-center space-x-2 text-yellow-600">
                <AlertCircle className="w-5 h-5" />
                <span className="text-sm">Cần huấn luyện mô hình</span>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="space-y-6">
          <div className="card">
            <div className="flex items-center space-x-2 mb-6">
              <Play className="w-5 h-5 text-blue-600" />
              <h3 className="text-lg font-semibold text-gray-900">
                Thông tin Video
              </h3>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Tiêu đề video
                </label>
                <input
                  type="text"
                  value={videoData.title}
                  onChange={(e) => handleInputChange("title", e.target.value)}
                  placeholder="Nhập tiêu đề video..."
                  className="input-field"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Lượt xem hiện tại
                  </label>
                  <input
                    type="number"
                    value={videoData.views}
                    onChange={(e) => handleInputChange("views", e.target.value)}
                    placeholder="0"
                    className="input-field"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Lượt thích
                  </label>
                  <input
                    type="number"
                    value={videoData.likes}
                    onChange={(e) => handleInputChange("likes", e.target.value)}
                    placeholder="0"
                    className="input-field"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Lượt không thích
                  </label>
                  <input
                    type="number"
                    value={videoData.dislikes}
                    onChange={(e) =>
                      handleInputChange("dislikes", e.target.value)
                    }
                    placeholder="0"
                    className="input-field"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Số bình luận
                  </label>
                  <input
                    type="number"
                    value={videoData.comment_count}
                    onChange={(e) =>
                      handleInputChange("comment_count", e.target.value)
                    }
                    placeholder="0"
                    className="input-field"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Danh mục
                </label>
                <select
                  value={videoData.category_id}
                  onChange={(e) =>
                    handleInputChange("category_id", e.target.value)
                  }
                  className="input-field"
                >
                  <option value={0}>Chọn danh mục</option>
                  {categories.map((category) => (
                    <option key={category.id} value={category.id}>
                      {category.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Tags (phân cách bằng |)
                </label>
                <textarea
                  value={videoData.tags}
                  onChange={(e) => handleInputChange("tags", e.target.value)}
                  placeholder="tag1|tag2|tag3..."
                  rows={3}
                  className="input-field"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Giờ đăng (0-23)
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="23"
                    value={videoData.publish_hour}
                    onChange={(e) =>
                      handleInputChange("publish_hour", e.target.value)
                    }
                    placeholder="12"
                    className="input-field"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Độ tuổi video (1-4)
                  </label>
                  <select
                    value={videoData.video_age_proxy}
                    onChange={(e) =>
                      handleInputChange("video_age_proxy", e.target.value)
                    }
                    className="input-field"
                  >
                    <option value={1}>Rất mới (0-1 ngày)</option>
                    <option value={2}>Mới (2-7 ngày)</option>
                    <option value={3}>Gần đây (8-30 ngày)</option>
                    <option value={4}>Cũ (&gt;30 ngày)</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Thực hiện Dự đoán
            </h3>

            <div className="space-y-3">
              <button
                onClick={handlePredictAll}
                disabled={loading || !mlHealth?.is_trained || !isFormValid()}
                className="btn-primary w-full flex items-center justify-center space-x-2"
              >
                <Zap className="w-4 h-4" />
                <span>Dự đoán Tất cả</span>
              </button>

              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => handlePredict("trending")}
                  disabled={loading || !mlHealth?.is_trained || !isFormValid()}
                  className="btn-secondary text-xs flex items-center justify-center space-x-1"
                >
                  <TrendingUp className="w-3 h-3" />
                  <span>Trending</span>
                </button>

                <button
                  onClick={() => handlePredict("views")}
                  disabled={loading || !mlHealth?.is_trained || !isFormValid()}
                  className="btn-secondary text-xs flex items-center justify-center space-x-1"
                >
                  <Eye className="w-3 h-3" />
                  <span>Views</span>
                </button>

                <button
                  onClick={() => handlePredict("cluster")}
                  disabled={loading || !mlHealth?.is_trained || !isFormValid()}
                  className="btn-secondary text-xs flex items-center justify-center space-x-1"
                >
                  <Target className="w-3 h-3" />
                  <span>Cluster</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Predictions Results */}
        <div className="space-y-6">
          <PredictionCard
            title="Dự đoán Trending"
            icon={TrendingUp}
            result={predictions.trending}
            color="green"
            type="trending"
          />

          <PredictionCard
            title="Dự đoán Lượt xem"
            icon={Eye}
            result={predictions.views}
            color="blue"
            type="views"
          />

          <PredictionCard
            title="Phân loại Nội dung"
            icon={Target}
            result={predictions.cluster}
            color="purple"
            type="cluster"
          />
        </div>
      </div>

      {/* Sample Predictions */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <BarChart3 className="w-5 h-5 text-gray-600" />
          <h3 className="text-lg font-semibold text-gray-900">Dự đoán Mẫu</h3>
          <span className="text-sm text-gray-500">
            (Click để chọn và kiểm tra)
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Tech Review Sample Prediction */}
          <div
            className={`p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
              selectedSample === "tech"
                ? "border-blue-500 bg-blue-50 shadow-md"
                : "border-gray-200 bg-gray-50 hover:border-gray-300"
            }`}
            onClick={() => loadSample("tech")}
          >
            <h4 className="font-medium text-gray-900 mb-2">
              Tech Review Sample
            </h4>

            <div className="space-y-2">
              <div>
                <span className="text-sm text-gray-600">Trending:</span>
                <div className="flex items-center">
                  <div
                    className={`w-3 h-3 rounded-full mr-2 ${
                      samplePredictions.tech.predictions.trending.prediction
                        .trending_probability > 0.7
                        ? "bg-green-500"
                        : samplePredictions.tech.predictions.trending.prediction
                            .trending_probability > 0.4
                        ? "bg-yellow-500"
                        : "bg-red-500"
                    }`}
                  />
                  <span className="text-sm font-semibold">
                    {samplePredictions.tech.predictions.trending.prediction
                      .prediction === 1
                      ? "Có khả năng trending"
                      : "Không trending"}
                  </span>
                </div>
              </div>

              <div>
                <span className="text-sm text-gray-600">Lượt xem:</span>
                <span className="text-sm font-semibold">
                  {formatNumber(
                    samplePredictions.tech.predictions.views.prediction
                      .predicted_views
                  )}
                </span>
              </div>

              <div>
                <span className="text-sm text-gray-600">Cluster:</span>
                <span className="text-sm font-semibold">
                  {
                    samplePredictions.tech.predictions.cluster.prediction
                      .cluster
                  }
                </span>
              </div>

              <div>
                <span className="text-sm text-gray-600">Loại nội dung:</span>
                <span className="text-sm font-semibold">
                  High-View Medium-Engagement SEO-Optimized (Entertainment)
                </span>
              </div>
            </div>
          </div>

          {/* Entertainment Sample Prediction */}
          <div
            className={`p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
              selectedSample === "entertainment"
                ? "border-blue-500 bg-blue-50 shadow-md"
                : "border-gray-200 bg-gray-50 hover:border-gray-300"
            }`}
            onClick={() => loadSample("entertainment")}
          >
            <h4 className="font-medium text-gray-900 mb-2">
              Entertainment Sample
            </h4>

            <div className="space-y-2">
              <div>
                <span className="text-sm text-gray-600">Trending:</span>
                <div className="flex items-center">
                  <div
                    className={`w-3 h-3 rounded-full mr-2 ${
                      samplePredictions.entertainment.predictions.trending
                        .prediction.trending_probability > 0.7
                        ? "bg-green-500"
                        : samplePredictions.entertainment.predictions.trending
                            .prediction.trending_probability > 0.4
                        ? "bg-yellow-500"
                        : "bg-red-500"
                    }`}
                  />
                  <span className="text-sm font-semibold">
                    {samplePredictions.entertainment.predictions.trending
                      .prediction.prediction === 1
                      ? "Có khả năng trending"
                      : "Không trending"}
                  </span>
                </div>
              </div>

              <div>
                <span className="text-sm text-gray-600">Lượt xem:</span>
                <span className="text-sm font-semibold">
                  {formatNumber(
                    samplePredictions.entertainment.predictions.views.prediction
                      .predicted_views
                  )}
                </span>
              </div>

              <div>
                <span className="text-sm text-gray-600">Cluster:</span>
                <span className="text-sm font-semibold">
                  {
                    samplePredictions.entertainment.predictions.cluster
                      .prediction.cluster
                  }
                </span>
              </div>

              <div>
                <span className="text-sm text-gray-600">Loại nội dung:</span>
                <span className="text-sm font-semibold">
                  {
                    samplePredictions.entertainment.predictions.cluster
                      .prediction.cluster_type
                  }
                </span>
              </div>
            </div>
          </div>

          {/* Trending Sample Prediction */}
          <div
            className={`p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
              selectedSample === "trending"
                ? "border-blue-500 bg-blue-50 shadow-md"
                : "border-gray-200 bg-gray-50 hover:border-gray-300"
            }`}
            onClick={() => loadSample("trending")}
          >
            <h4 className="font-medium text-gray-900 mb-2">Trending Sample</h4>

            <div className="space-y-2">
              <div>
                <span className="text-sm text-gray-600">Trending:</span>
                <div className="flex items-center">
                  <div
                    className={`w-3 h-3 rounded-full mr-2 ${
                      samplePredictions.trending.predictions.trending.prediction
                        .trending_probability > 0.7
                        ? "bg-green-500"
                        : samplePredictions.trending.predictions.trending
                            .prediction.trending_probability > 0.4
                        ? "bg-yellow-500"
                        : "bg-red-500"
                    }`}
                  />
                  <span className="text-sm font-semibold">
                    {samplePredictions.trending.predictions.trending.prediction
                      .prediction === 1
                      ? "Có khả năng trending"
                      : "Không trending"}
                  </span>
                </div>
              </div>

              <div>
                <span className="text-sm text-gray-600">Lượt xem:</span>
                <span className="text-sm font-semibold">
                  {formatNumber(
                    samplePredictions.trending.predictions.views.prediction
                      .predicted_views
                  )}
                </span>
              </div>

              <div>
                <span className="text-sm text-gray-600">Cluster:</span>
                <span className="text-sm font-semibold">
                  {
                    samplePredictions.trending.predictions.cluster.prediction
                      .cluster
                  }
                </span>
              </div>

              <div>
                <span className="text-sm text-gray-600">Loại nội dung:</span>
                <span className="text-sm font-semibold">
                  {
                    samplePredictions.trending.predictions.cluster.prediction
                      .cluster_type
                  }
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionPage;
