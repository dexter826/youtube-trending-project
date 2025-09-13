import React from "react";
import { TrendingUp, Eye, Target, CheckCircle } from "lucide-react";
import LoadingSpinner from "../LoadingSpinner";

const PredictionCard = ({
  title,
  icon: Icon,
  result,
  color = "blue",
  type,
  loading,
}) => (
  <div className="card">
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center space-x-2">
        <Icon className={`w-5 h-5 text-${color}-600`} />
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
      </div>
      {result?.prediction && !loading && (
        <div className="flex items-center space-x-1">
          <CheckCircle className="w-4 h-4 text-green-500" />
          <span className="text-xs text-green-600">Hoàn thành</span>
        </div>
      )}
    </div>

    <div className="min-h-[200px] flex items-center justify-center">
      {loading ? (
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

const formatNumber = (num) => {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + "M";
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + "K";
  }
  return num?.toLocaleString() || "0";
};

const PredictionResults = ({ predictions, predictionLoading }) => {
  return (
    <div className="space-y-6">
      <PredictionCard
        title="Dự đoán Trending"
        icon={TrendingUp}
        result={predictions.trending}
        color="green"
        type="trending"
        loading={predictionLoading.trending}
      />

      <PredictionCard
        title="Dự đoán Lượt xem"
        icon={Eye}
        result={predictions.views}
        color="blue"
        type="views"
        loading={predictionLoading.views}
      />

      <PredictionCard
        title="Phân loại Nội dung"
        icon={Target}
        result={predictions.cluster}
        color="purple"
        type="cluster"
        loading={predictionLoading.cluster}
      />
    </div>
  );
};

export default PredictionResults;