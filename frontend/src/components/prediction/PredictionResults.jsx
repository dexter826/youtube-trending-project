import React from "react";
import { TrendingUp, Target, CheckCircle } from "lucide-react";

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
      {result && !loading && (
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
          {/* Hiển thị kết quả dướidạng bảng */}
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
                {/* Sử dụng result thay vì result.prediction */}
                {result?.predicted_days !== undefined && (
                  <tr>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                      Số ngày dự kiến nằm trong Trending
                    </td>
                    <td
                      className={`border border-gray-300 px-4 py-2 text-sm font-bold text-${color}-600`}
                    >
                      {result.predicted_days}
                    </td>
                  </tr>
                )}

                {result?.cluster !== undefined && (
                  <tr>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                      Cluster
                    </td>
                    <td
                      className={`border border-gray-300 px-4 py-2 text-sm font-bold text-${color}-600`}
                    >
                      {result.cluster}
                    </td>
                  </tr>
                )}
                {result?.cluster_type && (
                  <tr>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                      Loại nội dung
                    </td>
                    <td className={`border border-gray-300 px-4 py-2 text-sm`}>
                      <span
                        className={`px-2 py-1 bg-${color}-100 text-${color}-700 text-sm rounded-full`}
                      >
                        {result.cluster_type}
                      </span>
                    </td>
                  </tr>
                )}
                {result?.confidence && (
                  <tr>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                      Độ tin cậy
                    </td>
                    <td
                      className={`border border-gray-300 px-4 py-2 text-sm font-medium ${
                        result.confidence === "high"
                          ? "text-green-600"
                          : result.confidence === "medium"
                          ? "text-yellow-600"
                          : "text-red-600"
                      }`}
                    >
                      {result.confidence === "high"
                        ? "Cao"
                        : result.confidence === "medium"
                        ? "Trung bình"
                        : "Thấp"}
                    </td>
                  </tr>
                )}
                {result?.method && (
                  <tr>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                      Phương pháp
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-500">
                      {result.method}
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

// No number formatting needed currently

const PredictionResults = ({ predictions, predictionLoading }) => {
  return (
    <div className="space-y-6">
      <PredictionCard
        title="Dự đoán Số ngày Trending"
        icon={TrendingUp}
        result={predictions.days}
        color="blue"
        type="days"
        loading={predictionLoading.days}
      />

      <PredictionCard
        title="Phân cụm Nội dung"
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
