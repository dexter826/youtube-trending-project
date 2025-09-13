import React from "react";
import { BarChart3 } from "lucide-react";

const formatNumber = (num) => {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + "M";
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + "K";
  }
  return num?.toLocaleString() || "0";
};

const PredictionHistory = ({
  samplePredictions,
  selectedSample,
  onLoadSample,
}) => {
  return (
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
          onClick={() => onLoadSample("tech")}
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
          onClick={() => onLoadSample("entertainment")}
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
          onClick={() => onLoadSample("trending")}
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
  );
};

export default PredictionHistory;