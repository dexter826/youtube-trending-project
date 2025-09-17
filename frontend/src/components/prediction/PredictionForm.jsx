import React from "react";
import { Play, Zap, TrendingUp, Eye, Target } from "lucide-react";

const PredictionForm = ({
  videoData,
  onInputChange,
  onPredict,
  onPredictAll,
  loading,
  mlHealth,
  categories,
  isFormValid,
}) => {
  const handleInputChange = (field, value) => {
    onInputChange(field, value);
  };

  return (
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
              {Array.isArray(categories) ? (
                categories.map((category) => (
                  <option key={category.id} value={category.id}>
                    {category.name}
                  </option>
                ))
              ) : (
                Object.entries(categories || {}).map(([id, name]) => (
                  <option key={id} value={id}>
                    {name}
                  </option>
                ))
              )}
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


        </div>
      </div>

      {/* Action Buttons */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Thực hiện Dự đoán
        </h3>

        <div className="space-y-3">
          <button
            onClick={onPredictAll}
            disabled={loading || !mlHealth?.is_trained || !isFormValid}
            className="btn-primary w-full flex items-center justify-center space-x-2"
          >
            <Zap className="w-4 h-4" />
            <span>Dự đoán Tất cả</span>
          </button>

          <div className="grid grid-cols-3 gap-2">
            <button
              onClick={() => onPredict("trending")}
              disabled={loading || !mlHealth?.is_trained || !isFormValid}
              className="btn-secondary text-xs flex items-center justify-center space-x-1"
            >
              <TrendingUp className="w-3 h-3" />
              <span>Trending</span>
            </button>

            <button
              onClick={() => onPredict("views")}
              disabled={loading || !mlHealth?.is_trained || !isFormValid}
              className="btn-secondary text-xs flex items-center justify-center space-x-1"
            >
              <Eye className="w-3 h-3" />
              <span>Views</span>
            </button>

            <button
              onClick={() => onPredict("cluster")}
              disabled={loading || !mlHealth?.is_trained || !isFormValid}
              className="btn-secondary text-xs flex items-center justify-center space-x-1"
            >
              <Target className="w-3 h-3" />
              <span>Cluster</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionForm;