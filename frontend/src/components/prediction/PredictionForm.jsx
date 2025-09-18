import React, { useState } from "react";
import { Play, Zap, Loader2 } from "lucide-react";

const PredictionForm = ({
  onPredictByUrl,
  loading,
  mlHealth,
  videoMetadata,
}) => {
  const [url, setUrl] = useState("");
  const [predictedUrl, setPredictedUrl] = useState("");

  const formatNumber = (num) => {
    if (!num && num !== 0) return "N/A";
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + "M";
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + "K";
    }
    return num.toLocaleString();
  };

  const isValid = () =>
    url.trim().length > 0 && (mlHealth?.is_trained ?? false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isValid()) return;
    const trimmedUrl = url.trim();
    setPredictedUrl(trimmedUrl);
    onPredictByUrl(trimmedUrl);
  };

  const handleUrlChange = (e) => {
    const newUrl = e.target.value;
    setUrl(newUrl);
    // Clear predicted URL if it doesn't match current URL
    if (predictedUrl && newUrl.trim() !== predictedUrl) {
      setPredictedUrl("");
    }
  };

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="flex items-center space-x-2 mb-6">
          <Play className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">
            Dự đoán từ YouTube URL
          </h3>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Link YouTube
            </label>
            <input
              type="url"
              value={url}
              onChange={handleUrlChange}
              placeholder="https://www.youtube.com/watch?v=..."
              className="input-field"
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading || !isValid()}
            className="btn-primary w-full flex items-center justify-center space-x-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Đang dự đoán...</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                <span>Dự đoán</span>
              </>
            )}
          </button>
        </form>

        {/* Video Metadata Display */}
        {videoMetadata && url.trim() === predictedUrl && (
          <div className="mt-6 pt-6 border-t border-gray-200">
            <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
              <Play className="w-5 h-5 text-green-600" />
              <span>Metadata Video</span>
            </h4>

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
                  <tr>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                      Tiêu đề
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-900">
                      {videoMetadata.title}
                    </td>
                  </tr>
                  <tr>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                      Video ID
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-900 font-mono">
                      {videoMetadata.id}
                    </td>
                  </tr>
                  <tr>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                      Lượt xem
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-900">
                      {formatNumber(videoMetadata.views)}
                    </td>
                  </tr>
                  <tr>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                      Lượt thích
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-900">
                      {formatNumber(videoMetadata.likes)}
                    </td>
                  </tr>
                  <tr>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                      Bình luận
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-900">
                      {formatNumber(videoMetadata.comment_count)}
                    </td>
                  </tr>
                  <tr>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                      Category ID
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-900">
                      {videoMetadata.category_id}
                    </td>
                  </tr>
                  <tr>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                      Giờ đăng
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-900">
                      {videoMetadata.publish_hour}:00
                    </td>
                  </tr>
                  <tr>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-600">
                      Tuổi video (proxy)
                    </td>
                    <td className="border border-gray-300 px-4 py-2 text-sm text-gray-900">
                      {videoMetadata.video_age_proxy}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionForm;
