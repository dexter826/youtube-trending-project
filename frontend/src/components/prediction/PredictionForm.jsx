import React, { useState } from "react";
import { Play, Zap, Loader2 } from "lucide-react";

const PredictionForm = ({ onPredictByUrl, loading, mlHealth }) => {
  const [url, setUrl] = useState("");

  const isValid = () =>
    url.trim().length > 0 && (mlHealth?.is_trained ?? false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!isValid()) return;
    onPredictByUrl(url.trim());
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
              onChange={(e) => setUrl(e.target.value)}
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
      </div>
    </div>
  );
};

export default PredictionForm;
