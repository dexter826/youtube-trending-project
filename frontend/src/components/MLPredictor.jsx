import React, { useState, useEffect } from 'react';
import { Sparkles, Brain, AlertCircle, CheckCircle2, XCircle } from 'lucide-react';
import apiService from '../services/apiService';
import LoadingSpinner from './LoadingSpinner';

const MLPredictor = () => {
  const [videoData, setVideoData] = useState({
    title: '',
    views: 1000,
    likes: 100,
    dislikes: 5,
    comment_count: 10,
    category_id: 28,
    tags: '',
    publish_time: new Date().toISOString(),
    comments_disabled: false,
    ratings_disabled: false
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [quickMode, setQuickMode] = useState(true);

  // Categories mapping
  const categories = {
    1: 'Film & Animation',
    2: 'Autos & Vehicles', 
    10: 'Music',
    15: 'Pets & Animals',
    17: 'Sports',
    19: 'Travel & Events',
    20: 'Gaming',
    22: 'People & Blogs',
    23: 'Comedy',
    24: 'Entertainment',
    25: 'News & Politics',
    26: 'Howto & Style',
    27: 'Education',
    28: 'Science & Technology'
  };

  useEffect(() => {
    loadModelInfo();
  }, []);

  const loadModelInfo = async () => {
    try {
      const info = await apiService.getModelInfo();
      setModelInfo(info);
    } catch (err) {
      console.error('Failed to load model info:', err);
    }
  };

  const handleInputChange = (field, value) => {
    setVideoData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleQuickPredict = async () => {
    if (!videoData.title.trim()) {
      setError('Vui lòng nhập tiêu đề video');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await apiService.quickPredictTrending(
        videoData.title,
        videoData.views,
        videoData.likes,
        videoData.comment_count,
        videoData.category_id
      );
      
      setPrediction(result);
    } catch (err) {
      setError(`Dự đoán thất bại: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleFullPredict = async () => {
    if (!videoData.title.trim()) {
      setError('Vui lòng nhập tiêu đề video');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await apiService.predictTrending(videoData);
      setPrediction(result);
    } catch (err) {
      setError(`Dự đoán thất bại: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getPredictionColor = (probability) => {
    if (probability > 0.7) return 'text-green-600';
    if (probability > 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getPredictionIcon = (isTrending) => {
    return isTrending ? (
      <CheckCircle2 className="w-8 h-8 text-green-500" />
    ) : (
      <XCircle className="w-8 h-8 text-red-500" />
    );
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-purple-100 rounded-lg">
          <Brain className="w-6 h-6 text-purple-600" />
        </div>
        <div>
          <h3 className="text-xl font-semibold text-gray-900">
            🤖 AI Trending Predictor
          </h3>
          <p className="text-sm text-gray-600">
            Dự đoán khả năng video sẽ trending bằng Machine Learning
          </p>
        </div>
      </div>

      {/* Mode Toggle */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setQuickMode(true)}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            quickMode 
              ? 'bg-purple-600 text-white' 
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          ⚡ Dự đoán nhanh
        </button>
        <button
          onClick={() => setQuickMode(false)}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            !quickMode 
              ? 'bg-purple-600 text-white' 
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          🔧 Dự đoán chi tiết
        </button>
      </div>

      {/* Input Form */}
      <div className="space-y-4 mb-6">
        {/* Title - Always visible */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Tiêu đề video *
          </label>
          <input
            type="text"
            value={videoData.title}
            onChange={(e) => handleInputChange('title', e.target.value)}
            placeholder="Nhập tiêu đề video..."
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        </div>

        {/* Basic metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Lượt xem
            </label>
            <input
              type="number"
              value={videoData.views}
              onChange={(e) => handleInputChange('views', parseInt(e.target.value) || 0)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Likes
            </label>
            <input
              type="number"
              value={videoData.likes}
              onChange={(e) => handleInputChange('likes', parseInt(e.target.value) || 0)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Comments
            </label>
            <input
              type="number"
              value={videoData.comment_count}
              onChange={(e) => handleInputChange('comment_count', parseInt(e.target.value) || 0)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Category
            </label>
            <select
              value={videoData.category_id}
              onChange={(e) => handleInputChange('category_id', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
            >
              {Object.entries(categories).map(([id, name]) => (
                <option key={id} value={id}>{name}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Advanced fields for full mode */}
        {!quickMode && (
          <div className="space-y-4 border-t pt-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Dislikes
                </label>
                <input
                  type="number"
                  value={videoData.dislikes}
                  onChange={(e) => handleInputChange('dislikes', parseInt(e.target.value) || 0)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Publish Time
                </label>
                <input
                  type="datetime-local"
                  value={videoData.publish_time.slice(0, 16)}
                  onChange={(e) => handleInputChange('publish_time', new Date(e.target.value).toISOString())}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Tags (phân cách bằng |)
              </label>
              <input
                type="text"
                value={videoData.tags}
                onChange={(e) => handleInputChange('tags', e.target.value)}
                placeholder="tag1|tag2|tag3"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
              />
            </div>

            <div className="flex gap-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={videoData.comments_disabled}
                  onChange={(e) => handleInputChange('comments_disabled', e.target.checked)}
                  className="mr-2"
                />
                Comments disabled
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={videoData.ratings_disabled}
                  onChange={(e) => handleInputChange('ratings_disabled', e.target.checked)}
                  className="mr-2"
                />
                Ratings disabled
              </label>
            </div>
          </div>
        )}
      </div>

      {/* Predict Button */}
      <button
        onClick={quickMode ? handleQuickPredict : handleFullPredict}
        disabled={loading || !videoData.title.trim()}
        className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
      >
        {loading ? (
          <LoadingSpinner size="small" />
        ) : (
          <Sparkles className="w-5 h-5" />
        )}
        {loading ? 'Đang dự đoán...' : '🚀 Dự đoán Trending'}
      </button>

      {/* Error Message */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-red-500" />
          <span className="text-red-700">{error}</span>
        </div>
      )}

      {/* Prediction Results */}
      {prediction && (
        <div className="mt-6 space-y-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-gray-900">Kết quả dự đoán</h4>
              {quickMode ? (
                <span className={`font-bold text-lg ${prediction.is_trending ? 'text-green-600' : 'text-red-600'}`}>
                  {prediction.trending_probability}
                </span>
              ) : (
                getPredictionIcon(prediction.prediction?.is_trending)
              )}
            </div>

            {quickMode ? (
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Kết quả:</span>
                  <span className={`font-semibold ${prediction.is_trending ? 'text-green-600' : 'text-red-600'}`}>
                    {prediction.is_trending ? '✅ Có khả năng Trending' : '❌ Khó trending'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Độ tin cậy:</span>
                  <span className="font-medium">{prediction.confidence}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Đánh giá:</span>
                  <span className="font-medium">{prediction.recommendation}</span>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Khả năng trending:</span>
                  <span className={`font-bold text-lg ${getPredictionColor(prediction.prediction?.trending_probability)}`}>
                    {(prediction.prediction?.trending_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Độ tin cậy:</span>
                  <span className="font-medium">{prediction.prediction?.confidence_level}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Model sử dụng:</span>
                  <span className="font-medium">{prediction.model_info?.model_used}</span>
                </div>
              </div>
            )}

            {/* Recommendations */}
            {prediction.advice && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <h5 className="font-medium text-gray-900 mb-2">💡 Gợi ý cải thiện:</h5>
                <ul className="space-y-1">
                  {prediction.advice.map((advice, index) => (
                    <li key={index} className="text-sm text-gray-600 flex items-start gap-2">
                      <span className="text-purple-500">•</span>
                      {advice}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {!quickMode && prediction.recommendation?.recommendations && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <h5 className="font-medium text-gray-900 mb-2">💡 Gợi ý chi tiết:</h5>
                <ul className="space-y-1">
                  {prediction.recommendation.recommendations.map((rec, index) => (
                    <li key={index} className="text-sm text-gray-600 flex items-start gap-2">
                      <span className="text-purple-500">•</span>
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Model Info */}
      {modelInfo && (
        <div className="mt-4 text-xs text-gray-500 bg-gray-50 rounded p-3">
          <div className="flex justify-between items-center">
            <span>Models available: {modelInfo.available_models?.join(', ')}</span>
            <span>Trained: {new Date(modelInfo.trained_at).toLocaleDateString('vi-VN')}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MLPredictor;
