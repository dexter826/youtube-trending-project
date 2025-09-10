import React, { useState } from 'react';
import { Brain, Sparkles, TrendingUp, TrendingDown } from 'lucide-react';
import apiService from '../services/apiService';

const QuickMLPredictor = () => {
  const [title, setTitle] = useState('');
  const [views, setViews] = useState(1000);
  const [likes, setLikes] = useState(100);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    if (!title.trim()) return;

    setLoading(true);
    try {
      const prediction = await apiService.quickPredictTrending(title, views, likes);
      setResult(prediction);
    } catch (error) {
      console.error('Prediction failed:', error);
      setResult({ error: 'Dự đoán thất bại' });
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setTitle('');
    setViews(1000);
    setLikes(100);
    setResult(null);
  };

  return (
    <div className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg p-6 border border-purple-200">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-purple-100 rounded-lg">
          <Brain className="w-5 h-5 text-purple-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            🤖 AI Trending Predictor
          </h3>
          <p className="text-sm text-gray-600">
            Dự đoán nhanh khả năng video trending
          </p>
        </div>
      </div>

      <div className="space-y-3">
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Nhập tiêu đề video..."
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
        />
        
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Views</label>
            <input
              type="number"
              value={views}
              onChange={(e) => setViews(parseInt(e.target.value) || 0)}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Likes</label>
            <input
              type="number"
              value={likes}
              onChange={(e) => setLikes(parseInt(e.target.value) || 0)}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
            />
          </div>
        </div>

        <button
          onClick={handlePredict}
          disabled={loading || !title.trim()}
          className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          {loading ? (
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
          ) : (
            <Sparkles className="w-4 h-4" />
          )}
          {loading ? 'Dự đoán...' : 'Dự đoán'}
        </button>

        {result && !result.error && (
          <div className={`p-4 rounded-lg border-2 ${
            result.is_trending 
              ? 'bg-green-50 border-green-200' 
              : 'bg-red-50 border-red-200'
          }`}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {result.is_trending ? (
                  <TrendingUp className="w-5 h-5 text-green-600" />
                ) : (
                  <TrendingDown className="w-5 h-5 text-red-600" />
                )}
                <span className={`font-semibold ${
                  result.is_trending ? 'text-green-700' : 'text-red-700'
                }`}>
                  {result.is_trending ? 'Có khả năng Trending!' : 'Khó Trending'}
                </span>
              </div>
              <span className={`text-lg font-bold ${
                result.is_trending ? 'text-green-600' : 'text-red-600'
              }`}>
                {result.trending_probability}
              </span>
            </div>
            
            <div className="text-sm text-gray-600 mb-3">
              <strong>Độ tin cậy:</strong> {result.confidence}
            </div>

            {result.advice && result.advice.length > 0 && (
              <div className="text-sm">
                <div className="font-medium text-gray-700 mb-1">💡 Gợi ý:</div>
                <ul className="space-y-1">
                  {result.advice.slice(0, 2).map((advice, index) => (
                    <li key={index} className="text-gray-600 text-xs flex items-start gap-1">
                      <span className="text-purple-500 mt-1">•</span>
                      <span>{advice}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <button
              onClick={resetForm}
              className="mt-3 text-xs text-purple-600 hover:text-purple-700 underline"
            >
              Dự đoán video khác
            </button>
          </div>
        )}

        {result && result.error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="text-red-700 text-sm">{result.error}</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default QuickMLPredictor;
