import React, { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';

const MLAnalysisPanel = () => {
  const [mlData, setMlData] = useState({
    clustering: null,
    predictions: null,
    sentiment: null,
    anomalies: null,
    categories: null
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('clustering');

  useEffect(() => {
    fetchMLAnalysis();
  }, []);

  const fetchMLAnalysis = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch all ML analysis data
      const [clustering, predictions, sentiment, anomalies, categories] = await Promise.all([
        apiService.getMLClustering(),
        apiService.getMLPredictions(), 
        apiService.getMLSentiment(),
        apiService.getMLAnomalies(),
        apiService.getMLCategories()
      ]);

      setMlData({
        clustering: clustering.clustering_results,
        predictions: predictions.prediction_model,
        sentiment: sentiment.sentiment_analysis,
        anomalies: anomalies.anomaly_detection,
        categories: categories.category_analysis
      });

    } catch (err) {
      console.error('Error fetching ML analysis:', err);
      setError('Không thể tải dữ liệu phân tích ML. Vui lòng chạy Spark job trước.');
    } finally {
      setLoading(false);
    }
  };

  const TabButton = ({ tabKey, label, isActive, onClick }) => (
    <button
      onClick={() => onClick(tabKey)}
      className={`px-4 py-2 font-medium text-sm rounded-lg transition-colors ${
        isActive
          ? 'bg-blue-500 text-white'
          : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
      }`}
    >
      {label}
    </button>
  );

  const ClusteringView = () => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">Phân cụm Video</h3>
      <p className="text-gray-600">Video được nhóm theo mức độ tương tác sử dụng K-Means clustering</p>
      
      {mlData.clustering && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {mlData.clustering.map((cluster, index) => (
            <div key={index} className="bg-white border border-gray-200 rounded-lg p-4">
              <h4 className="font-medium text-gray-800 mb-2">Cụm {cluster.cluster}</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Số video:</span>
                  <span className="font-medium">{cluster.video_count?.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">TB lượt xem:</span>
                  <span className="font-medium">{Math.round(cluster.avg_views)?.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">TB lượt thích:</span>
                  <span className="font-medium">{Math.round(cluster.avg_likes)?.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Tỷ lệ thích:</span>
                  <span className="font-medium">{(cluster.avg_like_rate * 100)?.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Điểm tương tác:</span>
                  <span className="font-medium">{(cluster.avg_engagement * 100)?.toFixed(3)}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const PredictionsView = () => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">Mô hình Dự đoán Độ phổ biến</h3>
      <p className="text-gray-600">Random Forest dự đoán độ phổ biến video dựa trên metadata</p>
      
      {mlData.predictions && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {(mlData.predictions.model_accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Độ chính xác mô hình</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {mlData.predictions.total_predictions?.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Tổng dự đoán test</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-orange-600">
                {mlData.predictions.low_threshold?.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Ngưỡng độ phổ biến thấp</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-red-600">
                {mlData.predictions.high_threshold?.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Ngưỡng độ phổ biến cao</div>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-gray-50 rounded">
            <h4 className="font-medium text-gray-800 mb-2">Phân loại độ phổ biến:</h4>
            <div className="text-sm space-y-1">
              <div><span className="font-medium">Thấp:</span> ≤ {mlData.predictions.low_threshold?.toLocaleString()} lượt xem</div>
              <div><span className="font-medium">Trung bình:</span> {mlData.predictions.low_threshold?.toLocaleString()} - {mlData.predictions.high_threshold?.toLocaleString()} lượt xem</div>
              <div><span className="font-medium">Cao:</span> &gt; {mlData.predictions.high_threshold?.toLocaleString()} lượt xem</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const SentimentView = () => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">Phân tích Cảm xúc Tiêu đề</h3>
      <p className="text-gray-600">Phân tích tông cảm xúc và pattern clickbait trong tiêu đề video</p>
      
      {mlData.sentiment && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className={`text-2xl font-bold ${mlData.sentiment.avg_sentiment > 0 ? 'text-green-600' : 'text-red-600'}`}>
                {mlData.sentiment.avg_sentiment?.toFixed(3)}
              </div>
              <div className="text-sm text-gray-600">Điểm cảm xúc trung bình</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {mlData.sentiment.avg_positive?.toFixed(2)}
              </div>
              <div className="text-sm text-gray-600">Từ tích cực TB/video</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {mlData.sentiment.avg_negative?.toFixed(2)}
              </div>
              <div className="text-sm text-gray-600">Từ tiêu cực TB/video</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {mlData.sentiment.avg_clickbait?.toFixed(2)}
              </div>
              <div className="text-sm text-gray-600">Điểm clickbait TB</div>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-gray-50 rounded">
            <h4 className="font-medium text-gray-800 mb-2">Giải thích:</h4>
            <div className="text-sm space-y-1">
              <div><span className="font-medium">Điểm cảm xúc:</span> Từ tích cực - từ tiêu cực (điểm dương = tích cực)</div>
              <div><span className="font-medium">Clickbait:</span> Số từ khóa thu hút click như "shocking", "unbelievable"</div>
              <div><span className="font-medium">Tổng video:</span> {mlData.sentiment.total_videos?.toLocaleString()} video được phân tích</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const AnomaliesView = () => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">Phát hiện Video Bất thường</h3>
      <p className="text-gray-600">Video có pattern tương tác bất thường (outliers)</p>
      
      {mlData.anomalies && (
        <div className="space-y-4">
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h4 className="font-medium text-gray-800 mb-3">Thống kê Tổng quan</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">
                  {mlData.anomalies.summary?.anomalous_count}
                </div>
                <div className="text-sm text-gray-600">Video bất thường</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-blue-600">
                  {mlData.anomalies.summary?.max_views?.toLocaleString()}
                </div>
                <div className="text-sm text-gray-600">Lượt xem cao nhất</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-green-600">
                  {mlData.anomalies.summary?.max_likes?.toLocaleString()}
                </div>
                <div className="text-sm text-gray-600">Lượt thích cao nhất</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-orange-600">
                  {Math.round(mlData.anomalies.summary?.avg_dislikes)?.toLocaleString()}
                </div>
                <div className="text-sm text-gray-600">TB không thích</div>
              </div>
            </div>
          </div>

          {mlData.anomalies.top_anomalies && mlData.anomalies.top_anomalies.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h4 className="font-medium text-gray-800 mb-3">Top Video Bất thường nhất</h4>
              <div className="space-y-3">
                {mlData.anomalies.top_anomalies.slice(0, 5).map((video, index) => (
                  <div key={index} className="border border-gray-100 rounded p-3 hover:bg-gray-50">
                    <div className="font-medium text-sm text-gray-800 mb-1 line-clamp-2">
                      {video.title}
                    </div>
                    <div className="flex flex-wrap gap-4 text-xs text-gray-600">
                      <span>👁️ {video.views?.toLocaleString()} views</span>
                      <span>👍 {video.likes?.toLocaleString()} likes</span>
                      <span>👎 {video.dislikes?.toLocaleString()} dislikes</span>
                      <span>🌍 {video.country}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="bg-yellow-50 border border-yellow-200 rounded p-4">
            <h4 className="font-medium text-yellow-800 mb-2">Tiêu chí Phát hiện Bất thường:</h4>
            <div className="text-sm text-yellow-700 space-y-1">
              <div>• Lượt xem vượt quá 3 độ lệch chuẩn so với trung bình</div>
              <div>• Lượt thích vượt quá 3 độ lệch chuẩn so với trung bình</div>
              <div>• Tỷ lệ dislike/(like+1) &gt; 50% (video gây tranh cãi)</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const CategoriesView = () => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">Phân tích Hiệu suất theo Danh mục</h3>
      <p className="text-gray-600">So sánh hiệu suất và pattern tương tác giữa các danh mục video</p>
      
      {mlData.categories && (
        <div className="space-y-4">
          {mlData.categories.category_performance && (
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h4 className="font-medium text-gray-800 mb-3">Bảng xếp hạng Danh mục</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left p-2 font-medium text-gray-600">Hạng</th>
                      <th className="text-left p-2 font-medium text-gray-600">Danh mục</th>
                      <th className="text-right p-2 font-medium text-gray-600">Số video</th>
                      <th className="text-right p-2 font-medium text-gray-600">TB lượt xem</th>
                      <th className="text-right p-2 font-medium text-gray-600">Tỷ lệ thích</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mlData.categories.category_performance.slice(0, 10).map((category, index) => (
                      <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="p-2 font-medium text-gray-800">#{index + 1}</td>
                        <td className="p-2">
                          <span className="font-medium">Danh mục {category.category_id}</span>
                        </td>
                        <td className="p-2 text-right">{category.video_count?.toLocaleString()}</td>
                        <td className="p-2 text-right font-medium">{Math.round(category.avg_views)?.toLocaleString()}</td>
                        <td className="p-2 text-right">
                          <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                            {(category.avg_like_rate * 100)?.toFixed(2)}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {mlData.categories.best_category && mlData.categories.worst_category && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <h4 className="font-medium text-green-800 mb-3">🏆 Danh mục Hiệu suất Cao nhất</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Danh mục:</span>
                    <span className="font-medium">{mlData.categories.best_category.category_id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>TB lượt xem:</span>
                    <span className="font-medium">{Math.round(mlData.categories.best_category.avg_views)?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Số video:</span>
                    <span className="font-medium">{mlData.categories.best_category.video_count}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Tỷ lệ thích:</span>
                    <span className="font-medium">{(mlData.categories.best_category.avg_like_rate * 100)?.toFixed(2)}%</span>
                  </div>
                </div>
              </div>
              
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <h4 className="font-medium text-red-800 mb-3">📉 Danh mục Hiệu suất Thấp nhất</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Danh mục:</span>
                    <span className="font-medium">{mlData.categories.worst_category.category_id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>TB lượt xem:</span>
                    <span className="font-medium">{Math.round(mlData.categories.worst_category.avg_views)?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Số video:</span>
                    <span className="font-medium">{mlData.categories.worst_category.video_count}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Tỷ lệ thích:</span>
                    <span className="font-medium">{(mlData.categories.worst_category.avg_like_rate * 100)?.toFixed(2)}%</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="bg-blue-50 border border-blue-200 rounded p-4">
            <h4 className="font-medium text-blue-800 mb-2">📊 Thống kê Tổng quan:</h4>
            <div className="text-sm text-blue-700 space-y-1">
              <div>• Tổng số danh mục: {mlData.categories.total_categories} danh mục</div>
              <div>• Xếp hạng dựa trên trung bình lượt xem</div>
              <div>• Tỷ lệ thích = (Tổng lượt thích / Tổng lượt xem) × 100%</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-3 text-gray-600">🤖 Đang tải phân tích Machine Learning...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="text-center py-8">
          <div className="text-6xl mb-4">⚠️</div>
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Lỗi tải dữ liệu ML</h3>
          <div className="text-gray-600 mb-4">{error}</div>
          <button
            onClick={fetchMLAnalysis}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            🔄 Thử lại
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200">
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-bold text-gray-800 mb-4">🤖 Phân tích Machine Learning</h2>
        <p className="text-gray-600 mb-4">
          Kết quả phân tích sâu sử dụng các thuật toán Machine Learning: clustering, prediction, sentiment analysis, anomaly detection
        </p>
        
        {/* Tab Navigation */}
        <div className="flex flex-wrap gap-2">
          <TabButton
            tabKey="clustering"
            label="🎯 Phân cụm Video"
            isActive={activeTab === 'clustering'}
            onClick={setActiveTab}
          />
          <TabButton
            tabKey="predictions"
            label="🔮 Dự đoán Độ phổ biến"
            isActive={activeTab === 'predictions'}
            onClick={setActiveTab}
          />
          <TabButton
            tabKey="sentiment"
            label="💭 Phân tích Cảm xúc"
            isActive={activeTab === 'sentiment'}
            onClick={setActiveTab}
          />
          <TabButton
            tabKey="anomalies"
            label="🕵️ Phát hiện Bất thường"
            isActive={activeTab === 'anomalies'}
            onClick={setActiveTab}
          />
          <TabButton
            tabKey="categories"
            label="📈 Phân tích Danh mục"
            isActive={activeTab === 'categories'}
            onClick={setActiveTab}
          />
        </div>
      </div>

      <div className="p-6">
        {activeTab === 'clustering' && <ClusteringView />}
        {activeTab === 'predictions' && <PredictionsView />}
        {activeTab === 'sentiment' && <SentimentView />}
        {activeTab === 'anomalies' && <AnomaliesView />}
        {activeTab === 'categories' && <CategoriesView />}
      </div>
    </div>
  );
};

export default MLAnalysisPanel;
