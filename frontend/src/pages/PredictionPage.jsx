import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  TrendingUp, 
  Eye, 
  Target,
  Zap,
  AlertCircle,
  CheckCircle,
  Play,
  BarChart3
} from 'lucide-react';
import { useApi } from '../context/ApiContext';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';

const PredictionPage = () => {
  const { 
    predictTrending, 
    predictViews, 
    predictCluster,
    checkMLHealth,
    fetchCategories,
    loading, 
    error 
  } = useApi();

  const [mlHealth, setMlHealth] = useState(null);
  const [categories, setCategories] = useState([]);
  const [videoData, setVideoData] = useState({
    title: '',
    views: 0,
    likes: 0,
    dislikes: 0,
    comment_count: 0,
    category_id: 0,
    tags: ''
  });
  const [predictions, setPredictions] = useState({
    trending: null,
    views: null,
    cluster: null
  });
  const [activeTab, setActiveTab] = useState('trending');

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const [healthData, categoriesData] = await Promise.all([
          checkMLHealth(),
          fetchCategories()
        ]);
        setMlHealth(healthData);
        setCategories(categoriesData.categories || []);
      } catch (err) {
        console.error('Failed to load initial data:', err);
      }
    };

    loadInitialData();
  }, []);

  const handleInputChange = (field, value) => {
    setVideoData(prev => ({
      ...prev,
      [field]: field.includes('count') || field === 'views' || field === 'likes' || field === 'dislikes' || field === 'category_id' 
        ? parseInt(value) || 0 
        : value
    }));
  };

  const handlePredict = async (type) => {
    try {
      let result;
      switch (type) {
        case 'trending':
          result = await predictTrending(videoData);
          setPredictions(prev => ({ ...prev, trending: result }));
          break;
        case 'views':
          result = await predictViews(videoData);
          setPredictions(prev => ({ ...prev, views: result }));
          break;
        case 'cluster':
          result = await predictCluster(videoData);
          setPredictions(prev => ({ ...prev, cluster: result }));
          break;
        default:
          break;
      }
    } catch (err) {
      console.error(`Failed to predict ${type}:`, err);
    }
  };

  const handlePredictAll = async () => {
    try {
      const [trendingResult, viewsResult, clusterResult] = await Promise.all([
        predictTrending(videoData),
        predictViews(videoData),
        predictCluster(videoData)
      ]);

      setPredictions({
        trending: trendingResult,
        views: viewsResult,
        cluster: clusterResult
      });
    } catch (err) {
      console.error('Failed to predict all:', err);
    }
  };

  const formatNumber = (num) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num?.toLocaleString() || '0';
  };

  const PredictionCard = ({ title, icon: Icon, result, color = 'blue' }) => (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Icon className={`w-5 h-5 text-${color}-600`} />
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        </div>
        {result && (
          <div className="flex items-center space-x-1">
            <CheckCircle className="w-4 h-4 text-green-500" />
            <span className="text-xs text-green-600">Hoàn thành</span>
          </div>
        )}
      </div>
      
      {result ? (
        <div className="space-y-3">
          {result.prediction && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Kết quả:</span>
              <span className={`text-lg font-bold text-${color}-600 break-words max-w-xs`}>
                {typeof result.prediction === 'object' 
                  ? JSON.stringify(result.prediction, null, 2)
                  : result.prediction
                }
              </span>
            </div>
          )}
          
          {result.trending_probability && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Xác suất Trending:</span>
              <span className={`text-lg font-bold ${
                result.trending_probability > 0.7 ? 'text-green-600' : 
                result.trending_probability > 0.4 ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {(result.trending_probability * 100).toFixed(1)}%
              </span>
            </div>
          )}
          
          {result.predicted_views && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Dự đoán lượt xem:</span>
              <span className={`text-lg font-bold text-${color}-600`}>
                {formatNumber(result.predicted_views)}
              </span>
            </div>
          )}
          
          {result.cluster_type && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Loại nội dung:</span>
              <span className={`px-2 py-1 bg-${color}-100 text-${color}-700 text-sm rounded-full`}>
                {result.cluster_type}
              </span>
            </div>
          )}
          
          {result.confidence && (
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Độ tin cậy:</span>
              <span className={`text-sm font-medium ${
                result.confidence === 'high' ? 'text-green-600' : 
                result.confidence === 'medium' ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {result.confidence === 'high' ? 'Cao' : 
                 result.confidence === 'medium' ? 'Trung bình' : 'Thấp'}
              </span>
            </div>
          )}
          
          <div className="text-xs text-gray-500 pt-2 border-t">
            Phương pháp: {result.method || 'Spark MLlib'}
          </div>
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          <Icon className={`w-12 h-12 text-gray-300 mx-auto mb-2`} />
          <p>Chưa có dự đoán</p>
        </div>
      )}
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
              <div className={`w-3 h-3 rounded-full ${
                mlHealth.is_trained ? 'bg-green-500' : 'bg-red-500'
              }`} />
              <div>
                <h3 className="text-lg font-semibold text-gray-900">
                  Trạng thái ML Models
                </h3>
                <p className="text-sm text-gray-600">
                  {mlHealth.is_trained 
                    ? `${mlHealth.total_models} mô hình đã sẵn sàng` 
                    : 'Mô hình chưa được huấn luyện'
                  }
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
                Th��ng tin Video
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
                  onChange={(e) => handleInputChange('title', e.target.value)}
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
                    onChange={(e) => handleInputChange('views', e.target.value)}
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
                    onChange={(e) => handleInputChange('likes', e.target.value)}
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
                    onChange={(e) => handleInputChange('dislikes', e.target.value)}
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
                    onChange={(e) => handleInputChange('comment_count', e.target.value)}
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
                  onChange={(e) => handleInputChange('category_id', e.target.value)}
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
                  onChange={(e) => handleInputChange('tags', e.target.value)}
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
                onClick={handlePredictAll}
                disabled={loading || !mlHealth?.is_trained}
                className="btn-primary w-full flex items-center justify-center space-x-2"
              >
                <Zap className="w-4 h-4" />
                <span>Dự đoán Tất cả</span>
              </button>
              
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => handlePredict('trending')}
                  disabled={loading || !mlHealth?.is_trained}
                  className="btn-secondary text-xs flex items-center justify-center space-x-1"
                >
                  <TrendingUp className="w-3 h-3" />
                  <span>Trending</span>
                </button>
                
                <button
                  onClick={() => handlePredict('views')}
                  disabled={loading || !mlHealth?.is_trained}
                  className="btn-secondary text-xs flex items-center justify-center space-x-1"
                >
                  <Eye className="w-3 h-3" />
                  <span>Views</span>
                </button>
                
                <button
                  onClick={() => handlePredict('cluster')}
                  disabled={loading || !mlHealth?.is_trained}
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
          {loading && <LoadingSpinner message="Đang thực hiện dự đoán..." />}
          
          <PredictionCard
            title="Dự đoán Trending"
            icon={TrendingUp}
            result={predictions.trending}
            color="green"
          />
          
          <PredictionCard
            title="Dự đoán Lượt xem"
            icon={Eye}
            result={predictions.views}
            color="blue"
          />
          
          <PredictionCard
            title="Phân loại Nội dung"
            icon={Target}
            result={predictions.cluster}
            color="purple"
          />
        </div>
      </div>

      {/* Sample Data */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <BarChart3 className="w-5 h-5 text-gray-600" />
          <h3 className="text-lg font-semibold text-gray-900">
            Dữ liệu Mẫu
          </h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button
            onClick={() => setVideoData({
              title: "Amazing Tech Review 2024 - Must Watch!",
              views: 50000,
              likes: 2500,
              dislikes: 100,
              comment_count: 500,
              category_id: 28,
              tags: "tech|review|2024|gadgets|technology"
            })}
            className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 text-left"
          >
            <h4 className="font-medium text-gray-900">Tech Review</h4>
            <p className="text-sm text-gray-600">Video công nghệ phổ biến</p>
          </button>
          
          <button
            onClick={() => setVideoData({
              title: "Funny Cat Compilation - Hilarious Moments",
              views: 100000,
              likes: 8000,
              dislikes: 200,
              comment_count: 1200,
              category_id: 23,
              tags: "funny|cats|pets|compilation|animals"
            })}
            className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 text-left"
          >
            <h4 className="font-medium text-gray-900">Entertainment</h4>
            <p className="text-sm text-gray-600">Video giải trí về thú cưng</p>
          </button>
          
          <button
            onClick={() => setVideoData({
              title: "How to Learn Programming in 2024",
              views: 25000,
              likes: 1800,
              dislikes: 50,
              comment_count: 300,
              category_id: 27,
              tags: "programming|tutorial|education|coding|learn"
            })}
            className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 text-left"
          >
            <h4 className="font-medium text-gray-900">Education</h4>
            <p className="text-sm text-gray-600">Video giáo dục lập trình</p>
          </button>
        </div>
      </div>
    </div>
  );
};

export default PredictionPage;