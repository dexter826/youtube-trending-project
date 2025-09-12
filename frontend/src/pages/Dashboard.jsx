import React, { useState, useEffect, useCallback } from 'react';
import { 
  TrendingUp, 
  Eye, 
  ThumbsUp, 
  MessageCircle, 
  Database,
  Activity,
  Globe,
  Calendar,
  RefreshCw
} from 'lucide-react';
import { useApi } from '../context/ApiContext';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';

const Dashboard = () => {
  const { 
    fetchStatistics, 
    fetchCountries, 
    fetchDatabaseStats,
    checkMLHealth,
    loading, 
    error 
  } = useApi();

  const [statistics, setStatistics] = useState(null);
  const [countries, setCountries] = useState([]);
  const [selectedCountry, setSelectedCountry] = useState('');
  const [dbStats, setDbStats] = useState(null);
  const [mlHealth, setMlHealth] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const loadDashboardData = useCallback(async () => {
    try {
      // Load all dashboard data
      const [statsData, countriesData, dbData, mlData] = await Promise.all([
        fetchStatistics(selectedCountry || null),
        fetchCountries(),
        fetchDatabaseStats(),
        checkMLHealth()
      ]);

      setStatistics(statsData);
      setCountries(countriesData.countries || []);
      setDbStats(dbData);
      setMlHealth(mlData);
      setLastUpdated(new Date());
    } catch (err) {
      // Error handled by ApiContext
    }
  }, [selectedCountry, fetchStatistics, fetchCountries, fetchDatabaseStats, checkMLHealth]);

  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  const formatNumber = (num) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num?.toLocaleString() || '0';
  };

  const StatCard = ({ title, value, icon: Icon, color = 'blue', subtitle }) => (
    <div className="stat-card">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className={`text-2xl font-bold text-${color}-600`}>
            {typeof value === 'number' ? formatNumber(value) : value || '0'}
          </p>
          {subtitle && (
            <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
          )}
        </div>
        <div className={`p-3 bg-${color}-100 rounded-lg`}>
          <Icon className={`w-6 h-6 text-${color}-600`} />
        </div>
      </div>
    </div>
  );

  if (loading && !statistics) {
    return <LoadingSpinner message="Đang tải dữ liệu dashboard..." />;
  }

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            Dashboard Analytics
          </h1>
          <p className="mt-2 text-gray-600">
            Tổng quan về dữ liệu YouTube Trending và hiệu suất mô hình ML
          </p>
        </div>
        
        <div className="mt-4 sm:mt-0 flex items-center space-x-4">
          {/* Country Filter */}
          <select
            value={selectedCountry}
            onChange={(e) => setSelectedCountry(e.target.value)}
            className="input-field w-48"
          >
            <option value="">Tất cả quốc gia</option>
            {countries.map((country) => (
              <option key={country} value={country}>
                {country.toUpperCase()}
              </option>
            ))}
          </select>
          
          {/* Refresh Button */}
          <button
            onClick={loadDashboardData}
            disabled={loading}
            className="btn-secondary flex items-center space-x-2"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Làm mới</span>
          </button>
        </div>
      </div>

      {error && <ErrorMessage message={error} />}

      {/* Statistics Cards */}
      {statistics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard
            title="Tổng số Video"
            value={statistics.statistics?.total_videos}
            icon={TrendingUp}
            color="blue"
            subtitle={selectedCountry ? `Quốc gia: ${selectedCountry.toUpperCase()}` : 'Toàn cầu'}
          />
          <StatCard
            title="Lượt xem trung bình"
            value={statistics.statistics?.avg_views}
            icon={Eye}
            color="green"
          />
          <StatCard
            title="Lượt thích trung bình"
            value={statistics.statistics?.avg_likes}
            icon={ThumbsUp}
            color="red"
          />
          <StatCard
            title="Bình luận trung bình"
            value={statistics.statistics?.avg_comments}
            icon={MessageCircle}
            color="purple"
          />
        </div>
      )}

      {/* System Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Database Status */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 flex items-center">
              <Database className="w-5 h-5 mr-2 text-blue-600" />
              Trạng thái Database
            </h3>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-sm text-green-600">Hoạt động</span>
            </div>
          </div>
          
          {dbStats && (
            <div className="space-y-3">
              {Object.entries(dbStats.collections || {}).map(([collection, count]) => (
                <div key={collection} className="flex justify-between items-center py-2 border-b border-gray-100 last:border-b-0">
                  <span className="text-sm text-gray-600 capitalize">
                    {collection.replace('_', ' ')}
                  </span>
                  <span className="text-sm font-medium text-gray-900">
                    {formatNumber(count)} records
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ML Model Status */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 flex items-center">
              <Activity className="w-5 h-5 mr-2 text-purple-600" />
              Trạng thái ML Models
            </h3>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                mlHealth?.is_trained ? 'bg-green-500' : 'bg-yellow-500'
              }`}></div>
              <span className={`text-sm ${
                mlHealth?.is_trained ? 'text-green-600' : 'text-yellow-600'
              }`}>
                {mlHealth?.is_trained ? 'Đã huấn luyện' : 'Chưa huấn luyện'}
              </span>
            </div>
          </div>
          
          {mlHealth && (
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Framework</span>
                <span className="text-sm font-medium text-gray-900">
                  {mlHealth.framework || 'Spark MLlib'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Loaded Models</span>
                <span className="text-sm font-medium text-gray-900">
                  {mlHealth.total_models || 0}/3
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Storage</span>
                <span className="text-sm font-medium text-gray-900">
                  {mlHealth.storage || 'HDFS'}
                </span>
              </div>
              {mlHealth.loaded_models && (
                <div className="mt-3">
                  <p className="text-xs text-gray-500 mb-2">Available Models:</p>
                  <div className="flex flex-wrap gap-1">
                    {mlHealth.loaded_models.map((model) => (
                      <span
                        key={model}
                        className="px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full"
                      >
                        {model.replace('_', ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card text-center">
          <Globe className="w-8 h-8 text-blue-600 mx-auto mb-3" />
          <h3 className="text-lg font-semibold text-gray-900">Quốc gia</h3>
          <p className="text-2xl font-bold text-blue-600">{countries.length}</p>
          <p className="text-sm text-gray-500">Đang theo dõi</p>
        </div>
        
        <div className="card text-center">
          <Calendar className="w-8 h-8 text-green-600 mx-auto mb-3" />
          <h3 className="text-lg font-semibold text-gray-900">Cập nhật</h3>
          <p className="text-sm font-medium text-green-600">
            {lastUpdated ? lastUpdated.toLocaleTimeString('vi-VN') : 'Chưa có'}
          </p>
          <p className="text-sm text-gray-500">Lần cuối</p>
        </div>
        
        <div className="card text-center">
          <TrendingUp className="w-8 h-8 text-purple-600 mx-auto mb-3" />
          <h3 className="text-lg font-semibold text-gray-900">Hiệu suất</h3>
          <p className="text-2xl font-bold text-purple-600">
            {statistics?.statistics?.max_views ? formatNumber(statistics.statistics.max_views) : '0'}
          </p>
          <p className="text-sm text-gray-500">Lượt xem cao nhất</p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;