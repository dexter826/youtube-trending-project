import React, { useState, useEffect, useCallback } from "react";
import {
  TrendingUp,
  Eye,
  ThumbsUp,
  MessageCircle,
  Database,
  RefreshCw,
  BarChart3,
  Brain,
  CheckCircle,
  AlertCircle,
  HardDrive,
} from "lucide-react";
import { useApi } from "../context/ApiContext";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorMessage from "../components/ErrorMessage";

const Dashboard = () => {
  const {
    fetchStatistics,
    fetchCountries,
    fetchDatabaseStats,
    checkMLHealth,
    loading,
    error,
  } = useApi();

  const [statistics, setStatistics] = useState(null);
  const [countries, setCountries] = useState([]);
  const [selectedCountry, setSelectedCountry] = useState("");
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
        checkMLHealth(),
      ]);

      setStatistics(statsData);
      setCountries(countriesData.countries || []);
      setDbStats(dbData);
      setMlHealth(mlData);
      setLastUpdated(new Date());
    } catch (err) {
      // Error handled by ApiContext
    }
  }, [
    selectedCountry,
    fetchStatistics,
    fetchCountries,
    fetchDatabaseStats,
    checkMLHealth,
  ]);

  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  const formatNumber = (num) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + "M";
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + "K";
    }
    return num?.toLocaleString() || "0";
  };

  const StatCard = ({ title, value, icon: Icon, color = "blue", subtitle }) => (
    <div className="stat-card">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className={`text-2xl font-bold text-${color}-600`}>
            {typeof value === "number" ? formatNumber(value) : value || "0"}
          </p>
          {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
        </div>
        <div className={`p-3 bg-${color}-100 rounded-lg`}>
          <Icon className={`w-6 h-6 text-${color}-600`} />
        </div>
      </div>
    </div>
  );

  const StatusCard = ({
    title,
    status,
    icon: Icon,
    children,
    color = "blue",
  }) => (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Icon className={`w-5 h-5 text-${color}-600`} />
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        </div>
        <div className="flex items-center space-x-2">
          <div
            className={`w-2 h-2 rounded-full ${
              status === "healthy" || status === "success"
                ? "bg-green-500"
                : status === "warning"
                ? "bg-yellow-500"
                : "bg-red-500"
            }`}
          />
          <span
            className={`text-sm ${
              status === "healthy" || status === "success"
                ? "text-green-600"
                : status === "warning"
                ? "text-yellow-600"
                : "text-red-600"
            }`}
          >
            {status === "healthy"
              ? "Khỏe mạnh"
              : status === "success"
              ? "Thành công"
              : status === "warning"
              ? "Cảnh báo"
              : "Lỗi"}
          </span>
        </div>
      </div>
      {children}
    </div>
  );

  if (loading && !statistics) {
    return <LoadingSpinner message="Đang tải dữ liệu dashboard..." />;
  }

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header with compact info */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            Dashboard Analytics
          </h1>
          <p className="mt-1 text-sm text-gray-600">
            {countries.length} quốc gia • Cập nhật:{" "}
            {lastUpdated ? lastUpdated.toLocaleTimeString("vi-VN") : "Chưa có"}{" "}
            • Max views:{" "}
            {statistics?.statistics?.max_views
              ? formatNumber(statistics.statistics.max_views)
              : "0"}
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
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
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
            subtitle={
              selectedCountry
                ? `Quốc gia: ${selectedCountry.toUpperCase()}`
                : "Toàn cầu"
            }
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
        <StatusCard
          title="Trạng thái Database"
          status="healthy"
          icon={Database}
          color="blue"
        >
          {dbStats ? (
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Total Collections</span>
                <span className="text-sm font-medium text-gray-900">
                  {dbStats.total_collections}
                </span>
              </div>

              <div className="space-y-2">
                {Object.entries(dbStats.collections || {}).map(
                  ([collection, count]) => (
                    <div
                      key={collection}
                      className="flex justify-between items-center py-1"
                    >
                      <span className="text-xs text-gray-600 capitalize">
                        {collection.replace("_", " ")}
                      </span>
                      <span className="text-xs font-medium text-gray-900">
                        {formatNumber(count)}
                      </span>
                    </div>
                  )
                )}
              </div>
            </div>
          ) : (
            <LoadingSpinner size="small" message="Đang tải..." />
          )}
        </StatusCard>

        {/* ML Models Status */}
        <StatusCard
          title="Trạng thái ML Models"
          status={mlHealth?.is_trained ? "healthy" : "warning"}
          icon={Brain}
          color="purple"
        >
          {mlHealth ? (
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Framework</span>
                <span className="text-sm font-medium text-gray-900">
                  {mlHealth.framework}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Loaded Models</span>
                <span className="text-sm font-medium text-gray-900">
                  {mlHealth.total_models}/2
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Storage</span>
                <span className="text-sm font-medium text-gray-900">
                  {mlHealth.storage}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Spark Session</span>
                <span
                  className={`text-sm font-medium ${
                    mlHealth.spark_session ? "text-green-600" : "text-red-600"
                  }`}
                >
                  {mlHealth.spark_session ? "Active" : "Inactive"}
                </span>
              </div>

              {mlHealth.loaded_models && mlHealth.loaded_models.length > 0 && (
                <div className="mt-4">
                  <p className="text-sm text-gray-600 mb-2">
                    Available Models:
                  </p>
                  <div className="space-y-1">
                    {mlHealth.loaded_models.map((model) => (
                      <div
                        key={model}
                        className="flex items-center justify-between py-1"
                      >
                        <span className="text-xs text-gray-700 capitalize">
                          {model.replace("_", " ")}
                        </span>
                        <div className="flex items-center space-x-1">
                          <CheckCircle className="w-3 h-3 text-green-500" />
                          <span className="text-xs text-green-600">Ready</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <LoadingSpinner size="small" message="Đang kiểm tra..." />
          )}
        </StatusCard>
      </div>

      {/* Model Performance Metrics */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-6">
          <BarChart3 className="w-5 h-5 text-green-600" />
          <h3 className="text-lg font-semibold text-gray-900">
            Hiệu suất Mô hình
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="text-center p-6 bg-gradient-to-br from-green-50 to-green-100 rounded-lg">
            <TrendingUp className="w-8 h-8 text-green-600 mx-auto mb-3" />
            <h4 className="text-lg font-semibold text-gray-900">
              Days Regressor
            </h4>
            <p className="text-sm text-gray-600 mb-2">
              Dự đoán số ngày trending
            </p>
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span>RMSE:</span>
                <span className="font-medium">
                  {mlHealth?.metrics?.regression?.rmse
                    ? mlHealth.metrics.regression.rmse.toFixed(3)
                    : "N/A"}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span>MAE:</span>
                <span className="font-medium">
                  {mlHealth?.metrics?.regression?.mae
                    ? mlHealth.metrics.regression.mae.toFixed(3)
                    : "N/A"}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span>R²:</span>
                <span className="font-medium">
                  {mlHealth?.metrics?.regression?.r2_score
                    ? `${(mlHealth.metrics.regression.r2_score * 100).toFixed(
                        1
                      )}%`
                    : "N/A"}
                </span>
              </div>
            </div>
          </div>

          <div className="text-center p-6 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg">
            <HardDrive className="w-8 h-8 text-purple-600 mx-auto mb-3" />
            <h4 className="text-lg font-semibold text-gray-900">
              Content Clusterer
            </h4>
            <p className="text-sm text-gray-600 mb-2">Phân loại nội dung</p>
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span>Silhouette:</span>
                <span className="font-medium">
                  {mlHealth?.metrics?.clustering?.silhouette_score
                    ? mlHealth.metrics.clustering.silhouette_score.toFixed(3)
                    : "N/A"}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span>Clusters:</span>
                <span className="font-medium">
                  {mlHealth?.metrics?.clustering?.num_clusters || "N/A"}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span>Features:</span>
                <span className="font-medium">
                  {mlHealth?.metrics?.clustering?.features_used
                    ? mlHealth.metrics.clustering.features_used.length
                    : "N/A"}
                </span>
              </div>
            </div>
          </div>
        </div>

        {!mlHealth?.is_trained && (
          <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-5 h-5 text-yellow-600" />
              <span className="text-sm font-medium text-yellow-800">
                Mô hình chưa được huấn luyện. Vui lòng huấn luyện mô hình để xem
                các chỉ số hiệu suất.
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
