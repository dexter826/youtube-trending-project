import React, { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
} from "recharts";
import {
  Filter,
  Download,
  Eye,
  ThumbsUp,
  MessageCircle,
  TrendingUp,
  Calendar,
  Globe,
  Play,
} from "lucide-react";
import { useApi } from "../context/ApiContext";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorMessage from "../components/ErrorMessage";

const TrendingAnalysis = () => {
  const {
    fetchTrendingVideos,
    fetchCountries,
    fetchCategories,
    fetchWordcloudData,
    loading,
    error,
  } = useApi();

  const [videos, setVideos] = useState([]);
  const [countries, setCountries] = useState([]);
  const [categories, setCategories] = useState([]);
  const [wordcloudData, setWordcloudData] = useState([]);
  const [filters, setFilters] = useState({
    country: "",
    category: "",
    limit: 50,
  });

  const loadData = async () => {
    try {
      const [videosData, countriesData, categoriesData, wordcloudResult] =
        await Promise.all([
          fetchTrendingVideos(filters),
          fetchCountries(),
          fetchCategories(),
          fetchWordcloudData(filters.country || null),
        ]);

      setVideos(videosData.videos || []);
      setCountries(countriesData.countries || []);
      setCategories(categoriesData.categories || []);
      setWordcloudData(wordcloudResult.wordcloud_data || []);
    } catch (err) {
      console.error("Failed to load trending analysis data:", err);
    }
  };

  useEffect(() => {
    loadData();
  }, [filters]);

  const handleFilterChange = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  // Prepare chart data
  const categoryData = categories
    .map((cat) => {
      const count = videos.filter((v) => v.category_id === cat.id).length;
      return {
        name: cat.name,
        count,
        percentage:
          videos.length > 0 ? ((count / videos.length) * 100).toFixed(1) : 0,
      };
    })
    .filter((item) => item.count > 0);

  const viewsData = videos.slice(0, 10).map((video) => ({
    title: video.title?.substring(0, 30) + "..." || "Untitled",
    views: video.views || 0,
    likes: video.likes || 0,
    comments: video.comment_count || 0,
  }));

  const COLORS = [
    "#3B82F6",
    "#EF4444",
    "#10B981",
    "#F59E0B",
    "#8B5CF6",
    "#EC4899",
    "#6B7280",
    "#14B8A6",
  ];

  const formatNumber = (num) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + "M";
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + "K";
    }
    return num?.toLocaleString() || "0";
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            Phân tích Video Trending
          </h1>
          <p className="mt-2 text-gray-600">
            Khám phá xu hướng và thống kê chi tiết về video trending
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="flex items-center mb-4">
          <Filter className="w-5 h-5 text-gray-600 mr-2" />
          <h3 className="text-lg font-semibold text-gray-900">Bộ lọc</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Globe className="w-4 h-4 inline mr-1" />
              Quốc gia
            </label>
            <select
              value={filters.country}
              onChange={(e) => handleFilterChange("country", e.target.value)}
              className="input-field"
            >
              <option value="">Tất cả quốc gia</option>
              {countries.map((country) => (
                <option key={country} value={country}>
                  {country.toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <TrendingUp className="w-4 h-4 inline mr-1" />
              Danh mục
            </label>
            <select
              value={filters.category}
              onChange={(e) => handleFilterChange("category", e.target.value)}
              className="input-field"
            >
              <option value="">Tất cả danh mục</option>
              {categories.map((category) => (
                <option key={category.id} value={category.id}>
                  {category.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Calendar className="w-4 h-4 inline mr-1" />
              Số lượng
            </label>
            <select
              value={filters.limit}
              onChange={(e) =>
                handleFilterChange("limit", parseInt(e.target.value))
              }
              className="input-field"
            >
              <option value={25}>25 video</option>
              <option value={50}>50 video</option>
              <option value={100}>100 video</option>
              <option value={200}>200 video</option>
            </select>
          </div>

          <div className="flex items-end">
            <button
              onClick={loadData}
              disabled={loading}
              className="btn-primary w-full flex items-center justify-center space-x-2"
            >
              <Download className="w-4 h-4" />
              <span>Tải dữ liệu</span>
            </button>
          </div>
        </div>
      </div>

      {error && <ErrorMessage message={error} />}

      {loading && <LoadingSpinner message="Đang tải dữ liệu phân tích..." />}

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="stat-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Tổng Video</p>
              <p className="text-2xl font-bold text-blue-600">
                {videos.length}
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-blue-600" />
          </div>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Tổng lượt xem</p>
              <p className="text-2xl font-bold text-green-600">
                {formatNumber(
                  videos.reduce((sum, v) => sum + (v.views || 0), 0)
                )}
              </p>
            </div>
            <Eye className="w-8 h-8 text-green-600" />
          </div>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">
                Tổng lượt thích
              </p>
              <p className="text-2xl font-bold text-red-600">
                {formatNumber(
                  videos.reduce((sum, v) => sum + (v.likes || 0), 0)
                )}
              </p>
            </div>
            <ThumbsUp className="w-8 h-8 text-red-600" />
          </div>
        </div>

        <div className="stat-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">
                Tổng bình luận
              </p>
              <p className="text-2xl font-bold text-purple-600">
                {formatNumber(
                  videos.reduce((sum, v) => sum + (v.comment_count || 0), 0)
                )}
              </p>
            </div>
            <MessageCircle className="w-8 h-8 text-purple-600" />
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Category Distribution */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Phân bố theo Danh mục
          </h3>
          {categoryData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={categoryData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percentage }) => `${name}: ${percentage}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {categoryData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={COLORS[index % COLORS.length]}
                    />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500">
              Không có dữ liệu để hiển thị
            </div>
          )}
        </div>

        {/* Top Videos by Views */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Top 10 Video theo Lượt xem
          </h3>
          {viewsData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={viewsData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tickFormatter={formatNumber} />
                <YAxis dataKey="title" type="category" width={100} />
                <Tooltip
                  formatter={(value) => [formatNumber(value), "Lượt xem"]}
                />
                <Bar dataKey="views" fill="#3B82F6" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500">
              Không có dữ liệu để hiển thị
            </div>
          )}
        </div>
      </div>

      {/* Video List */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">
            Danh sách Video Trending
          </h3>
          <span className="text-sm text-gray-500">
            Hiển thị {videos.length} video
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Video
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Lượt xem
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Lượt thích
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Bình luận
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Danh mục
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Xem Video
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {videos.slice(0, 20).map((video, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-6 py-4">
                    <div className="text-sm font-medium text-gray-900 max-w-xs truncate">
                      {video.title || "Untitled"}
                    </div>
                    <div className="text-sm text-gray-500">
                      {video.channel_title || "Unknown Channel"}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {formatNumber(video.views)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {formatNumber(video.likes)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {formatNumber(video.comment_count)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                      {video.category_title || "Unknown"}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {video.youtube_link ? (
                      <a
                        href={video.youtube_link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center px-3 py-1 text-sm font-medium text-blue-600 bg-blue-50 hover:bg-blue-100 rounded-md transition-colors"
                      >
                        <Play className="w-4 h-4 mr-1" />
                        Xem
                      </a>
                    ) : (
                      <span className="text-gray-400">N/A</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default TrendingAnalysis;
