import React, { useState, useEffect, useCallback } from "react";
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
  ScatterChart,
  Scatter,
  ZAxis,
  Legend,
} from "recharts";
import {
  Filter,
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
    fetchDates,
    loading,
    error,
  } = useApi();

  const [videos, setVideos] = useState([]);
  const [countries, setCountries] = useState([]);
  const [categories, setCategories] = useState([]);
  const [dates, setDates] = useState([]);
  const [page, setPage] = useState(1);
  const pageSize = 20;
  const [filters, setFilters] = useState({
    country: "",
    category: "",
    date: "",
    sortBy: "views",
    order: "desc",
  });

  const loadData = useCallback(async () => {
    try {
      const [videosData, countriesData, categoriesData, datesData] = await Promise.all([
        fetchTrendingVideos(filters),
        fetchCountries(),
        fetchCategories(filters.country || null),
        fetchDates(filters.country || null),
      ]);

      setVideos(videosData.videos || []);
      setCountries(countriesData.countries || []);
      setCategories(
        Object.entries(categoriesData.categories || {}).map(([id, name]) => ({
          id: parseInt(id),
          name,
        }))
      );
      setDates(datesData.dates || []);
    } catch (err) {
      // Error handled by ApiContext
    }
  }, [filters, fetchTrendingVideos, fetchCountries, fetchCategories, fetchDates]);

  // Debounce load when filters change
  useEffect(() => {
    const t = setTimeout(() => {
      loadData();
    }, 300);
    return () => clearTimeout(t);
  }, [loadData]);

  // Auto-select latest date when dates loaded and no date selected
  useEffect(() => {
    if (!filters.date && dates && dates.length > 0) {
      setFilters((prev) => ({ ...prev, date: dates[0] }));
    }
  }, [dates, filters.date]);

  const handleFilterChange = (key, value) => {
    setPage(1);
    if (key === 'country') {
      // Reset dependent filters so new lists (dates/categories) are fetched and re-selected
      setFilters((prev) => ({ ...prev, country: value, category: '', date: '' }));
    } else {
      setFilters((prev) => ({ ...prev, [key]: value }));
    }
  };

  // Color palette used across charts
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

  // Prepare chart data
  const categoryData = categories
    .map((cat) => {
      const count = videos.filter((v) => v.category_id === cat.id).length;
      return {
        id: cat.id,
        name: cat.name,
        count,
        percentage:
          videos.length > 0 ? ((count / videos.length) * 100).toFixed(1) : 0,
      };
    })
    .filter((item) => item.count > 0);

  // Top Channels aggregation: count videos per channel and average engagement/views
  const channelMap = videos.reduce((acc, v) => {
    const ch = v.channel_title || "Unknown Channel";
    const views = v.views || 0;
    const likes = v.likes || 0;
    const comments = v.comment_count || 0;
    if (!acc[ch]) acc[ch] = { channel: ch, count: 0, viewsSum: 0, engSum: 0, engDen: 0 };
    acc[ch].count += 1;
    acc[ch].viewsSum += views;
    if (views > 0) {
      acc[ch].engSum += (likes + comments) / views;
      acc[ch].engDen += 1;
    }
    return acc;
  }, {});

  let channelStats = Object.values(channelMap)
    .map((c) => ({
      channel: c.channel,
      count: c.count,
      avgViews: c.count > 0 ? c.viewsSum / c.count : 0,
      avgEng: c.engDen > 0 ? c.engSum / c.engDen : 0,
    }))
    .sort((a, b) => b.count - a.count);

  channelStats = channelStats.slice(0, 12);

  // Fixed height to match prior chart sizing
  const topChannelsHeight = 600;

  // Scatter data: Views vs Engagement (size by comments), limited to top 100 videos by views (as provided order)
  const scatterRaw = videos.slice(0, 100).map((v) => ({
    x: v.views || 0,
    y: (v.views || 0) > 0 ? ((v.likes || 0) + (v.comment_count || 0)) / (v.views || 1) : 0,
    z: Math.max(10, Math.min(300, v.comment_count || 0)),
    title: v.title || "Untitled",
    channel: v.channel_title || "Unknown Channel",
    category_id: v.category_id,
  }));

  // Group scatter points by category to colorize
  const catNameById = (id) => {
    const c = categories.find((k) => k.id === id);
    return c ? c.name : "Other";
  };

  const scatterGroupsMap = scatterRaw.reduce((acc, p) => {
    const name = catNameById(p.category_id);
    if (!acc[name]) acc[name] = [];
    acc[name].push(p);
    return acc;
  }, {});

  const scatterGroups = Object.entries(scatterGroupsMap)
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 7) // limit number of legend entries/colors
    .map(([name, data], idx) => ({ name, data, color: COLORS[idx % COLORS.length] }));

  // Get category name from ID
  const getCategoryName = (categoryId) => {
    const category = categories.find(cat => cat.id === categoryId);
    return category ? category.name : 'Unknown';
  };

  // Generate YouTube link from video ID
  const getYouTubeLink = (video) => {
    if (video.youtube_link) return video.youtube_link;
    if (video.video_id) return `https://www.youtube.com/watch?v=${video.video_id}`;
    if (video.id) return `https://www.youtube.com/watch?v=${video.id}`;
    return null;
  };


  const formatNumber = (num) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + "M";
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + "K";
    }
    return num?.toLocaleString() || "0";
  };

  const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;

  const calcEngagement = (v) => {
    const views = v.views || 0;
    const likes = v.likes || 0;
    const comments = v.comment_count || 0;
    return views > 0 ? (likes + comments) / views : 0;
  };

  const pagedVideos = videos.slice((page - 1) * pageSize, page * pageSize);
  const totalPages = Math.max(1, Math.ceil(videos.length / pageSize));

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
        {/* Summary badges */}
        <div className="mt-4 flex flex-wrap gap-3 text-sm">
          <span className="px-3 py-1 rounded-full bg-gray-100 text-gray-700">Ngày: <span className="font-medium">{filters.date || (dates[0] || 'Mới nhất')}</span></span>
          <span className="px-3 py-1 rounded-full bg-gray-100 text-gray-700">Sắp xếp: <span className="font-medium">{filters.sortBy}</span></span>
          <span className="px-3 py-1 rounded-full bg-gray-100 text-gray-700">Thứ tự: <span className="font-medium">{filters.order}</span></span>
          <span className="px-3 py-1 rounded-full bg-gray-100 text-gray-700">Kết quả: <span className="font-medium">{videos.length}</span></span>
        </div>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="flex items-center mb-4">
          <Filter className="w-5 h-5 text-gray-600 mr-2" />
          <h3 className="text-lg font-semibold text-gray-900">Bộ lọc</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
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
              Ngày
            </label>
            <select
              value={filters.date}
              onChange={(e) => handleFilterChange("date", e.target.value)}
              className="input-field"
            >
              <option value="">Mới nhất</option>
              {dates.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Sắp xếp theo
            </label>
            <select
              value={filters.sortBy}
              onChange={(e) => handleFilterChange("sortBy", e.target.value)}
              className="input-field"
            >
              <option value="views">Lượt xem</option>
              <option value="likes">Lượt thích</option>
              <option value="comments">Bình luận</option>
              <option value="engagement">Tương tác</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Thứ tự
            </label>
            <select
              value={filters.order}
              onChange={(e) => handleFilterChange("order", e.target.value)}
              className="input-field"
            >
              <option value="desc">Giảm dần</option>
              <option value="asc">Tăng dần</option>
            </select>
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
              <p className="text-sm font-medium text-gray-600">Tỷ lệ tương tác TB</p>
              <p className="text-2xl font-bold text-blue-600">
                {formatPercent(
                  (() => {
                    const valid = videos.filter(v => (v.views || 0) > 0);
                    if (valid.length === 0) return 0;
                    const sum = valid.reduce((acc, v) => acc + ((v.likes || 0) + (v.comment_count || 0)) / (v.views || 1), 0);
                    return sum / valid.length;
                  })()
                )}
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
        {/* Category Distribution (Donut + interactive legend) */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Phân bố theo Danh mục
          </h3>
          {categoryData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={categoryData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ percentage }) => `${percentage}%`}
                    innerRadius={55}
                    outerRadius={90}
                    fill="#8884d8"
                    dataKey="count"
                  >
                    {categoryData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-2">
                {categoryData.map((c, idx) => (
                  <button
                    key={c.id}
                    onClick={() => handleFilterChange('category', String(c.id))}
                    className={`flex items-center justify-between px-3 py-2 rounded-md border transition ${String(filters.category) === String(c.id) ? 'bg-blue-50 border-blue-200' : 'bg-white border-gray-200 hover:bg-gray-50'}`}
                  >
                    <div className="flex items-center">
                      <span className="w-3 h-3 rounded-sm mr-2" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
                      <span className="text-sm text-gray-800">{c.name}</span>
                    </div>
                    <span className="text-sm text-gray-500">{c.count} • {c.percentage}%</span>
                  </button>
                ))}
                {filters.category && (
                  <button
                    onClick={() => handleFilterChange('category', '')}
                    className="px-3 py-2 rounded-md border border-gray-200 hover:bg-gray-50 text-sm text-gray-700"
                  >Bỏ lọc danh mục</button>
                )}
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500">
              Không có dữ liệu để hiển thị
            </div>
          )}
        </div>

        {/* Top Channels by Trending Count */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-1">
            Kênh xuất hiện nhiều trong Trending
          </h3>
          <p className="text-sm text-gray-500 mb-4">Hiển thị số video trending theo kênh. Tooltip có Engagement TB và Views TB.</p>
          {channelStats.length > 0 ? (
            <ResponsiveContainer width="100%" height={topChannelsHeight}>
              <BarChart data={channelStats} layout="horizontal" margin={{ left: 8, right: 16, top: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tickFormatter={formatNumber} />
                <YAxis dataKey="channel" type="category" width={180} interval={0} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const d = payload[0].payload;
                      return (
                        <div className="p-3 bg-white rounded-md shadow text-sm">
                          <div className="font-semibold mb-1">{d.channel}</div>
                          <div>Số video: <span className="font-medium">{d.count}</span></div>
                          <div>Views TB: <span className="font-medium">{formatNumber(Math.round(d.avgViews))}</span></div>
                          <div>Engagement TB: <span className="font-medium">{formatPercent(d.avgEng)}</span></div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="count" fill="#3B82F6" radius={[4, 4, 4, 4]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500">Không có dữ liệu</div>
          )}
        </div>
      </div>

      {/* Scatter: Views vs Engagement (size by Comments) */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-1">Views vs Engagement (Scatter)</h3>
        <p className="text-sm text-gray-500 mb-4">Màu theo danh mục, kích thước theo số bình luận. Tối đa 100 video.</p>
        {scatterGroups.length > 0 ? (
          <ResponsiveContainer width="100%" height={380}>
            <ScatterChart margin={{ left: 24, right: 24, top: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="Views" tickFormatter={formatNumber} />
              <YAxis type="number" dataKey="y" name="Engagement" tickFormatter={formatPercent} />
              <ZAxis type="number" dataKey="z" range={[40, 200]} name="Comments" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }}
                formatter={(value, name) => {
                  if (name === 'x') return [formatNumber(value), 'Views'];
                  if (name === 'y') return [formatPercent(value), 'Engagement'];
                  if (name === 'z') return [formatNumber(value), 'Comments'];
                  return [value, name];
                }}
                labelFormatter={(label, payload) => (payload && payload[0] && payload[0].payload ? payload[0].payload.title : '')}
              />
              <Legend />
              {scatterGroups.map((g, idx) => (
                <Scatter key={g.name} name={g.name} data={g.data} fill={g.color} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex items-center justify-center h-64 text-gray-500">Không có dữ liệu</div>
        )}
      </div>

      {/* Video List */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">
            Danh sách Video Trending
          </h3>
          <span className="text-sm text-gray-500">
            Hiển thị {pagedVideos.length} / {videos.length} video
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
                  Tương tác
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
              {pagedVideos.map((video, index) => (
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
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {formatPercent(calcEngagement(video))}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                      {getCategoryName(video.category_id) || "Unknown"}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {getYouTubeLink(video) ? (
                      <a
                        href={getYouTubeLink(video)}
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

        {/* Pagination */}
        <div className="mt-4 flex items-center justify-between">
          <div className="text-sm text-gray-600">Trang {page} / {totalPages}</div>
          <div className="space-x-2">
            <button
              className="px-3 py-1 rounded-md border border-gray-300 text-sm disabled:opacity-50"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >Trước</button>
            <button
              className="px-3 py-1 rounded-md border border-gray-300 text-sm disabled:opacity-50"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            >Sau</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrendingAnalysis;
