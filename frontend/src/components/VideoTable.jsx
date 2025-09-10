import React, { useState } from 'react';
import { ExternalLink, Eye, ThumbsUp, ThumbsDown, MessageCircle, Hash } from 'lucide-react';

const VideoTable = ({ videos }) => {
  const [sortBy, setSortBy] = useState('views');
  const [sortOrder, setSortOrder] = useState('desc');

  if (!videos || videos.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        Không có dữ liệu video
      </div>
    );
  }

  const handleSort = (column) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortOrder('desc');
    }
  };

  const sortedVideos = [...videos].sort((a, b) => {
    let aValue = a[sortBy];
    let bValue = b[sortBy];

    if (typeof aValue === 'string') {
      aValue = aValue.toLowerCase();
      bValue = bValue.toLowerCase();
    }

    if (sortOrder === 'asc') {
      return aValue > bValue ? 1 : -1;
    } else {
      return aValue < bValue ? 1 : -1;
    }
  });

  const formatNumber = (num) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num?.toLocaleString() || '0';
  };

  const truncateText = (text, maxLength = 60) => {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  };

  const SortButton = ({ column, children }) => (
    <button
      onClick={() => handleSort(column)}
      className="flex items-center space-x-1 text-left font-medium text-gray-700 hover:text-gray-900 transition-colors"
    >
      <span>{children}</span>
      {sortBy === column && (
        <span className="text-xs">
          {sortOrder === 'asc' ? '↑' : '↓'}
        </span>
      )}
    </button>
  );

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Thứ hạng
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              <SortButton column="title">Video</SortButton>
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              <SortButton column="channel_title">Kênh</SortButton>
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              <SortButton column="views">
                <Eye className="inline h-3 w-3 mr-1" />
                Lượt xem
              </SortButton>
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              <SortButton column="likes">
                <ThumbsUp className="inline h-3 w-3 mr-1" />
                Thích
              </SortButton>
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              <SortButton column="comment_count">
                <MessageCircle className="inline h-3 w-3 mr-1" />
                Bình luận
              </SortButton>
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              <Hash className="inline h-3 w-3 mr-1" />
              Danh mục
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Hành động
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {sortedVideos.map((video, index) => (
            <tr key={video.video_id || index} className="hover:bg-gray-50 transition-colors">
              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                #{index + 1}
              </td>
              <td className="px-6 py-4">
                <div className="max-w-xs">
                  <p className="text-sm font-medium text-gray-900" title={video.title}>
                    {truncateText(video.title)}
                  </p>
                  {video.tags && (
                    <p className="text-xs text-gray-500 mt-1" title={video.tags}>
                      {truncateText(video.tags, 40)}
                    </p>
                  )}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <p className="text-sm text-gray-900" title={video.channel_title}>
                  {truncateText(video.channel_title, 20)}
                </p>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm">
                  <p className="font-medium text-gray-900">
                    {formatNumber(video.views)}
                  </p>
                  <p className="text-xs text-gray-500">
                    {video.views?.toLocaleString('vi-VN')}
                  </p>
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="flex items-center space-x-2">
                  <div className="text-sm">
                    <p className="font-medium text-green-600">
                      {formatNumber(video.likes)}
                    </p>
                    {video.dislikes > 0 && (
                      <p className="text-xs text-red-500 flex items-center">
                        <ThumbsDown className="h-3 w-3 mr-1" />
                        {formatNumber(video.dislikes)}
                      </p>
                    )}
                  </div>
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <p className="text-sm font-medium text-gray-900">
                  {formatNumber(video.comment_count)}
                </p>
                <p className="text-xs text-gray-500">
                  {video.comment_count?.toLocaleString('vi-VN')}
                </p>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                  {video.category_id}
                </span>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm">
                {video.video_id && (
                  <a
                    href={`https://www.youtube.com/watch?v=${video.video_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center text-youtube-red hover:text-red-700 transition-colors"
                  >
                    <ExternalLink className="h-4 w-4 mr-1" />
                    Xem
                  </a>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      
      {/* Table Footer */}
      <div className="bg-gray-50 px-6 py-3">
        <p className="text-xs text-gray-500">
          Hiển thị {sortedVideos.length} video thịnh hành. Nhấp vào tiêu đề cột để sắp xếp.
        </p>
      </div>
    </div>
  );
};

export default VideoTable;
