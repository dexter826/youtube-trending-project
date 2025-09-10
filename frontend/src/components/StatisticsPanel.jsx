import React from 'react';
import { Eye, ThumbsUp, MessageCircle, Video, TrendingUp, BarChart3 } from 'lucide-react';

const StatisticsPanel = ({ statistics }) => {
  if (!statistics) {
    return null;
  }

  const formatNumber = (num) => {
    if (num >= 1000000000) {
      return (num / 1000000000).toFixed(1) + 'B';
    } else if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num?.toLocaleString() || '0';
  };

  const stats = [
    {
      label: 'Tổng số video',
      value: statistics.total_videos,
      icon: Video,
      color: 'bg-blue-500',
      format: 'number'
    },
    {
      label: 'Tổng lượt xem',
      value: statistics.total_views,
      icon: Eye,
      color: 'bg-green-500',
      format: 'large'
    },
    {
      label: 'Lượt xem trung bình',
      value: statistics.average_views,
      icon: TrendingUp,
      color: 'bg-purple-500',
      format: 'large'
    },
    {
      label: 'Lượt xem tối đa',
      value: statistics.max_views,
      icon: BarChart3,
      color: 'bg-red-500',
      format: 'large'
    },
    {
      label: 'Tổng lượt thích',
      value: statistics.total_likes,
      icon: ThumbsUp,
      color: 'bg-pink-500',
      format: 'large'
    },
    {
      label: 'Tổng bình luận',
      value: statistics.total_comments,
      icon: MessageCircle,
      color: 'bg-yellow-500',
      format: 'large'
    }
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      {stats.map((stat, index) => {
        const Icon = stat.icon;
        const formattedValue = stat.format === 'large' 
          ? formatNumber(stat.value)
          : stat.value?.toLocaleString() || '0';

        return (
          <div
            key={index}
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 hover:shadow-md transition-shadow"
          >
            <div className="flex items-center">
              <div className={`${stat.color} p-2 rounded-lg mr-3`}>
                <Icon className="h-4 w-4 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">
                  {stat.label}
                </p>
                <p className="text-lg font-bold text-gray-700">
                  {formattedValue}
                </p>
              </div>
            </div>
            {stat.format === 'large' && stat.value && (
              <p className="text-xs text-gray-500 mt-1">
                Chính xác: {stat.value.toLocaleString('vi-VN')}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default StatisticsPanel;
