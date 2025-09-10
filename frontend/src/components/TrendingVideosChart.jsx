import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const TrendingVideosChart = ({ videos }) => {
  if (!videos || videos.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Không có dữ liệu video
      </div>
    );
  }

  // Prepare data for the chart
  const chartData = {
    labels: videos.slice(0, 10).map((video, index) => `#${index + 1}`),
    datasets: [
      {
        label: 'Lượt xem',
        data: videos.slice(0, 10).map(video => video.views),
        backgroundColor: 'rgba(255, 0, 0, 0.7)',
        borderColor: 'rgba(255, 0, 0, 1)',
        borderWidth: 1,
      },
      {
        label: 'Lượt thích',
        data: videos.slice(0, 10).map(video => video.likes),
        backgroundColor: 'rgba(34, 197, 94, 0.7)',
        borderColor: 'rgba(34, 197, 94, 1)',
        borderWidth: 1,
      }
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: false,
      },
      tooltip: {
        callbacks: {
          title: (context) => {
            const index = context[0].dataIndex;
            const video = videos[index];
            return video.title.length > 50 
              ? video.title.substring(0, 50) + '...'
              : video.title;
          },
          label: (context) => {
            const value = context.parsed.y;
            const label = context.dataset.label;
            return `${label}: ${value.toLocaleString('vi-VN')}`;
          },
          afterLabel: (context) => {
            const index = context.dataIndex;
            const video = videos[index];
            return [
              `Kênh: ${video.channel_title}`,
              `Bình luận: ${video.comment_count.toLocaleString('vi-VN')}`
            ];
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: function(value) {
            if (value >= 1000000) {
              return (value / 1000000).toFixed(1) + 'M';
            } else if (value >= 1000) {
              return (value / 1000).toFixed(1) + 'K';
            }
            return value;
          }
        }
      },
      x: {
        ticks: {
          callback: function(value, index) {
            const video = videos[index];
            if (video) {
              return video.title.length > 15 
                ? video.title.substring(0, 15) + '...'
                : video.title;
            }
            return `#${index + 1}`;
          }
        }
      }
    },
    onClick: (event, elements) => {
      if (elements.length > 0) {
        const index = elements[0].index;
        const video = videos[index];
        if (video.video_id) {
          window.open(`https://www.youtube.com/watch?v=${video.video_id}`, '_blank');
        }
      }
    }
  };

  return (
    <div className="chart-container">
      <Bar data={chartData} options={options} />
      <p className="text-xs text-gray-500 mt-2 text-center">
        Nhấp vào thanh để mở video trên YouTube
      </p>
    </div>
  );
};

export default TrendingVideosChart;
