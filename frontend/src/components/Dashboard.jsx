import React from "react";
import TrendingVideosChart from "./TrendingVideosChart";
import WordCloudComponent from "./WordCloudComponent";
import StatisticsPanel from "./StatisticsPanel";
import VideoTable from "./VideoTable";

const Dashboard = ({
  trendingData,
  wordcloudData,
  selectedCountry,
  selectedDate,
}) => {
  if (!trendingData && !wordcloudData) {
    return null;
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Kết quả phân tích
        </h2>
        <p className="text-gray-600">
          Dữ liệu YouTube thịnh hành cho{" "}
          <span className="font-medium">{selectedCountry}</span> vào{" "}
          <span className="font-medium">
            {new Date(selectedDate).toLocaleDateString("vi-VN", {
              weekday: "long",
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
          </span>
        </p>
      </div>

      {/* Statistics Panel */}
      {trendingData?.statistics && (
        <StatisticsPanel statistics={trendingData.statistics} />
      )}

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Trending Videos Chart */}
        {trendingData?.top_videos && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              📊 Video thịnh hành hàng đầu (Lượt xem)
            </h3>
            <TrendingVideosChart videos={trendingData.top_videos} />
          </div>
        )}

        {/* Word Cloud */}
        {wordcloudData?.words && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              ☁️ Đám mây từ khóa tiêu đề
            </h3>
            <WordCloudComponent words={wordcloudData.words} />
          </div>
        )}
      </div>

      {/* Video Table */}
      {trendingData?.top_videos && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            📋 Thông tin chi tiết video
          </h3>
          <VideoTable videos={trendingData.top_videos} />
        </div>
      )}

      {/* Processing Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
          {trendingData?.processed_at && (
            <div>
              <span className="font-medium">Đã xử lý:</span>
              <span className="ml-2">
                {new Date(trendingData.processed_at).toLocaleString("vi-VN")}
              </span>
            </div>
          )}
          {wordcloudData?.processed_at && (
            <div>
              <span className="font-medium">Đám mây từ khóa được tạo:</span>
              <span className="ml-2">
                {new Date(wordcloudData.processed_at).toLocaleString("vi-VN")}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
