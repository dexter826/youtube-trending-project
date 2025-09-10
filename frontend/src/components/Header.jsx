import React from "react";
import { Play, Activity, Database } from "lucide-react";

const Header = ({ healthStatus }) => {
  const getStatusColor = () => {
    if (!healthStatus) return "bg-gray-500";
    return healthStatus.status === "healthy" ? "bg-green-500" : "bg-red-500";
  };

  const getStatusText = () => {
    if (!healthStatus) return "Đang tải...";
    return healthStatus.status === "healthy" ? "Trực tuyến" : "Ngoại tuyến";
  };

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            <div className="bg-youtube-red p-2 rounded-lg">
              <Play className="h-6 w-6 text-white" fill="currentColor" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Phân tích xu hướng YouTube
              </h1>
              <p className="text-sm text-gray-600">
                Pipeline Big Data cho phân tích xu hướng video
              </p>
            </div>
          </div>

          {/* Status Indicators */}
          <div className="flex items-center space-x-6">
            {/* API Status */}
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-gray-500" />
              <span className="text-sm text-gray-600">API</span>
              <div className={`w-2 h-2 rounded-full ${getStatusColor()}`}></div>
              <span className="text-sm font-medium text-gray-700">
                {getStatusText()}
              </span>
            </div>

            {/* Data Status */}
            {healthStatus?.data && (
              <div className="flex items-center space-x-2">
                <Database className="h-4 w-4 text-gray-500" />
                <span className="text-sm text-gray-600">Quốc gia</span>
                <span className="text-sm font-medium text-gray-700">
                  {healthStatus.data.total_countries}
                </span>
              </div>
            )}

            {/* Tech Stack Badge */}
            <div className="hidden md:flex items-center space-x-2 bg-gray-100 px-3 py-1 rounded-full">
              <span className="text-xs font-medium text-gray-700">
                Spark + MongoDB + FastAPI + React + Machine Learning
              </span>
            </div>
          </div>
        </div>

        {/* Health Status Details */}
        {healthStatus?.data && (
          <div className="mt-4 bg-gray-50 rounded-lg p-3">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-500">Quốc gia:</span>
                <span className="ml-2 font-medium">
                  {healthStatus.data.total_countries}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Ngày:</span>
                <span className="ml-2 font-medium">
                  {healthStatus.data.total_dates}
                </span>
              </div>
              <div className="col-span-2">
                <span className="text-gray-500">Có sẵn:</span>
                <span className="ml-2 font-medium text-green-600">
                  {healthStatus.data.countries_available
                    ?.slice(0, 5)
                    .join(", ")}
                  {healthStatus.data.countries_available?.length > 5 && "..."}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;
