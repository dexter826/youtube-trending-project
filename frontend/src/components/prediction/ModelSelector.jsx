import React from "react";

const ModelSelector = ({ mlHealth }) => {
  if (!mlHealth) return null;

  return (
    <div className="card">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div
            className={`w-3 h-3 rounded-full ${
              mlHealth.is_trained ? "bg-green-500" : "bg-red-500"
            }`}
          />
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              Trạng thái ML Models
            </h3>
            <p className="text-sm text-gray-600">
              {mlHealth.is_trained
                ? `${mlHealth.total_models} mô hình đã sẵn sàng`
                : "Mô hình chưa được huấn luyện"}
            </p>
          </div>
        </div>

        {!mlHealth.is_trained && (
          <div className="flex items-center space-x-2 text-yellow-600">
            <span className="text-sm">Cần huấn luyện mô hình</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelSelector;