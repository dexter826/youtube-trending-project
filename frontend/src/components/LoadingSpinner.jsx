import React from "react";
import { Loader2 } from "lucide-react";

const LoadingSpinner = ({ message = "Đang tải...", size = "default" }) => {
  const sizeClasses = {
    small: "w-4 h-4",
    default: "w-8 h-8",
    large: "w-12 h-12",
  };

  return (
    <div className="flex flex-col items-center justify-center py-12">
      <Loader2
        className={`${sizeClasses[size]} text-red-600 animate-spin mb-4`}
      />
      <p className="text-gray-600 text-center">{message}</p>
    </div>
  );
};

export default LoadingSpinner;
