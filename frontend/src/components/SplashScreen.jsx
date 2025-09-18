import React, { useEffect, useState } from "react";
import Lottie from "lottie-react";
import splashscreenData from "../assets/splashscreen_yt.json";

const SplashScreen = ({ onComplete }) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    // Auto-hide splash screen after animation completes (3.72 seconds based on the JSON)
    const timer = setTimeout(() => {
      setIsVisible(false);
      if (onComplete) {
        onComplete();
      }
    }, 4000); // 4 seconds to ensure animation completes

    return () => clearTimeout(timer);
  }, [onComplete]);

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-white">
      <div className="w-64 h-64 transform scale-90">
        <Lottie
          animationData={splashscreenData}
          loop={false}
          autoplay={true}
          style={{ width: "100%", height: "100%" }}
        />
      </div>
      <div className="transform -translate-y-16 text-center">
        <h1 className="text-3xl font-bold text-red-600 mb-2">
          YouTube Analytics
        </h1>
        <p className="text-lg text-gray-600">Big Data & ML Platform</p>
      </div>
    </div>
  );
};

export default SplashScreen;
