import React, { useState } from "react";

import { usePrediction } from "../hooks/usePrediction";
import { useModelData } from "../hooks/useModelData";
import PredictionForm from "../components/prediction/PredictionForm";
import PredictionResults from "../components/prediction/PredictionResults";
import PredictionHistory from "../components/prediction/PredictionHistory";
import ModelSelector from "../components/prediction/ModelSelector";
import ErrorMessage from "../components/ErrorMessage";
import { useApi } from "../context/ApiContext";

const PredictionPage = () => {
  const { loading, error } = useApi();
  const { mlHealth, categories } = useModelData();
  const {
    predictions,
    predictionLoading,
    handlePredict,
    handlePredictAll,
    clearPredictions,
  } = usePrediction();

  const [videoData, setVideoData] = useState({
    title: "",
    views: 0,
    likes: 0,
    dislikes: 0,
    comment_count: 0,
    category_id: 0,
    tags: "",
    publish_hour: 12,
    video_age_proxy: 2,
  });

  const [samplePredictions, setSamplePredictions] = useState({
    tech: {
      data: {
        title: "iPhone 15 Pro Max Review - Is It Worth The Hype?",
        views: 50000,
        likes: 2500,
        dislikes: 150,
        comment_count: 800,
        category_id: 28, // Science & Technology
        tags: "iPhone|iPhone 15|review|tech|Apple|smartphone",
        publish_hour: 14,
        video_age_proxy: 2,
      },
      predictions: {
        trending: {
          prediction: {
            trending_probability: 0.014,
            prediction: 0,
            confidence: "high",
            method: "spark_mllib",
          },
        },
        views: {
          prediction: {
            predicted_views: 51633,
            confidence: "high",
            method: "spark_mllib",
          },
        },
        cluster: {
          prediction: {
            cluster: 0,
            cluster_type: "Nội dung Tác động Cao",
            confidence: "high",
            method: "spark_mllib",
          },
        },
      },
    },
    entertainment: {
      data: {
        title: "Top 10 Funniest Cat Videos of 2024!",
        views: 100000,
        likes: 8000,
        dislikes: 200,
        comment_count: 1200,
        category_id: 24, // Entertainment
        tags: "cats|funny|videos|pets|comedy|animals",
        publish_hour: 18,
        video_age_proxy: 1,
      },
      predictions: {
        trending: {
          prediction: {
            trending_probability: 0.025,
            prediction: 0,
            confidence: "high",
            method: "spark_mllib",
          },
        },
        views: {
          prediction: {
            predicted_views: 100601,
            confidence: "high",
            method: "spark_mllib",
          },
        },
        cluster: {
          prediction: {
            cluster: 1,
            cluster_type: "Nội dung Đại chúng",
            confidence: "high",
            method: "spark_mllib",
          },
        },
      },
    },
    trending: {
      data: {
        title: "Breaking News: Major Event Shocks the World!",
        views: 1000000,
        likes: 150000,
        dislikes: 5000,
        comment_count: 25000,
        category_id: 25, // News & Politics
        tags: "news|breaking|viral|trending|world|event",
        publish_hour: 8,
        video_age_proxy: 1,
      },
      predictions: {
        trending: {
          prediction: {
            trending_probability: 0.85,
            prediction: 1,
            confidence: "high",
            method: "spark_mllib",
          },
        },
        views: {
          prediction: {
            predicted_views: 1200000,
            confidence: "high",
            method: "spark_mllib",
          },
        },
        cluster: {
          prediction: {
            cluster: 2,
            cluster_type: "Nội dung Tiềm năng",
            confidence: "high",
            method: "spark_mllib",
          },
        },
      },
    },
  });

  const [selectedSample, setSelectedSample] = useState(null);

  const handleInputChange = (field, value) => {
    setVideoData((prev) => ({
      ...prev,
      [field]:
        field.includes("count") ||
        field === "views" ||
        field === "likes" ||
        field === "dislikes" ||
        field === "category_id" ||
        field === "publish_hour" ||
        field === "video_age_proxy"
          ? parseInt(value) || 0
          : value,
    }));
  };

  const handlePredictWrapper = (type) => {
    handlePredict(type, videoData);
    setSelectedSample(null); // Clear selected sample when doing real prediction
  };

  const handlePredictAllWrapper = () => {
    handlePredictAll(videoData);
    setSelectedSample(null);
  };

  const loadSample = (sampleKey) => {
    const sample = samplePredictions[sampleKey];
    if (sample) {
      setVideoData(sample.data);
      setSelectedSample(sampleKey);
      // Reset predictions to show sample predictions
      clearPredictions();
      // Set sample predictions after clearing
      setTimeout(() => {
        // This would need to be handled differently in a real implementation
        // For now, we'll just set the video data
      }, 0);
    }
  };

  const isFormValid = () => {
    return (
      videoData.title.trim() !== "" &&
      videoData.category_id !== 0 &&
      videoData.views >= 0 &&
      videoData.likes >= 0 &&
      videoData.dislikes >= 0 &&
      videoData.comment_count >= 0
    );
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            Dự đoán Video Trending
          </h1>
          <p className="mt-2 text-gray-600">
            Sử dụng Machine Learning để dự đoán hiệu suất video
          </p>
        </div>
      </div>

      {error && <ErrorMessage message={error} />}

      {/* ML Health Status */}
      <ModelSelector mlHealth={mlHealth} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <PredictionForm
          videoData={videoData}
          onInputChange={handleInputChange}
          onPredict={handlePredictWrapper}
          onPredictAll={handlePredictAllWrapper}
          loading={loading}
          mlHealth={mlHealth}
          categories={categories}
          isFormValid={isFormValid}
        />

        {/* Predictions Results */}
        <PredictionResults
          predictions={predictions}
          predictionLoading={predictionLoading}
        />
      </div>

      {/* Sample Predictions */}
      <PredictionHistory
        samplePredictions={samplePredictions}
        selectedSample={selectedSample}
        onLoadSample={loadSample}
      />
    </div>
  );
};

export default PredictionPage;
