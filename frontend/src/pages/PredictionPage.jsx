import React, { useState } from "react";

import { useMlHealth } from "../hooks/useApiQueries";
import {
  usePredictionMutation,
  usePrediction as useCachedPrediction,
} from "../hooks/useApiQueries";
import PredictionForm from "../components/prediction/PredictionForm";
import PredictionResults from "../components/prediction/PredictionResults";
import ModelSelector from "../components/prediction/ModelSelector";
import ErrorMessage from "../components/ErrorMessage";

const PredictionPage = () => {
  const [currentUrl, setCurrentUrl] = useState("");
  const { data: mlHealth, error: mlHealthError } = useMlHealth();
  const predictionMutation = usePredictionMutation();
  const { data: cachedPrediction } = useCachedPrediction(currentUrl);

  // Use cached prediction if available, otherwise use mutation result
  const predictions = cachedPrediction || predictionMutation.data;
  const predictionLoading = predictionMutation.isPending;
  const videoMetadata = predictions?.result?.input_video;

  const handlePredictByUrl = async (url) => {
    setCurrentUrl(url);
    await predictionMutation.mutateAsync(url);
  };

  const hasError = mlHealthError || predictionMutation.error;

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

      {hasError && <ErrorMessage message={hasError.message} />}

      {/* ML Health Status */}
      <ModelSelector mlHealth={mlHealth} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* URL-based Input Form */}
        <PredictionForm
          onPredictByUrl={(url) => handlePredictByUrl(url)}
          loading={predictionLoading}
          mlHealth={mlHealth}
          videoMetadata={videoMetadata}
        />

        {/* Predictions Results */}
        <PredictionResults
          predictions={predictions}
          predictionLoading={predictionLoading}
        />
      </div>
    </div>
  );
};

export default PredictionPage;
