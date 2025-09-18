import React from "react";

import { usePrediction } from "../hooks/usePrediction";
import { useModelData } from "../hooks/useModelData";
import PredictionForm from "../components/prediction/PredictionForm";
import PredictionResults from "../components/prediction/PredictionResults";
import ModelSelector from "../components/prediction/ModelSelector";
import ErrorMessage from "../components/ErrorMessage";
import { useApi } from "../context/ApiContext";

const PredictionPage = () => {
  const { loading, error } = useApi();
  const { mlHealth } = useModelData();
  const {
    predictions,
    predictionLoading,
    handlePredictByUrl,
  } = usePrediction();


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
        {/* URL-based Input Form */}
        <PredictionForm
          onPredictByUrl={(url, apiKey) => handlePredictByUrl(url, apiKey)}
          loading={loading}
          mlHealth={mlHealth}
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
