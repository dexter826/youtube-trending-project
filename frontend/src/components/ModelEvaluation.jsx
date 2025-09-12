import React, { useState, useEffect } from "react";
import LoadingSpinner from "./LoadingSpinner";
import ErrorMessage from "./ErrorMessage";
import apiService from "../services/apiService";

const ModelEvaluation = () => {
  const [healthData, setHealthData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchHealthData();
  }, []);

  const fetchHealthData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getMLHealth();
      setHealthData(data);
    } catch (err) {
      setError(err.message || "Failed to fetch ML health data");
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error} onRetry={fetchHealthData} />;
  if (!healthData)
    return <ErrorMessage message="No ML health data available" />;

  const getStatusColor = (isLoaded) => {
    return isLoaded ? "text-green-600" : "text-red-600";
  };

  const getStatusBg = (isLoaded) => {
    return isLoaded
      ? "bg-green-50 border-green-200"
      : "bg-red-50 border-red-200";
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 space-y-6">
      <div className="border-b pb-4">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          ML Service Status
        </h2>
        <div className="flex flex-wrap gap-4 text-sm text-gray-600">
          <span>
            Framework: <strong>{healthData.framework}</strong>
          </span>
          <span>
            Storage: <strong>{healthData.storage}</strong>
          </span>
          <span>
            Training Status:{" "}
            <strong className={getStatusColor(healthData.is_trained)}>
              {healthData.is_trained ? "Trained" : "Not Trained"}
            </strong>
          </span>
        </div>
      </div>

      {/* Models Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div
          className={`p-4 rounded-lg border-2 ${getStatusBg(
            healthData.loaded_models.includes("trending_classifier")
          )}`}
        >
          <h3 className="text-lg font-semibold mb-2">Trending Classifier</h3>
          <div
            className={`text-2xl font-bold ${getStatusColor(
              healthData.loaded_models.includes("trending_classifier")
            )}`}
          >
            {healthData.loaded_models.includes("trending_classifier")
              ? "Loaded"
              : "Not Loaded"}
          </div>
          <p className="text-sm text-gray-600 mt-1">
            Predicts if videos will trend
          </p>
        </div>

        <div
          className={`p-4 rounded-lg border-2 ${getStatusBg(
            healthData.loaded_models.includes("views_regressor")
          )}`}
        >
          <h3 className="text-lg font-semibold mb-2">Views Regressor</h3>
          <div
            className={`text-2xl font-bold ${getStatusColor(
              healthData.loaded_models.includes("views_regressor")
            )}`}
          >
            {healthData.loaded_models.includes("views_regressor")
              ? "Loaded"
              : "Not Loaded"}
          </div>
          <p className="text-sm text-gray-600 mt-1">
            Predicts video view counts
          </p>
        </div>

        <div
          className={`p-4 rounded-lg border-2 ${getStatusBg(
            healthData.loaded_models.includes("content_clusterer")
          )}`}
        >
          <h3 className="text-lg font-semibold mb-2">Content Clusterer</h3>
          <div
            className={`text-2xl font-bold ${getStatusColor(
              healthData.loaded_models.includes("content_clusterer")
            )}`}
          >
            {healthData.loaded_models.includes("content_clusterer")
              ? "Loaded"
              : "Not Loaded"}
          </div>
          <p className="text-sm text-gray-600 mt-1">Groups similar content</p>
        </div>
      </div>

      {/* Model Details */}
      {healthData.model_details && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Model Details
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(healthData.model_details).map(
              ([modelName, details]) => (
                <div key={modelName} className="bg-white p-3 rounded border">
                  <h4 className="font-semibold text-gray-700 mb-2 capitalize">
                    {modelName.replace(/_/g, " ")}
                  </h4>
                  <div className="space-y-1 text-sm">
                    {Object.entries(details).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-gray-600 capitalize">
                          {key.replace(/_/g, " ")}:
                        </span>
                        <span className="font-medium text-gray-800">
                          {typeof value === "object"
                            ? JSON.stringify(value)
                            : String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )
            )}
          </div>
        </div>
      )}

      {/* System Info */}
      <div className="bg-blue-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-blue-800 mb-3">
          System Information
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-blue-700 font-medium">
              Total Models Loaded:
            </span>
            <span className="ml-2 font-bold text-blue-900">
              {healthData.loaded_models.length}
            </span>
          </div>
          <div>
            <span className="text-blue-700 font-medium">Model Type:</span>
            <span className="ml-2 font-bold text-blue-900">
              {healthData.model_type}
            </span>
          </div>
          <div>
            <span className="text-blue-700 font-medium">Available Models:</span>
            <span className="ml-2 font-bold text-blue-900">
              {healthData.loaded_models.join(", ")}
            </span>
          </div>
          <div>
            <span className="text-blue-700 font-medium">
              Training Required:
            </span>
            <span
              className={`ml-2 font-bold ${
                healthData.is_trained ? "text-green-600" : "text-red-600"
              }`}
            >
              {healthData.is_trained ? "No" : "Yes"}
            </span>
          </div>
        </div>
      </div>

      {/* Refresh Button */}
      <div className="flex justify-center pt-4">
        <button
          onClick={fetchHealthData}
          disabled={loading}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Refreshing..." : "Refresh Status"}
        </button>
      </div>
    </div>
  );
};

export default ModelEvaluation;
