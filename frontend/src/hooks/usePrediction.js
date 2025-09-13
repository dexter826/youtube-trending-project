import { useState } from "react";
import { useApi } from "../context/ApiContext";

export const usePrediction = () => {
  const {
    predictTrending,
    predictViews,
    predictCluster,
  } = useApi();

  const [predictions, setPredictions] = useState({
    trending: null,
    views: null,
    cluster: null,
  });

  const [predictionLoading, setPredictionLoading] = useState({
    trending: false,
    views: false,
    cluster: false,
    all: false,
  });

  const handlePredict = async (type, videoData) => {
    setPredictionLoading((prev) => ({ ...prev, [type]: true }));
    // Clear previous predictions to avoid showing stale data
    setPredictions((prev) => ({ ...prev, [type]: null }));

    try {
      let result;
      switch (type) {
        case "trending":
          result = await predictTrending(videoData);
          setPredictions((prev) => ({ ...prev, trending: result }));
          break;
        case "views":
          result = await predictViews(videoData);
          setPredictions((prev) => ({ ...prev, views: result }));
          break;
        case "cluster":
          result = await predictCluster(videoData);
          setPredictions((prev) => ({ ...prev, cluster: result }));
          break;
        default:
          break;
      }
    } catch (err) {
      // Error handled by ApiContext
    } finally {
      setPredictionLoading((prev) => ({ ...prev, [type]: false }));
    }
  };

  const handlePredictAll = async (videoData) => {
    setPredictionLoading((prev) => ({
      ...prev,
      all: true,
      trending: true,
      views: true,
      cluster: true,
    }));
    // Clear all predictions
    setPredictions({ trending: null, views: null, cluster: null });

    try {
      const [trendingResult, viewsResult, clusterResult] = await Promise.all([
        predictTrending(videoData),
        predictViews(videoData),
        predictCluster(videoData),
      ]);

      setPredictions({
        trending: trendingResult,
        views: viewsResult,
        cluster: clusterResult,
      });
    } catch (err) {
      // Error handled by ApiContext
    } finally {
      setPredictionLoading((prev) => ({
        ...prev,
        all: false,
        trending: false,
        views: false,
        cluster: false,
      }));
    }
  };

  const clearPredictions = () => {
    setPredictions({ trending: null, views: null, cluster: null });
  };

  return {
    predictions,
    predictionLoading,
    handlePredict,
    handlePredictAll,
    clearPredictions,
  };
};