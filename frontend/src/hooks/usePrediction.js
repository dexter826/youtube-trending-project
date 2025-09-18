import { useState } from "react";
import { useApi } from "../context/ApiContext";

export const usePrediction = () => {
  const { predictDays, predictCluster, predictByUrl } = useApi();

  const [predictions, setPredictions] = useState({
    days: null,
    cluster: null,
  });

  const [videoMetadata, setVideoMetadata] = useState(null);

  const [predictionLoading, setPredictionLoading] = useState({
    days: false,
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
        case "days":
          result = await predictDays(videoData);
          setPredictions((prev) => ({ ...prev, days: result }));
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

  // New: Predict by YouTube URL (returns both days + cluster)
  const handlePredictByUrl = async (url) => {
    setPredictionLoading((prev) => ({ ...prev, all: true, days: true, cluster: true }));
    setPredictions({ days: null, cluster: null });
    setVideoMetadata(null);
    try {
      const data = await predictByUrl(url);
      const days = data?.result?.prediction?.days || null;
      const cluster = data?.result?.prediction?.cluster || null;
      const metadata = data?.result?.input_video || null;
      setPredictions({ days, cluster });
      setVideoMetadata(metadata);
    } catch (e) {
      // handled by ApiContext
    } finally {
      setPredictionLoading((prev) => ({ ...prev, all: false, days: false, cluster: false }));
    }
  };

  const handlePredictAll = async (videoData) => {
    setPredictionLoading((prev) => ({
      ...prev,
      all: true,
      days: true,
      cluster: true,
    }));
    // Clear all predictions
    setPredictions({ days: null, cluster: null });

    try {
      const [daysResult, clusterResult] = await Promise.all([
        predictDays(videoData),
        predictCluster(videoData),
      ]);

      setPredictions({ days: daysResult, cluster: clusterResult });
    } catch (err) {
      // Error handled by ApiContext
    } finally {
      setPredictionLoading((prev) => ({
        ...prev,
        all: false,
        days: false,
        cluster: false,
      }));
    }
  };

  const clearPredictions = () => {
    setPredictions({ days: null, cluster: null });
  };

  return {
    predictions,
    predictionLoading,
    videoMetadata,
    handlePredict,
    handlePredictAll,
    handlePredictByUrl,
    clearPredictions,
  };
};