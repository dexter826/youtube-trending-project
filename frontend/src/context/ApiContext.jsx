import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
} from "react";
import axios from "axios";

const ApiContext = createContext();

export const useApi = () => {
  const context = useContext(ApiContext);
  if (!context) {
    throw new Error("useApi must be used within an ApiProvider");
  }
  return context;
};

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Configure axios defaults
axios.defaults.baseURL = API_BASE_URL;
axios.defaults.timeout = 30000;

export const ApiProvider = ({ children }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);

  // API Health Check
  const checkApiHealth = useCallback(async () => {
    try {
      const response = await axios.get("/health");
      setApiHealth(response.data);
      return response.data;
    } catch (err) {
      setApiHealth({ status: "unhealthy", error: err.message });
      return null;
    }
  }, []);

  // Generic API call wrapper
  const apiCall = useCallback(
    async (method, endpoint, data = null, config = {}) => {
      setLoading(true);
      setError(null);

      try {
        const response = await axios({
          method,
          url: endpoint,
          data,
          ...config,
        });

        setLoading(false);
        return response.data;
      } catch (err) {
        setLoading(false);
        const errorMessage =
          err.response?.data?.detail || err.message || "API call failed";
        setError(errorMessage);
        throw new Error(errorMessage);
      }
    },
    []
  );

  // Data fetching methods
  const fetchTrendingVideos = useCallback(
    async (filters = {}) => {
      const params = new URLSearchParams();
      if (filters.country) params.append("country", filters.country);
      if (filters.category) params.append("category", filters.category);
      if (filters.date) params.append("date", filters.date);
      if (filters.sortBy) params.append("sort_by", filters.sortBy);
      if (filters.order) params.append("order", filters.order);
      return apiCall("GET", `/trending?${params.toString()}`);
    },
    [apiCall]
  );

  const fetchStatistics = useCallback(
    async (country = null) => {
      const params = country ? `?country=${country}` : "";
      return apiCall("GET", `/statistics${params}`);
    },
    [apiCall]
  );

  const fetchCountries = useCallback(async () => {
    return apiCall("GET", "/countries");
  }, [apiCall]);

  const fetchCategories = useCallback(
    async (country = null) => {
      const params = country ? `?country=${country}` : "";
      return apiCall("GET", `/categories${params}`);
    },
    [apiCall]
  );

  const fetchWordcloudData = useCallback(
    async (country = null) => {
      const params = country ? `?country=${country}` : "";
      return apiCall("GET", `/wordcloud${params}`);
    },
    [apiCall]
  );

  const fetchDates = useCallback(
    async (country = null) => {
      const params = country ? `?country=${country}` : "";
      return apiCall("GET", `/dates${params}`);
    },
    [apiCall]
  );

  // ML Methods
  const checkMLHealth = useCallback(async () => {
    return apiCall("GET", "/ml/health");
  }, [apiCall]);

  const trainModels = useCallback(async () => {
    return apiCall("POST", "/ml/train");
  }, [apiCall]);

  const predictDays = useCallback(
    async (videoData) => {
      return apiCall("POST", "/ml/predict/days", videoData);
    },
    [apiCall]
  );

  const predictCluster = useCallback(
    async (videoData) => {
      return apiCall("POST", "/ml/predict/cluster", videoData);
    },
    [apiCall]
  );

  // predict by YouTube URL (backend uses YOUTUBE_API_KEY from server env)
  const predictByUrl = useCallback(
    async (url) => {
      return apiCall("POST", "/ml/predict/url", { url });
    },
    [apiCall]
  );

  // Data processing
  const processData = useCallback(async () => {
    return apiCall("POST", "/data/process");
  }, [apiCall]);

  const fetchDatabaseStats = useCallback(async () => {
    return apiCall("GET", "/database-stats");
  }, [apiCall]);

  // Initialize API health check on mount
  useEffect(() => {
    checkApiHealth();
  }, [checkApiHealth]);

  const value = {
    // State
    loading,
    error,
    apiHealth,

    // Methods
    checkApiHealth,
    fetchTrendingVideos,
    fetchStatistics,
    fetchCountries,
    fetchCategories,
    fetchWordcloudData,
    fetchDates,
    checkMLHealth,
    trainModels,
    predictDays,
    predictCluster,
    predictByUrl,
    processData,
    fetchDatabaseStats,

    // Utilities
    setError,
    setLoading,
  };

  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
};
