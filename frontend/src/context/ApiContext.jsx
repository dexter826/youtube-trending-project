import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

const ApiContext = createContext();

export const useApi = () => {
  const context = useContext(ApiContext);
  if (!context) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
};

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Configure axios defaults
axios.defaults.baseURL = API_BASE_URL;
axios.defaults.timeout = 30000;

export const ApiProvider = ({ children }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);

  // API Health Check
  const checkApiHealth = async () => {
    try {
      const response = await axios.get('/health');
      setApiHealth(response.data);
      return response.data;
    } catch (err) {
      setApiHealth({ status: 'unhealthy', error: err.message });
      return null;
    }
  };

  // Generic API call wrapper
  const apiCall = async (method, endpoint, data = null, config = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios({
        method,
        url: endpoint,
        data,
        ...config
      });
      
      setLoading(false);
      return response.data;
    } catch (err) {
      setLoading(false);
      const errorMessage = err.response?.data?.detail || err.message || 'API call failed';
      setError(errorMessage);
      throw new Error(errorMessage);
    }
  };

  // Data fetching methods
  const fetchTrendingVideos = async (filters = {}) => {
    const params = new URLSearchParams();
    if (filters.country) params.append('country', filters.country);
    if (filters.category) params.append('category', filters.category);
    if (filters.limit) params.append('limit', filters.limit);
    
    return apiCall('GET', `/trending?${params.toString()}`);
  };

  const fetchStatistics = async (country = null) => {
    const params = country ? `?country=${country}` : '';
    return apiCall('GET', `/statistics${params}`);
  };

  const fetchCountries = async () => {
    return apiCall('GET', '/countries');
  };

  const fetchCategories = async () => {
    return apiCall('GET', '/categories');
  };

  const fetchWordcloudData = async (country = null) => {
    const params = country ? `?country=${country}` : '';
    return apiCall('GET', `/wordcloud${params}`);
  };

  const fetchDates = async (country = null) => {
    const params = country ? `?country=${country}` : '';
    return apiCall('GET', `/dates${params}`);
  };

  // ML Methods
  const checkMLHealth = async () => {
    return apiCall('GET', '/ml/health');
  };

  const trainModels = async () => {
    return apiCall('POST', '/ml/train');
  };

  const predictTrending = async (videoData) => {
    return apiCall('POST', '/ml/predict', videoData);
  };

  const predictViews = async (videoData) => {
    return apiCall('POST', '/ml/predict-views', videoData);
  };

  const predictCluster = async (videoData) => {
    return apiCall('POST', '/ml/clustering', videoData);
  };

  // Data processing
  const processData = async () => {
    return apiCall('POST', '/data/process');
  };

  const fetchDatabaseStats = async () => {
    return apiCall('GET', '/admin/database-stats');
  };

  // Initialize API health check on mount
  useEffect(() => {
    checkApiHealth();
  }, []);

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
    predictTrending,
    predictViews,
    predictCluster,
    processData,
    fetchDatabaseStats,
    
    // Utilities
    setError,
    setLoading
  };

  return (
    <ApiContext.Provider value={value}>
      {children}
    </ApiContext.Provider>
  );
};