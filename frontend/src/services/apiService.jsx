import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    
    // Handle different error types
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.detail || error.response.data?.message || 'Server error';
      throw new Error(message);
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('No response from server. Please check if the backend is running.');
    } else {
      // Something else happened
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
);

export const apiService = {
  // Health check
  async getHealth() {
    const response = await api.get('/health');
    return response.data;
  },

  // Get API status
  async getStatus() {
    const response = await api.get('/');
    return response.data;
  },

  // Get available countries
  async getCountries() {
    const response = await api.get('/countries');
    return response.data;
  },

  // Get available dates for a country
  async getDates(country = null) {
    const params = country ? { country } : {};
    const response = await api.get('/dates', { params });
    return response.data;
  },

  // Get trending videos
  async getTrendingVideos(country, date, limit = 10) {
    const response = await api.get('/trending', {
      params: { country, date, limit }
    });
    return response.data;
  },

  // Get wordcloud data
  async getWordcloudData(country, date, limit = 50) {
    const response = await api.get('/wordcloud', {
      params: { country, date, limit }
    });
    return response.data;
  },

  // Get analytics overview
  async getAnalytics(country = null, startDate = null, endDate = null) {
    const params = {};
    if (country) params.country = country;
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    
    const response = await api.get('/analytics', { params });
    return response.data;
  },

  // Get video details
  async getVideoDetails(videoId) {
    const response = await api.get(`/video/${videoId}`);
    return response.data;
  },

  // Get categories statistics
  async getCategoriesStats(country = null) {
    const params = country ? { country } : {};
    const response = await api.get('/categories', { params });
    return response.data;
  },

  // Machine Learning Analysis Endpoints
  async getMLClustering() {
    const response = await api.get('/ml-analysis/clustering');
    return response.data;
  },

  async getMLPredictions() {
    const response = await api.get('/ml-analysis/predictions');
    return response.data;
  },

  async getMLSentiment() {
    const response = await api.get('/ml-analysis/sentiment');
    return response.data;
  },

  async getMLAnomalies() {
    const response = await api.get('/ml-analysis/anomalies');
    return response.data;
  },

  async getMLCategories() {
    const response = await api.get('/ml-analysis/categories');
    return response.data;
  }
};

export default api;
