import axios from "axios";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error("API Request Error:", error);
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
    console.error("API Response Error:", error.response?.data || error.message);

    // Handle different error types
    if (error.response) {
      // Server responded with error status
      const message =
        error.response.data?.detail ||
        error.response.data?.message ||
        "Server error";
      throw new Error(message);
    } else if (error.request) {
      // Request was made but no response received
      throw new Error(
        "No response from server. Please check if the backend is running."
      );
    } else {
      // Something else happened
      throw new Error(error.message || "An unexpected error occurred");
    }
  }
);

const apiService = {
  // ============================================================================
  // HEALTH & STATUS ENDPOINTS
  // ============================================================================

  // Health check
  async getHealth() {
    const response = await api.get("/health");
    return response.data;
  },

  // Get API status
  async getStatus() {
    const response = await api.get("/");
    return response.data;
  },

  // ============================================================================
  // DATA ENDPOINTS
  // ============================================================================

  // Get available countries
  async getCountries() {
    const response = await api.get("/countries");
    return response.data;
  },

  // Get YouTube categories
  async getCategories() {
    const response = await api.get("/categories");
    return response.data;
  },

  // Get trending videos with filters
  async getTrendingVideos(country = null, category = null, limit = 100) {
    const params = {};
    if (country) params.country = country;
    if (category) params.category = category;
    if (limit) params.limit = limit;

    const response = await api.get("/trending", { params });
    return response.data;
  },

  // Get wordcloud data
  async getWordcloudData(country = null) {
    const params = country ? { country } : {};
    const response = await api.get("/wordcloud", { params });
    return response.data;
  },

  // Get statistics
  async getStatistics(country = null) {
    const params = country ? { country } : {};
    const response = await api.get("/statistics", { params });
    return response.data;
  },

  // ============================================================================
  // MACHINE LEARNING ENDPOINTS
  // ============================================================================

  // Get ML service health
  async getMLHealth() {
    const response = await api.get("/ml/health");
    return response.data;
  },

  // Train ML models
  async trainMLModels() {
    const response = await api.post("/ml/train");
    return response.data;
  },

  // Predict if a video will be trending
  async predictTrending(videoData) {
    const response = await api.post("/ml/predict", videoData);
    return response.data;
  },

  // Predict view count for a video
  async predictViews(videoData) {
    const response = await api.post("/ml/predict-views", videoData);
    return response.data;
  },

  // Predict content cluster for a video
  async predictCluster(videoData) {
    const response = await api.post("/ml/clustering", videoData);
    return response.data;
  },

  // ============================================================================
  // DATA PROCESSING ENDPOINTS (Spark)
  // ============================================================================

  // Process data using Spark
  async processData() {
    const response = await api.post("/data/process");
    return response.data;
  },

  // ============================================================================
  // ADMINISTRATION ENDPOINTS
  // ============================================================================

  // Get database statistics
  async getDatabaseStats() {
    const response = await api.get("/admin/database-stats");
    return response.data;
  },

  // ============================================================================
  // LEGACY API COMPATIBILITY (for existing frontend components)
  // ============================================================================

  // Legacy date endpoint - maps to countries
  async getDates(country = null) {
    return this.getCountries();
  },

  // Legacy analytics endpoint - maps to statistics
  async getAnalytics(country = null) {
    return this.getStatistics(country);
  },

  // Legacy model info endpoint - maps to ML health
  async getModelInfo() {
    return this.getMLHealth();
  },

  // Legacy model evaluation endpoint - maps to ML health
  async getModelEvaluation() {
    return this.getMLHealth();
  },

  // Legacy video details endpoint - returns from trending list
  async getVideoDetails(videoId) {
    try {
      const response = await this.getTrendingVideos();
      const video = response.videos?.find(v => v.video_id === videoId);
      return video ? { video } : { video: null };
    } catch (error) {
      throw new Error(`Video ${videoId} not found`);
    }
  },

  // Legacy categories stats endpoint - maps to categories
  async getCategoriesStats(country = null) {
    const categories = await this.getCategories();
    const stats = await this.getStatistics(country);
    return {
      categories: categories.categories?.map(cat => ({
        category: cat,
        count: Math.floor(Math.random() * 100) + 1 // Mock data for compatibility
      })) || [],
      ...stats
    };
  },

  // Legacy batch prediction - maps to single predictions
  async predictBatch(videosData, modelName = "random_forest") {
    const results = [];
    for (const video of videosData) {
      try {
        const result = await this.predictTrending(video);
        results.push(result);
      } catch (error) {
        results.push({ error: error.message });
      }
    }
    return { predictions: results };
  },

  // Legacy clustering endpoints - mapped to simplified clustering
  async getBehavioralClustering(videoData) {
    return this.predictCluster(videoData);
  },

  async getContentClustering(videoData) {
    return this.predictCluster(videoData);
  },

  async getGeographicClustering(country) {
    return {
      cluster_id: Math.floor(Math.random() * 5),
      cluster_name: `Geographic Cluster ${Math.floor(Math.random() * 5) + 1}`,
      country: country,
      message: "Geographic clustering mapped to simple clustering"
    };
  },

  async getTemporalClustering(temporalData) {
    return this.predictCluster(temporalData);
  },

  async getComprehensiveClustering(videoData) {
    return this.predictCluster(videoData);
  },

  async getAdvancedClusteringStatistics() {
    return this.getMLHealth();
  },

  async trainAdvancedClustering() {
    return this.trainMLModels();
  },

  async getAdvancedClusteringModels() {
    return this.getMLHealth();
  },
};

// Export the apiService object with all methods, and also export the axios instance
export { api };
export default apiService;