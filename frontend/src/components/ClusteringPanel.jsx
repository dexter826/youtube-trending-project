import React, { useState, useEffect } from 'react';
import apiService from '../services/apiService';

const ClusteringPanel = () => {
    const [clusteringData, setClusteringData] = useState(null);
    const [selectedVideo, setSelectedVideo] = useState(null);
    const [clusteringResult, setClusteringResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [trainingStatus, setTrainingStatus] = useState(null);

    useEffect(() => {
        loadClusteringStatistics();
    }, []);

    const loadClusteringStatistics = async () => {
        try {
            setLoading(true);
            const response = await apiService.getAdvancedClusteringStatistics();
            setClusteringData(response);
        } catch (error) {
            console.error('Failed to load clustering statistics:', error);
        } finally {
            setLoading(false);
        }
    };

    const trainAdvancedClustering = async () => {
        try {
            setLoading(true);
            setTrainingStatus('Training advanced clustering models...');
            
            const response = await apiService.trainAdvancedClustering();
            setTrainingStatus(response.status === 'success' 
                ? 'Advanced clustering models trained successfully!' 
                : 'Training failed: ' + response.message);
            
            // Reload statistics after training
            setTimeout(() => {
                loadClusteringStatistics();
                setTrainingStatus(null);
            }, 2000);
            
        } catch (error) {
            setTrainingStatus('Training failed: ' + error.message);
            setTimeout(() => setTrainingStatus(null), 3000);
        } finally {
            setLoading(false);
        }
    };

    const analyzeVideo = async () => {
        if (!selectedVideo) return;

        try {
            setLoading(true);
            const response = await apiService.getComprehensiveClustering(selectedVideo);
            setClusteringResult(response);
        } catch (error) {
            console.error('Failed to analyze video:', error);
        } finally {
            setLoading(false);
        }
    };

    const getClusterColor = (clusterId) => {
        const colors = [
            'bg-blue-100 text-blue-800',
            'bg-green-100 text-green-800',
            'bg-yellow-100 text-yellow-800',
            'bg-red-100 text-red-800',
            'bg-purple-100 text-purple-800',
            'bg-indigo-100 text-indigo-800',
            'bg-pink-100 text-pink-800',
            'bg-gray-100 text-gray-800'
        ];
        return colors[clusterId % colors.length];
    };

    return (
        <div className="p-6 bg-white rounded-lg shadow-lg">
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-800">
                    🧠 Advanced Clustering Analysis
                </h2>
                <button
                    onClick={trainAdvancedClustering}
                    disabled={loading}
                    className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50"
                >
                    {loading ? 'Training...' : 'Train Models'}
                </button>
            </div>

            {trainingStatus && (
                <div className={`mb-4 p-4 rounded-md ${
                    trainingStatus.includes('success') 
                        ? 'bg-green-100 text-green-700' 
                        : 'bg-red-100 text-red-700'
                }`}>
                    {trainingStatus}
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Clustering Statistics */}
                <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-gray-700">Clustering Statistics</h3>
                    
                    {clusteringData ? (
                        <div className="space-y-3">
                            <div className="bg-gray-50 p-4 rounded-md">
                                <h4 className="font-medium text-gray-600">Available Models</h4>
                                <p className="text-2xl font-bold text-indigo-600">
                                    {clusteringData.clustering_statistics?.total_models || 0}
                                </p>
                            </div>
                            
                            <div className="bg-gray-50 p-4 rounded-md">
                                <h4 className="font-medium text-gray-600">Clustering Types</h4>
                                <div className="flex flex-wrap gap-2 mt-2">
                                    {Object.entries(clusteringData.clustering_statistics?.cluster_types || {}).map(([type, count]) => (
                                        <span
                                            key={type}
                                            className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
                                        >
                                            {type}: {count}
                                        </span>
                                    ))}
                                </div>
                            </div>

                            <div className="bg-gray-50 p-4 rounded-md">
                                <h4 className="font-medium text-gray-600">Loaded Models</h4>
                                <div className="flex flex-wrap gap-2 mt-2">
                                    {(clusteringData.clustering_statistics?.loaded_models || []).map((model) => (
                                        <span
                                            key={model}
                                            className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-sm"
                                        >
                                            {model}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="text-gray-500">Loading clustering statistics...</div>
                    )}
                </div>

                {/* Video Analysis */}
                <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-gray-700">Video Clustering Analysis</h3>
                    
                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Sample Video Data
                            </label>
                            <textarea
                                className="w-full p-3 border border-gray-300 rounded-md"
                                rows="6"
                                placeholder='{"title": "Amazing Tech Review", "views": 1000000, "likes": 50000, "comment_count": 2500, "category_id": 28, "country": "US", "publish_time": "2024-01-15T14:30:00Z"}'
                                value={selectedVideo ? JSON.stringify(selectedVideo, null, 2) : ''}
                                onChange={(e) => {
                                    try {
                                        setSelectedVideo(JSON.parse(e.target.value));
                                    } catch {
                                        // Invalid JSON, ignore
                                    }
                                }}
                            />
                        </div>
                        
                        <button
                            onClick={analyzeVideo}
                            disabled={!selectedVideo || loading}
                            className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                        >
                            Analyze Video Clusters
                        </button>
                    </div>

                    {/* Clustering Results */}
                    {clusteringResult && (
                        <div className="mt-6 space-y-4">
                            <h4 className="font-semibold text-gray-700">Clustering Results</h4>
                            
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                {/* Behavioral Cluster */}
                                <div className="bg-gray-50 p-3 rounded-md">
                                    <h5 className="font-medium text-gray-600">🎯 Behavioral</h5>
                                    <span className={`inline-block px-2 py-1 rounded-full text-sm mt-1 ${
                                        getClusterColor(clusteringResult.behavioral_cluster?.cluster_id || 0)
                                    }`}>
                                        {clusteringResult.behavioral_cluster?.cluster_name || 'Unknown'}
                                    </span>
                                    <p className="text-xs text-gray-500 mt-1">
                                        Confidence: {((clusteringResult.behavioral_cluster?.confidence || 0) * 100).toFixed(1)}%
                                    </p>
                                </div>

                                {/* Content Cluster */}
                                <div className="bg-gray-50 p-3 rounded-md">
                                    <h5 className="font-medium text-gray-600">📝 Content</h5>
                                    <span className={`inline-block px-2 py-1 rounded-full text-sm mt-1 ${
                                        getClusterColor(clusteringResult.content_cluster?.cluster_id || 0)
                                    }`}>
                                        {clusteringResult.content_cluster?.cluster_name || 'Unknown'}
                                    </span>
                                    <p className="text-xs text-gray-500 mt-1">
                                        Confidence: {((clusteringResult.content_cluster?.confidence || 0) * 100).toFixed(1)}%
                                    </p>
                                </div>

                                {/* Geographic Cluster */}
                                <div className="bg-gray-50 p-3 rounded-md">
                                    <h5 className="font-medium text-gray-600">🌍 Geographic</h5>
                                    <span className={`inline-block px-2 py-1 rounded-full text-sm mt-1 ${
                                        getClusterColor(clusteringResult.geographic_cluster?.cluster_id || 0)
                                    }`}>
                                        {clusteringResult.geographic_cluster?.cluster_name || 'Unknown'}
                                    </span>
                                    <p className="text-xs text-gray-500 mt-1">
                                        Confidence: {((clusteringResult.geographic_cluster?.confidence || 0) * 100).toFixed(1)}%
                                    </p>
                                </div>

                                {/* Temporal Cluster */}
                                <div className="bg-gray-50 p-3 rounded-md">
                                    <h5 className="font-medium text-gray-600">⏰ Temporal</h5>
                                    <span className={`inline-block px-2 py-1 rounded-full text-sm mt-1 ${
                                        getClusterColor(clusteringResult.temporal_cluster?.cluster_id || 0)
                                    }`}>
                                        {clusteringResult.temporal_cluster?.cluster_name || 'Unknown'}
                                    </span>
                                    <p className="text-xs text-gray-500 mt-1">
                                        Confidence: {((clusteringResult.temporal_cluster?.confidence || 0) * 100).toFixed(1)}%
                                    </p>
                                </div>
                            </div>

                            <div className="bg-indigo-50 p-4 rounded-md">
                                <h5 className="font-medium text-indigo-700">Overall Clustering Score</h5>
                                <p className="text-2xl font-bold text-indigo-600">
                                    {((clusteringResult.overall_clustering_score || 0) * 100).toFixed(1)}%
                                </p>
                                <p className="text-sm text-indigo-600 mt-1">
                                    Framework: {clusteringResult.clustering_framework}
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Algorithm Information */}
            <div className="mt-8 bg-gray-50 p-4 rounded-md">
                <h3 className="text-lg font-semibold text-gray-700 mb-3">
                    🔬 Advanced Clustering Algorithms
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="text-center">
                        <h4 className="font-medium text-gray-600">Behavioral</h4>
                        <p className="text-sm text-gray-500">K-Means, Bisecting K-Means, Gaussian Mixture</p>
                    </div>
                    <div className="text-center">
                        <h4 className="font-medium text-gray-600">Content</h4>
                        <p className="text-sm text-gray-500">Word2Vec + K-Means</p>
                    </div>
                    <div className="text-center">
                        <h4 className="font-medium text-gray-600">Geographic</h4>
                        <p className="text-sm text-gray-500">K-Means Country Patterns</p>
                    </div>
                    <div className="text-center">
                        <h4 className="font-medium text-gray-600">Temporal</h4>
                        <p className="text-sm text-gray-500">K-Means Time Patterns</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ClusteringPanel;
