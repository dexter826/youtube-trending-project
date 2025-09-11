import React, { useState, useEffect } from 'react';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';
import apiService from '../services/apiService';

const ModelEvaluation = () => {
  const [evaluationData, setEvaluationData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchEvaluationData();
  }, []);

  const fetchEvaluationData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getModelEvaluation();
      setEvaluationData(data);
    } catch (err) {
      setError(err.message || 'Failed to fetch model evaluation data');
    } finally {
      setLoading(false);
    }
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatNumber = (value) => {
    return value.toLocaleString();
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error} onRetry={fetchEvaluationData} />;
  if (!evaluationData) return <ErrorMessage message="No evaluation data available" />;

  const { performance_metrics, confusion_matrix, feature_importance, training_date, model_type } = evaluationData;

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 space-y-6">
      <div className="border-b pb-4">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Model Evaluation</h2>
        <div className="flex flex-wrap gap-4 text-sm text-gray-600">
          <span>Model: <strong>{model_type}</strong></span>
          <span>Trained: <strong>{new Date(training_date).toLocaleDateString()}</strong></span>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-blue-800 mb-2">Accuracy</h3>
          <div className="text-3xl font-bold text-blue-600">
            {formatPercentage(performance_metrics.accuracy)}
          </div>
          <p className="text-sm text-gray-600 mt-1">Overall correctness</p>
        </div>

        <div className="bg-green-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-green-800 mb-2">Precision</h3>
          <div className="text-3xl font-bold text-green-600">
            {formatPercentage(performance_metrics.precision)}
          </div>
          <p className="text-sm text-gray-600 mt-1">True positives accuracy</p>
        </div>

        <div className="bg-purple-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-purple-800 mb-2">Recall</h3>
          <div className="text-3xl font-bold text-purple-600">
            {formatPercentage(performance_metrics.recall)}
          </div>
          <p className="text-sm text-gray-600 mt-1">Coverage of true positives</p>
        </div>

        <div className="bg-indigo-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-indigo-800 mb-2">F1-Score</h3>
          <div className="text-3xl font-bold text-indigo-600">
            {formatPercentage(performance_metrics.f1_score)}
          </div>
          <p className="text-sm text-gray-600 mt-1">Harmonic mean of precision & recall</p>
        </div>

        <div className="bg-orange-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-orange-800 mb-2">ROC-AUC</h3>
          <div className="text-3xl font-bold text-orange-600">
            {performance_metrics.roc_auc.toFixed(3)}
          </div>
          <p className="text-sm text-gray-600 mt-1">Area under ROC curve</p>
        </div>

        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Cross-Validation</h3>
          <div className="text-2xl font-bold text-gray-600">
            {formatPercentage(performance_metrics.cv_f1_mean)}
          </div>
          <p className="text-sm text-gray-600 mt-1">
            ± {formatPercentage(performance_metrics.cv_f1_std)}
          </p>
        </div>
      </div>

      {/* Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Confusion Matrix</h3>
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-2 text-center">
              <div></div>
              <div className="font-semibold text-sm">Predicted</div>
              <div></div>
              <div></div>
              <div className="text-xs bg-gray-200 p-1 rounded">Not Trending</div>
              <div className="text-xs bg-gray-200 p-1 rounded">Trending</div>
            </div>
            
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="flex flex-col justify-center">
                <div className="text-xs font-semibold mb-1">Actual</div>
                <div className="text-xs bg-gray-200 p-1 rounded">Not Trending</div>
              </div>
              <div className="bg-green-100 border-2 border-green-300 p-3 rounded">
                <div className="text-lg font-bold text-green-700">
                  {formatNumber(confusion_matrix.true_negatives)}
                </div>
                <div className="text-xs text-green-600">True Negative</div>
              </div>
              <div className="bg-red-100 border-2 border-red-300 p-3 rounded">
                <div className="text-lg font-bold text-red-700">
                  {formatNumber(confusion_matrix.false_positives)}
                </div>
                <div className="text-xs text-red-600">False Positive</div>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="flex items-center justify-center">
                <div className="text-xs bg-gray-200 p-1 rounded">Trending</div>
              </div>
              <div className="bg-red-100 border-2 border-red-300 p-3 rounded">
                <div className="text-lg font-bold text-red-700">
                  {formatNumber(confusion_matrix.false_negatives)}
                </div>
                <div className="text-xs text-red-600">False Negative</div>
              </div>
              <div className="bg-green-100 border-2 border-green-300 p-3 rounded">
                <div className="text-lg font-bold text-green-700">
                  {formatNumber(confusion_matrix.true_positives)}
                </div>
                <div className="text-xs text-green-600">True Positive</div>
              </div>
            </div>
          </div>
          
          <div className="mt-4 text-sm text-gray-600">
            Total samples: {formatNumber(confusion_matrix.total_samples)}
          </div>
        </div>

        {/* Feature Importance */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Top Features ({feature_importance.length} of {evaluationData.total_features})
          </h3>
          <div className="space-y-2">
            {feature_importance.map((feature, index) => {
              const maxImportance = Math.max(...feature_importance.map(f => f.importance));
              const widthPercentage = (feature.importance / maxImportance) * 100;
              
              return (
                <div key={index} className="relative">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm font-medium text-gray-700 truncate">
                      {feature.feature}
                    </span>
                    <span className="text-xs text-gray-500">
                      {feature.importance.toFixed(4)}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${widthPercentage}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Refresh Button */}
      <div className="flex justify-center pt-4">
        <button
          onClick={fetchEvaluationData}
          disabled={loading}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Refreshing...' : 'Refresh Evaluation'}
        </button>
      </div>
    </div>
  );
};

export default ModelEvaluation;
