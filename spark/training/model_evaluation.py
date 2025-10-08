"""
Model Evaluation Module
"""

from pyspark.ml.evaluation import (
    RegressionEvaluator,
    ClusteringEvaluator
)


class ModelEvaluation:
    def __init__(self, spark_session):
        self.spark = spark_session

    # Focus on clustering and days regression evaluation

    def evaluate_clustering_model(self, predictions, feature_cols):
        """Evaluate clustering model and return metrics"""
        evaluator = ClusteringEvaluator(
            predictionCol="cluster",
            featuresCol="scaledFeatures"
        )
        silhouette = evaluator.evaluate(predictions)

        clustering_metrics = {
            "silhouette_score": float(silhouette),
            "num_clusters": 3,
            "features_used": feature_cols
        }

        return clustering_metrics

    def evaluate_regression_model(self, predictions, feature_cols):
        """Evaluate regression model and return metrics"""
        evaluator = RegressionEvaluator(
            labelCol="log_views",
            predictionCol="prediction"
        )

        rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
        mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

        regression_metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "features_used": feature_cols
        }

        return regression_metrics

    def evaluate_days_regression_model(self, predictions, feature_cols):
        """Evaluate days-in-trending regression model and return metrics"""
        evaluator = RegressionEvaluator(
            labelCol="days_in_trending",
            predictionCol="prediction"
        )

        rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
        mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

        regression_metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "label": "days_in_trending",
            "features_used": feature_cols
        }

        return regression_metrics

    def get_category_distribution(self, df):
        """Get category distribution for analysis"""
        category_counts = df.groupBy("category_id").count().orderBy("count", ascending=False)
        return category_counts