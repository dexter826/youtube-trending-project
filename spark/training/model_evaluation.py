"""
Model Evaluation Module
"""

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
    ClusteringEvaluator
)


class ModelEvaluation:
    def __init__(self, spark_session):
        self.spark = spark_session

    def evaluate_trending_prediction_model(self, predictions, feature_cols):
        """Evaluate trending prediction model and return metrics"""
        # Binary classification metrics
        binary_evaluator = BinaryClassificationEvaluator(labelCol="is_trending")
        auc = binary_evaluator.evaluate(predictions)

        # Multiclass classification metrics
        multi_evaluator = MulticlassClassificationEvaluator(
            labelCol="is_trending",
            predictionCol="prediction"
        )

        accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
        precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
        recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
        f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})

        trending_metrics = {
            "auc": float(auc),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "features_used": feature_cols
        }

        return trending_metrics

    def evaluate_clustering_model(self, predictions, feature_cols):
        """Evaluate clustering model and return metrics"""
        evaluator = ClusteringEvaluator(
            predictionCol="cluster",
            featuresCol="scaledFeatures"
        )
        silhouette = evaluator.evaluate(predictions)

        clustering_metrics = {
            "silhouette_score": float(silhouette),
            "num_clusters": 4,
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

    def get_category_distribution(self, df):
        """Get category distribution for analysis"""
        category_counts = df.groupBy("category_id").count().orderBy("count", ascending=False)
        return category_counts