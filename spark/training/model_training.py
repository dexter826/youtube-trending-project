"""
Model Training Module
"""

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline


class ModelTraining:
    def __init__(self, spark_session):
        self.spark = spark_session

    def train_days_regression_model(self, data, feature_cols):
        """Train Random Forest Regressor to predict days_in_trending"""
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
        rf_regressor = RandomForestRegressor(
            featuresCol="scaledFeatures",
            labelCol="days_in_trending",
            predictionCol="prediction",
            numTrees=100,
            maxDepth=10,
            seed=42
        )

        pipeline = Pipeline(stages=[assembler, scaler, rf_regressor])

        train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
        model = pipeline.fit(train_data)

        predictions = model.transform(test_data)

        return model, train_data, test_data, predictions

    def train_clustering_model(self, data, feature_cols):
        """Train clustering model using K-Means"""
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
        kmeans = KMeans(
            featuresCol="scaledFeatures",
            predictionCol="cluster",
            k=3,
            seed=42,
            maxIter=200
        )

        pipeline = Pipeline(stages=[assembler, scaler, kmeans])
        model = pipeline.fit(data)

        predictions = model.transform(data)

        return model, predictions