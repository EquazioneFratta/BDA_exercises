"""
Problem 1(d) – Part 2: GBT (OneVsRest) comparison only.
Loads the already-saved RF model to recover metrics, then trains GBT.

Usage:
    spark-submit problem1d_gbt_comparison.py
"""

DATA_PATH = "heart_2020_cleaned.csv"
RF_BEST_MODEL_PATH = "./best_rf_agecategory_model"

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier, OneVsRest
from pyspark.ml import PipelineModel
from pyspark.mllib.evaluation import MulticlassMetrics

spark = (
    SparkSession.builder
    .appName("Problem1d_GBT_Comparison")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# =============================================================================
# 1. Reload the dataset (needed for test_df and GBT training)
# =============================================================================
heart_disease_schema = StructType([
    StructField("HeartDisease",     StringType(), nullable=True),
    StructField("BMI",              FloatType(),  nullable=True),
    StructField("Smoking",          StringType(), nullable=True),
    StructField("AlcoholDrinking",  StringType(), nullable=True),
    StructField("Stroke",           StringType(), nullable=True),
    StructField("PhysicalHealth",   FloatType(),  nullable=True),
    StructField("MentalHealth",     FloatType(),  nullable=True),
    StructField("DiffWalking",      StringType(), nullable=True),
    StructField("Sex",              StringType(), nullable=True),
    StructField("AgeCategory",      StringType(), nullable=True),
    StructField("Race",             StringType(), nullable=True),
    StructField("Diabetic",         StringType(), nullable=True),
    StructField("PhysicalActivity", StringType(), nullable=True),
    StructField("GenHealth",        StringType(), nullable=True),
    StructField("SleepTime",        FloatType(),  nullable=True),
    StructField("Asthma",           StringType(), nullable=True),
    StructField("KidneyDisease",    StringType(), nullable=True),
    StructField("SkinCancer",       StringType(), nullable=True),
])

df = spark.read.option("header", True).schema(heart_disease_schema).csv(DATA_PATH)

# Use the SAME seed=42 split so test_df is identical to the original run
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# =============================================================================
# 2. Load the saved RF CrossValidatorModel and recompute its test-set metrics
#    so the comparison table can be printed without re-running CrossValidator
# =============================================================================
print("\nLoading saved RF model …")
rf_cv_model = PipelineModel.load(RF_BEST_MODEL_PATH)

print("Generating RF predictions on test set …")
test_predictions_rf = rf_cv_model.transform(test_df)

pred_and_labels_rf = (
    test_predictions_rf
    .select("prediction", "label")
    .rdd
    .map(lambda row: (float(row["prediction"]), float(row["label"])))
)
pred_and_labels_rf.cache()
pred_and_labels_rf.count()

mc_metrics_rf = MulticlassMetrics(pred_and_labels_rf)

print("\n" + "=" * 70)
print("RF OVERALL Metrics (loaded from saved model)")
print("=" * 70)
print(f"  Overall Accuracy             : {mc_metrics_rf.accuracy:.6f}")
print(f"  Weighted Precision           : {mc_metrics_rf.weightedPrecision:.6f}")
print(f"  Weighted Recall              : {mc_metrics_rf.weightedRecall:.6f}")
print(f"  Weighted F1-Score            : {mc_metrics_rf.weightedFMeasure():.6f}")
print(f"  Weighted False Positive Rate : {mc_metrics_rf.weightedFalsePositiveRate:.6f}")

pred_and_labels_rf.unpersist()

# Recover the AgeCategory label mapping from the fitted RF pipeline
fitted_label_indexer = rf_cv_model.stages[0]
labels_by_index = {
    float(i): lbl
    for i, lbl in enumerate(fitted_label_indexer.labels)
}

# =============================================================================
# 3. Build and fit the GBT (OneVsRest) pipeline
#    Preprocessing stages are defined fresh but will produce identical
#    transformations because the schema and data are the same.
# =============================================================================
feature_str_cols = [
    "HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
    "Sex", "Race", "Diabetic", "PhysicalActivity", "GenHealth",
    "Asthma", "KidneyDisease", "SkinCancer",
]
numeric_cols = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

label_indexer   = StringIndexer(inputCol="AgeCategory", outputCol="label",      handleInvalid="keep")
feature_indexers = [StringIndexer(inputCol=c, outputCol=c + "Index", handleInvalid="keep") for c in feature_str_cols]
assembler       = VectorAssembler(inputCols=[c + "Index" for c in feature_str_cols] + numeric_cols, outputCol="features", handleInvalid="keep")

gbt_base = GBTClassifier(featuresCol="features", labelCol="label", maxIter=20, maxDepth=5, maxBins=32, seed=42)
ovr_gbt  = OneVsRest(classifier=gbt_base, featuresCol="features", labelCol="label", predictionCol="prediction")

pipeline_gbt = Pipeline(stages=[label_indexer] + feature_indexers + [assembler, ovr_gbt])

print("\nFitting GBT (OneVsRest) pipeline …")
train_df.cache()
train_df.count()

gbt_model = pipeline_gbt.fit(train_df)
train_df.unpersist()

gbt_model.write().overwrite().save("./best_gbt_agecategory_model")
print("GBT model saved.")

# =============================================================================
# 4. Evaluate GBT on test set
# =============================================================================
test_predictions_gbt = gbt_model.transform(test_df)

pred_and_labels_gbt = (
    test_predictions_gbt
    .select("prediction", "label")
    .rdd
    .map(lambda row: (float(row["prediction"]), float(row["label"])))
)
pred_and_labels_gbt.cache()
pred_and_labels_gbt.count()

mc_metrics_gbt = MulticlassMetrics(pred_and_labels_gbt)

print("\n" + "=" * 70)
print("GBT (OneVsRest) OVERALL Metrics")
print("=" * 70)
print(f"  Overall Accuracy             : {mc_metrics_gbt.accuracy:.6f}")
print(f"  Weighted Precision           : {mc_metrics_gbt.weightedPrecision:.6f}")
print(f"  Weighted Recall              : {mc_metrics_gbt.weightedRecall:.6f}")
print(f"  Weighted F1-Score            : {mc_metrics_gbt.weightedFMeasure():.6f}")
print(f"  Weighted False Positive Rate : {mc_metrics_gbt.weightedFalsePositiveRate:.6f}")

# =============================================================================
# 5. Side-by-side comparison
# =============================================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON  |  Test-Set Overall Metrics")
print("=" * 70)
print(f"{'Metric':<35} {'Random Forest':>15} {'GBT (OvR)':>15}")
print("-" * 68)
print(f"{'Accuracy':<35} {mc_metrics_rf.accuracy:>15.6f} {mc_metrics_gbt.accuracy:>15.6f}")
print(f"{'Weighted Precision':<35} {mc_metrics_rf.weightedPrecision:>15.6f} {mc_metrics_gbt.weightedPrecision:>15.6f}")
print(f"{'Weighted Recall':<35} {mc_metrics_rf.weightedRecall:>15.6f} {mc_metrics_gbt.weightedRecall:>15.6f}")
print(f"{'Weighted F1-Score':<35} {mc_metrics_rf.weightedFMeasure():>15.6f} {mc_metrics_gbt.weightedFMeasure():>15.6f}")
print(f"{'Weighted False Positive Rate':<35} {mc_metrics_rf.weightedFalsePositiveRate:>15.6f} {mc_metrics_gbt.weightedFalsePositiveRate:>15.6f}")
print("=" * 70)

# =============================================================================
# 6. Per-class metrics for GBT
# =============================================================================
print("\n" + "=" * 70)
print("GBT (OneVsRest) PER-CLASS Metrics")
print("=" * 70)
print(f"{'Index':<7} {'AgeCategory':<30} {'Precision':>12} {'Recall':>10} {'F1':>10}")
print("-" * 72)

for label_idx in sorted(labels_by_index.keys()):
    label_name = labels_by_index[label_idx]
    try:
        prec = mc_metrics_gbt.precision(label_idx)
        rec  = mc_metrics_gbt.recall(label_idx)
        f1   = mc_metrics_gbt.fMeasure(label_idx)
    except Exception:
        prec, rec, f1 = 0.0, 0.0, 0.0
    print(f"{int(label_idx):<7} {label_name:<30} {prec:>12.6f} {rec:>10.6f} {f1:>10.6f}")

print("=" * 70)
pred_and_labels_gbt.unpersist()

spark.stop()
