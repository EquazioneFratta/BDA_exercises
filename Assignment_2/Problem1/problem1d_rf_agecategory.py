"""
Big Data Analytics – Exercise Sheet #2
Problem 1 (d): Random Forest classifier on AgeCategory (multiclass target)
               evaluated with PySpark's MulticlassMetrics API.

University of Luxembourg – Summer Semester 2026
Course: Big Data Analytics (Prof. Dr. Martin Theobald)

Usage (standalone SLURM batch job):
    spark-submit problem1d_rf_agecategory.py

Run this as a SEPARATE spark-submit job from problem 1(a-c) because:
  1. The target label changes from binary (HeartDisease) to multiclass (AgeCategory).
  2. The full pipeline (StringIndexers, assembler, classifier) must be rebuilt
     from scratch with AgeCategory as the label – reusing the b/c pipeline
     objects would silently produce incorrect label assignments.
  3. Isolating the job makes SLURM resource accounting cleaner and avoids
     memory pressure from carrying the previous CrossValidator model in the
     same executor heap.
"""

# =============================================================================
# CONFIGURATION – update this path before submitting to the cluster
# =============================================================================
DATA_PATH = "heart_2020_cleaned.csv"
# On IRIS HPC replace with your HDFS path, e.g.:
# DATA_PATH = "hdfs:///user/<netid>/data/heart_2020_cleaned.csv"
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

# =============================================================================
# 1. SparkSession
# =============================================================================
spark = (
    SparkSession.builder
    .appName("Problem1d_AgeCategory_RandomForest")
    # Increase driver/executor memory if your SLURM allocation allows it;
    # RF with numTrees=20 + depth=24 on 320 K rows is memory-hungry.
    # .config("spark.driver.memory",   "4g")
    # .config("spark.executor.memory", "8g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("\n" + "=" * 70)
print("PROBLEM 1(d) — Random Forest  |  Target: AgeCategory  |  MulticlassMetrics")
print("=" * 70)

# =============================================================================
# 2. Load CSV with the same explicit schema as in 1(a)
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
    StructField("AgeCategory",      StringType(), nullable=True),  # ← NEW target
    StructField("Race",             StringType(), nullable=True),
    StructField("Diabetic",         StringType(), nullable=True),
    StructField("PhysicalActivity", StringType(), nullable=True),
    StructField("GenHealth",        StringType(), nullable=True),
    StructField("SleepTime",        FloatType(),  nullable=True),
    StructField("Asthma",           StringType(), nullable=True),
    StructField("KidneyDisease",    StringType(), nullable=True),
    StructField("SkinCancer",       StringType(), nullable=True),
])

df = (
    spark.read
    .option("header", True)
    .schema(heart_disease_schema)
    .csv(DATA_PATH)
)

print(f"Total rows loaded : {df.count():,}")

# =============================================================================
# d(i)  80/20 Train/Test Split
#        Same seed as problems b/c for reproducibility, but a fresh split on
#        the full DataFrame so the label distribution is consistent.
# =============================================================================
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()   # cache the training set – it will be read 5 × 36 times inside CV
print(f"Training rows : {train_df.count():,}")
print(f"Testing  rows : {test_df.count():,}")

# =============================================================================
# d(ii) Pipeline Construction
#
#  AgeCategory is the new multiclass target (13 ordered age-bracket strings).
#  HeartDisease is now just another feature.
#
#  All StringType columns must be indexed.  We split them into:
#    label_col       – AgeCategory (becomes "label")
#    feature_str_cols – everything else that is StringType
# =============================================================================

# All StringType columns that become FEATURES (HeartDisease is now a feature)
feature_str_cols = [
    "HeartDisease",       # binary categorical feature in this problem
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "Sex",
    "Race",
    "Diabetic",
    "PhysicalActivity",
    "GenHealth",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]

# Label indexer: AgeCategory → "label"
label_indexer = StringIndexer(
    inputCol="AgeCategory",
    outputCol="label",
    handleInvalid="keep"
)

# One StringIndexer per feature string column
feature_indexers = [
    StringIndexer(inputCol=c, outputCol=c + "Index", handleInvalid="keep")
    for c in feature_str_cols
]

# VectorAssembler: indexed feature cols + original numeric cols → "features"
numeric_cols = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]
assembler = VectorAssembler(
    inputCols=[c + "Index" for c in feature_str_cols] + numeric_cols,
    outputCol="features",
    handleInvalid="keep"
)

# RandomForestClassifier
#   featureSubsetStrategy="auto" → sqrt(numFeatures) for classification (Spark default)
rf_classifier = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    featureSubsetStrategy="auto",   # "automatic" training mode per assignment spec
    seed=42
)

pipeline_d = Pipeline(
    stages=[label_indexer] + feature_indexers + [assembler, rf_classifier]
)

# =============================================================================
# d(ii-cont)  Hyperparameter Grid
#
#   Grid dimensions:
#     numTrees  : [5, 10, 20]          – assignment requirement
#     impurity  : ["entropy", "gini"]  – same as b/c for comparability
#     maxDepth  : [6, 12, 24]          – same as b/c
#     maxBins   : [20, 50, 100]        – same as b/c
#
#   Total combos : 3 × 2 × 3 × 3 = 54
#   With 5-fold CV → 54 × 5 = 270 Spark jobs.
#   On IRIS HPC with sufficient cores this typically takes 10-30 min.
#
#   To reduce wall time during development, comment out the larger values
#   (e.g., keep only maxDepth=[6,12], maxBins=[20,50]) and restore for the
#   final SLURM submission.
# =============================================================================
param_grid_d = (
    ParamGridBuilder()
    .addGrid(rf_classifier.numTrees,  [5, 10, 20])
    .addGrid(rf_classifier.impurity,  ["entropy", "gini"])
    .addGrid(rf_classifier.maxDepth,  [6, 12, 24])
    .addGrid(rf_classifier.maxBins,   [20, 50, 100])
    .build()
)

print(f"\nTotal hyperparameter combinations : {len(param_grid_d)}")  # expect 54
print("Parameter combinations in the grid:")
for i, params in enumerate(param_grid_d, start=1):
    readable = {p.name: v for p, v in params.items()}
    print(f"  [{i:>2}] {readable}")

# =============================================================================
# d(iii) CrossValidator  (5 folds, same structure as problem b)
#
#   MulticlassClassificationEvaluator is used as the ranking evaluator inside
#   CrossValidator.  The final external evaluation (d(iv)) will use the
#   MLlib MulticlassMetrics RDD API as required by the assignment.
# =============================================================================
cv_evaluator_d = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"   # rank hyperparameter combos by overall accuracy
)

cross_validator_d = CrossValidator(
    estimator=pipeline_d,
    estimatorParamMaps=param_grid_d,
    evaluator=cv_evaluator_d,
    numFolds=5,
    seed=42,
    parallelism=1
)

print(f"\n  Estimator        : {cross_validator_d.getEstimator()}")
print(f"  Number of folds  : {cross_validator_d.getNumFolds()}")
print(f"  Param grid size  : {len(cross_validator_d.getEstimatorParamMaps())}")
print(f"  Evaluator metric : {cv_evaluator_d.getMetricName()}")
print(f"  Total fit jobs   : {cross_validator_d.getNumFolds() * len(param_grid_d)}")

# =============================================================================
# d(iii-cont)  Fit CrossValidator
# =============================================================================
print("\n" + "=" * 70)
print("WARNING: CrossValidator.fit() is about to start.")
print(f"  -> {len(param_grid_d)} param combinations × 5 folds = "
      f"{len(param_grid_d) * 5} Spark jobs.")
print("  -> This may take 10-30 minutes on the IRIS cluster.")
print("=" * 70 + "\n")

cv_model_d = cross_validator_d.fit(train_df)

# Print CV validation scores for all parameter combos
print("\n" + "=" * 70)
print("CrossValidator Results (avg Accuracy across 5 folds):")
print("=" * 70)
for params, metric in zip(cv_model_d.getEstimatorParamMaps(), cv_model_d.avgMetrics):
    readable = {p.name: v for p, v in params.items()}
    print(f"Metric: {metric:.6f} | Parameters: {readable}")

# Save the best model
RF_BEST_MODEL_PATH = "./best_rf_agecategory_model"
cv_model_d.bestModel.write().overwrite().save(RF_BEST_MODEL_PATH)
print(f"\nBest RF model saved to: {RF_BEST_MODEL_PATH}")

# Log the best parameter set for easy inspection
best_params = cv_model_d.getEstimatorParamMaps()[
    cv_model_d.avgMetrics.index(max(cv_model_d.avgMetrics))
]
print("\nBest parameter combination:")
for p, v in best_params.items():
    print(f"  {p.name}: {v}")

# =============================================================================
# d(iv)  Evaluate best model on test set using MulticlassMetrics API
#
#  MulticlassMetrics expects an RDD of (prediction, label) pairs where BOTH
#  values are floating-point.  We map the DataFrame output accordingly.
# =============================================================================
print("\n" + "=" * 70)
print("Evaluating best RF model on test set …")
print("=" * 70)

# Free training set from cache now that fitting is done
train_df.unpersist()

test_predictions_d = cv_model_d.bestModel.transform(test_df)

pred_and_labels_rdd = (
    test_predictions_d
    .select("prediction", "label")
    .rdd
    .map(lambda row: (float(row["prediction"]), float(row["label"])))
)

# Materialise the RDD once; MulticlassMetrics will iterate over it multiple
# times (once per metric), so caching avoids repeated recomputation.
pred_and_labels_rdd.cache()
pred_and_labels_rdd.count()  # trigger cache population

mc_metrics = MulticlassMetrics(pred_and_labels_rdd)

# ── Overall metrics ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("1(d) OVERALL Metrics via MulticlassMetrics API  [RF best model]")
print("=" * 70)
print(f"  Overall Accuracy             : {mc_metrics.accuracy:.6f}")
print(f"  Weighted Precision           : {mc_metrics.weightedPrecision:.6f}")
print(f"  Weighted Recall              : {mc_metrics.weightedRecall:.6f}")
print(f"  Weighted F1-Score            : {mc_metrics.weightedFMeasure():.6f}")
print(f"  Weighted False Positive Rate : {mc_metrics.weightedFalsePositiveRate:.6f}")

# ── Per-class metrics ─────────────────────────────────────────────────────────
# Retrieve the label-to-AgeCategory mapping from the fitted StringIndexer so
# we can print human-readable class names instead of raw numeric indices.
# The label_indexer is the first stage of the best model's pipeline.
fitted_label_indexer = cv_model_d.bestModel.stages[0]
labels_by_index = {
    float(i): lbl
    for i, lbl in enumerate(fitted_label_indexer.labels)
}

print("\n" + "=" * 70)
print("1(d) PER-CLASS Metrics via MulticlassMetrics API")
print("=" * 70)
print(f"{'Index':<7} {'AgeCategory':<30} {'Precision':>12} {'Recall':>10} {'F1':>10}")
print("-" * 72)

for label_idx in sorted(labels_by_index.keys()):
    label_name = labels_by_index[label_idx]
    try:
        prec = mc_metrics.precision(label_idx)
        rec  = mc_metrics.recall(label_idx)
        f1   = mc_metrics.fMeasure(label_idx)
    except Exception:
        # Some classes may have zero support in the test set
        prec, rec, f1 = 0.0, 0.0, 0.0
    print(f"{int(label_idx):<7} {label_name:<30} {prec:>12.6f} {rec:>10.6f} {f1:>10.6f}")

print("=" * 70)
print("1(d) complete.\n")

# Release RDD cache
pred_and_labels_rdd.unpersist()

# =============================================================================
# Done
# =============================================================================
spark.stop()
