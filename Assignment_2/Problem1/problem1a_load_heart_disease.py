"""
Big Data Analytics – Exercise Sheet #2
Problem 1 (a): Load the "Key Indicators of Heart Disease" CSV into a Spark DataFrame
               using an explicitly defined StructType schema.

University of Luxembourg – Summer Semester 2026
Course: Big Data Analytics (Prof. Dr. Martin Theobald)

Usage (batch job on IRIS HPC via SLURM):
    spark-submit problem1a_load_heart_disease.py

Dataset: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
  ~320 K rows, 18 columns:
    - 9 boolean  (Yes/No strings in the CSV  → kept as StringType for now; see note below)
    - 5 string   (categorical text)
    - 4 decimal  (FloatType)
"""

# =============================================================================
# CONFIGURATION – update this path before submitting to the cluster
# =============================================================================
DATA_PATH = "heart_2020_cleaned.csv"   # e.g. "hdfs:///user/<netid>/data/heart_2020_cleaned.csv"
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType
)

# ---------------------------------------------------------------------------
# 1. Initialise SparkSession  (mirrors RunDecisionTrees.ipynb)
# ---------------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("Problem1a_HeartDisease_LoadCSV")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")   # reduce console noise on the cluster

# ---------------------------------------------------------------------------
# 2. Explicit Schema Definition
#
# Column order matches the Kaggle CSV header exactly:
#
#  HeartDisease    – target label              (Yes / No)  → StringType *
#  BMI             – body mass index           (decimal)   → FloatType
#  Smoking         – ever smoked ≥100 cigs     (Yes / No)  → StringType *
#  AlcoholDrinking – heavy drinker             (Yes / No)  → StringType *
#  Stroke          – ever had a stroke         (Yes / No)  → StringType *
#  PhysicalHealth  – bad physical health days  (0–30)      → FloatType
#  MentalHealth    – bad mental health days    (0–30)      → FloatType
#  DiffWalking     – difficulty walking        (Yes / No)  → StringType *
#  Sex             – biological sex            (Male/Female)→ StringType
#  AgeCategory     – age bracket (13 categories)           → StringType
#  Race            – race / ethnicity          (6 values)  → StringType
#  Diabetic        – diabetic status           (4 values)  → StringType
#  PhysicalActivity– physical activity in past 30 days     → StringType *
#  GenHealth       – self-reported general health (5 cats) → StringType
#  SleepTime       – average sleep hours / night           → FloatType
#  Asthma          – ever told had asthma      (Yes / No)  → StringType *
#  KidneyDisease   – ever told had kidney dis. (Yes / No)  → StringType *
#  SkinCancer      – ever told had skin cancer (Yes / No)  → StringType *
#
# * Boolean columns: the CSV stores "Yes"/"No" strings.  We keep them as
#   StringType here so the raw values are preserved faithfully; a downstream
#   StringIndexer (used in Tasks b–d) will convert them to numeric indices
#   before feeding them into the ML pipeline – exactly as the provided
#   RunDecisionTrees notebook does for categorical features.
# ---------------------------------------------------------------------------

heart_disease_schema = StructType([
    # ── Target label ────────────────────────────────────────────────────────
    StructField("HeartDisease",      StringType(), nullable=True),

    # ── Decimal / continuous features ────────────────────────────────────────
    StructField("BMI",               FloatType(),  nullable=True),

    # ── Boolean features (stored as Yes/No strings in CSV) ──────────────────
    StructField("Smoking",           StringType(), nullable=True),
    StructField("AlcoholDrinking",   StringType(), nullable=True),
    StructField("Stroke",            StringType(), nullable=True),

    # ── Decimal features (continued) ────────────────────────────────────────
    StructField("PhysicalHealth",    FloatType(),  nullable=True),
    StructField("MentalHealth",      FloatType(),  nullable=True),

    # ── Boolean feature ──────────────────────────────────────────────────────
    StructField("DiffWalking",       StringType(), nullable=True),

    # ── Categorical / string features ────────────────────────────────────────
    StructField("Sex",               StringType(), nullable=True),
    StructField("AgeCategory",       StringType(), nullable=True),
    StructField("Race",              StringType(), nullable=True),
    StructField("Diabetic",          StringType(), nullable=True),

    # ── Boolean feature ──────────────────────────────────────────────────────
    StructField("PhysicalActivity",  StringType(), nullable=True),

    # ── Categorical feature ──────────────────────────────────────────────────
    StructField("GenHealth",         StringType(), nullable=True),

    # ── Decimal feature ──────────────────────────────────────────────────────
    StructField("SleepTime",         FloatType(),  nullable=True),

    # ── Boolean features ─────────────────────────────────────────────────────
    StructField("Asthma",            StringType(), nullable=True),
    StructField("KidneyDisease",     StringType(), nullable=True),
    StructField("SkinCancer",        StringType(), nullable=True),
])

# ---------------------------------------------------------------------------
# 3. Load CSV with explicit schema
#    - header=True  : first row is column names → skip it
#    - schema=...   : enforce our StructType instead of inferring
# ---------------------------------------------------------------------------
df = (
    spark.read
    .option("header", True)
    .schema(heart_disease_schema)
    .csv(DATA_PATH)
)

# ---------------------------------------------------------------------------
# 4. Sanity-check: print schema and first few rows
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Schema of the Heart Disease DataFrame")
print("=" * 70)
df.printSchema()

print("\n" + "=" * 70)
print("First 5 rows")
print("=" * 70)
df.show(5, truncate=False)

print("\n" + "=" * 70)
print(f"Total rows loaded: {df.count():,}")
print("=" * 70)

# =============================================================================
# TASK b(i): Train/Test Split + Basic Decision Tree ML Pipeline
# =============================================================================

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# 1. Random 80/20 Train/Test Split
#    - seed=42 ensures reproducibility across runs
# ---------------------------------------------------------------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print(f"Training rows : {train_df.count():,}")
print(f"Testing  rows : {test_df.count():,}")

# ---------------------------------------------------------------------------
# 2. StringIndexer Stages
#
#    Every column that holds strings (both categorical and Yes/No booleans)
#    must be converted to a numeric index before it can enter a VectorAssembler.
#
#    Convention used throughout: indexed column = original name + "Index"
#    handleInvalid="keep" ensures unseen labels in the test set get their
#    own index instead of raising an error.
# ---------------------------------------------------------------------------

# Columns that need indexing (all StringType columns except the label)
categorical_cols = [
    "Smoking",           # boolean Yes/No
    "AlcoholDrinking",   # boolean Yes/No
    "Stroke",            # boolean Yes/No
    "DiffWalking",       # boolean Yes/No
    "Sex",               # categorical  (Male / Female)
    "AgeCategory",       # categorical  (13 age brackets)
    "Race",              # categorical  (6 values)
    "Diabetic",          # categorical  (4 values)
    "PhysicalActivity",  # boolean Yes/No
    "GenHealth",         # categorical  (5 values)
    "Asthma",            # boolean Yes/No
    "KidneyDisease",     # boolean Yes/No
    "SkinCancer",        # boolean Yes/No
]

# Build one StringIndexer per categorical / boolean column
categorical_indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=col + "Index",
        handleInvalid="keep"  
    )
    for col in categorical_cols
]

# Separate StringIndexer for the target label column
label_indexer = StringIndexer(
    inputCol="HeartDisease",
    outputCol="label",          # DecisionTreeClassifier expects a column named "label" by default
    handleInvalid="skip" 
)

# ---------------------------------------------------------------------------
# 3. VectorAssembler Stage
#
#    Combines:
#      - the newly created *Index columns  (indexed categorical / boolean features)
#      - the four original FloatType columns (already numeric, no indexing needed)
#    into a single dense feature vector called "features".
# ---------------------------------------------------------------------------

# Names of the indexed categorical columns produced by the StringIndexers above
indexed_categorical_cols = [col + "Index" for col in categorical_cols]

# Original numeric (FloatType) columns – passed through as-is
numeric_cols = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

assembler = VectorAssembler(
    inputCols=indexed_categorical_cols + numeric_cols,  # order: categorical first, then numeric
    outputCol="features",
    handleInvalid="keep"   # skip/impute rows with null values rather than raising an error
)

# ---------------------------------------------------------------------------
# 4. DecisionTreeClassifier Stage
#
#    Uses the default Spark conventions:
#      - featuresCol="features"  → matches VectorAssembler outputCol above
#      - labelCol="label"        → matches label_indexer outputCol above
#    No hyperparameters are tuned here; that is deferred to b(iii).
# ---------------------------------------------------------------------------
dt_classifier = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    seed=42
)

# ---------------------------------------------------------------------------
# 5. Assemble the Pipeline
#
#    Stage order matters – each stage must produce the columns the next expects:
#      [label_indexer]            → adds "label"
#      [*categorical_indexers]    → adds "<col>Index" for every string column
#      [assembler]                → adds "features"
#      [dt_classifier]            → adds "prediction", "probability", etc.
# ---------------------------------------------------------------------------
pipeline = Pipeline(stages=[label_indexer] + categorical_indexers + [assembler, dt_classifier])

# ---------------------------------------------------------------------------
# 6. Fit the Pipeline on the Training Split
#
#    pipeline.fit() runs every stage in sequence:
#      - Indexers learn the label-to-index mappings from train_df
#      - VectorAssembler builds feature vectors
#      - DecisionTreeClassifier trains the tree
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Fitting the Decision Tree pipeline on the training split …")
print("=" * 70)

pipeline_model = pipeline.fit(train_df)

print("Pipeline fitted successfully.")
print(f"  Number of pipeline stages : {len(pipeline_model.stages)}")

# Quick sanity-check: transform a small sample and inspect the output columns
sample_predictions = pipeline_model.transform(train_df.limit(5))

print("\n" + "=" * 70)
print("Sample output columns after pipeline transform (5 training rows)")
print("=" * 70)
sample_predictions.select(
    "HeartDisease", "label", "features", "prediction"
).show(5, truncate=True)
# =============================================================================
# TASK b(ii): Hyperparameter Grid + CrossValidator Initialisation
# =============================================================================

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# ---------------------------------------------------------------------------
# 1. Retrieve the DecisionTreeClassifier stage from the fitted pipeline
#
#    pipeline_model.stages[-1] would give us the *fitted* estimator, but
#    ParamGridBuilder needs the *unfitted* estimator object (dt_classifier)
#    that we defined in b(i) – so we reference dt_classifier directly.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2. Build the Hyperparameter Grid
#
#    ParamGridBuilder constructs a list of parameter maps; CrossValidator
#    will train & evaluate the pipeline once for every combination.
#
#    Grid size: 2 (impurity) × 3 (maxDepth) × 3 (maxBins) = 18 combinations
#    With 5-fold CV → 18 × 5 = 90 individual Spark jobs in total.
# ---------------------------------------------------------------------------
param_grid = (
    ParamGridBuilder()

    # Splitting criterion used at each tree node
    # - "gini"    : Gini impurity  (default in Spark; tends to be faster)
    # - "entropy" : Information gain (can yield slightly deeper, more accurate trees)
    .addGrid(dt_classifier.impurity,  ["entropy", "gini"]) 

    # Maximum depth of the decision tree
    # - Shallow (6)  : faster, less prone to overfitting, lower variance
    # - Medium (12)  : balanced trade-off
    # - Deep   (24)  : captures complex patterns, higher risk of overfitting
    .addGrid(dt_classifier.maxDepth,  [6, 12, 24])

    # Maximum number of bins used when discretising continuous features
    # - Must be ≥ the highest cardinality of any categorical feature
    # - Higher values → finer splits, more memory & compute
    .addGrid(dt_classifier.maxBins,   [20, 50, 100])

    .build()   # returns a List[Dict[Param, value]] – one dict per combination
)

print(f"Total hyperparameter combinations : {len(param_grid)}")   # expect 18
print("Parameter combinations in the grid:")
for i, params in enumerate(param_grid, start=1):
    readable = {p.name: v for p, v in params.items()}
    print(f"  [{i:>2}] {readable}")

# ---------------------------------------------------------------------------
# 3. Placeholder Evaluator
#
#    CrossValidator requires an Evaluator at construction time to rank folds.
#    We use BinaryClassificationEvaluator (metricName defaults to "areaUnderROC")
#    as a structurally valid placeholder.
#
#    The custom evaluation logic (accuracy, precision, recall, F1, AUC) is
#    implemented in b(iii) after .fit() is called.
# ---------------------------------------------------------------------------
placeholder_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",   # default output column of DecisionTreeClassifier
    metricName="areaUnderROC"           # placeholder metric; overridden in b(iii)
)

# ---------------------------------------------------------------------------
# 4. Initialise the CrossValidator
#
#    estimator          : the full Pipeline from b(i) – CrossValidator will
#                         call pipeline.fit() internally for every fold/combo
#    estimatorParamMaps : the 18-combination grid built above
#    evaluator          : placeholder BinaryClassificationEvaluator
#    numFolds           : 5  → each combo is trained on 4/5 of train_df
#                              and evaluated on the remaining 1/5
#    seed               : 42 → reproducible fold assignments
#
#    NOTE: .fit() is NOT called here – that is deferred to b(iii).
# ---------------------------------------------------------------------------
cross_validator = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=placeholder_evaluator,
    numFolds=5,
    seed=42
)

print("\n" + "=" * 70)
print("CrossValidator initialised successfully (not yet fitted)")
print("=" * 70)
print(f"  Estimator        : {cross_validator.getEstimator()}")
print(f"  Number of folds  : {cross_validator.getNumFolds()}")
print(f"  Param grid size  : {len(cross_validator.getEstimatorParamMaps())}")
print(f"  Evaluator metric : {placeholder_evaluator.getMetricName()}")
print(f"  Total fit jobs   : {cross_validator.getNumFolds() * len(param_grid)}")
# =============================================================================
# TASK b(iii): Fit CrossValidator + Extract & Save Best Model
# TASK b(iv) : Evaluate Best Model on Test Set
# =============================================================================

from pyspark.mllib.evaluation import BinaryClassificationMetrics

# ---------------------------------------------------------------------------
# b(iii) - Fit the CrossValidator & Save Best Model
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("WARNING: CrossValidator.fit() is about to start.")
print("  -> 18 param combinations x 5 folds = 90 Spark jobs.")
print("=" * 70 + "\n")

# NOTE: If testing locally, this might take a while. Uncomment the sample in Task A if needed.
cv_model = cross_validator.fit(train_df)
print("\n" + "=" * 70)
print("CrossValidator Results (Average AUC-ROC across 5 folds for each parameter combination):")
print("=" * 70)
for params, metric in zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics):
    readable_params = {p.name: v for p, v in params.items()}
    print(f"Metric: {metric:.6f} | Parameters: {readable_params}")
BEST_MODEL_PATH = "./best_dt_model"
cv_model.bestModel.write().overwrite().save(BEST_MODEL_PATH)
print(f"\nBest model saved to: {BEST_MODEL_PATH}")

# ---------------------------------------------------------------------------
# b(iv) - Evaluate Best Model on Test Set using BinaryClassificationMetrics
# ---------------------------------------------------------------------------
predictions_df = cv_model.bestModel.transform(test_df)

# Build the RDD of (prediction, label) required by MLLib
score_and_labels_rdd = (
    predictions_df
    .select("prediction", "label")
    .rdd
    .map(lambda row: (float(row["prediction"]), float(row["label"])))
)

# 1. Use the required BinaryClassificationMetrics for AUC
binary_metrics = BinaryClassificationMetrics(score_and_labels_rdd)

print("\n" + "=" * 70)
print("Metrics via BinaryClassificationMetrics API")
print("=" * 70)
print(f"  Area Under ROC (AUC-ROC) : {binary_metrics.areaUnderROC:.6f}")
print(f"  Area Under PR  (AUC-PR)  : {binary_metrics.areaUnderPR:.6f}")

# 2. Manually calculate Accuracy, Precision, and Recall from the RDD 
#    to strictly avoid using MulticlassMetrics!
#    Label 1.0 = HeartDisease, Label 0.0 = No HeartDisease

# NOTE: The assignment requires using BinaryClassificationMetrics to obtain Accuracy, Precision, and Recall.
# However, Spark's RDD-based BinaryClassificationMetrics API only exposes threshold-dependent metrics 
# (e.g., pr(), fMeasureByThreshold()) or ROC/PR areas, but does not provide a direct scalar method for Accuracy.
# To fulfill the requirement of extracting these 3 specific metrics, we manually calculate the confusion 
# matrix (TP, TN, FP, FN) by extracting the labels from the RDD.

TP = score_and_labels_rdd.filter(lambda x: x[0] == 1.0 and x[1] == 1.0).count()
FP = score_and_labels_rdd.filter(lambda x: x[0] == 1.0 and x[1] == 0.0).count()
TN = score_and_labels_rdd.filter(lambda x: x[0] == 0.0 and x[1] == 0.0).count()
FN = score_and_labels_rdd.filter(lambda x: x[0] == 0.0 and x[1] == 1.0).count()

total = TP + FP + TN + FN

accuracy = (TP + TN) / total if total > 0 else 0.0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

print("\n" + "=" * 70)
print("Manual Metrics for 'HeartDisease' (Label = 1.0)")
print("=" * 70)
print(f"  Accuracy  : {accuracy:.6f}")
print(f"  Precision : {precision:.6f}")
print(f"  Recall    : {recall:.6f}")
print("\n" + "=" * 70)
print("Evaluation complete.")
print("=" * 70)


# =============================================================================
# TASK 1(c): Replace CrossValidator with TrainValidationSplit (trainRatio=0.8)
#
#
# Reuses from the existing script:
#   - train_df, test_df          (the 80/20 split from b(i))
#   - pipeline                   (the full Pipeline object from b(i))
#   - dt_classifier              (the unfitted DecisionTreeClassifier from b(i))
#   - param_grid                 (the 18-combination grid from b(ii))
#   - BinaryClassificationMetrics (already imported in b(iv))
#
# =============================================================================

from pyspark.ml.tuning import TrainValidationSplit
# BinaryClassificationEvaluator and BinaryClassificationMetrics are already
# imported in the b(ii) / b(iv) blocks above; no duplicate imports needed.

print("\n" + "=" * 70)
print("PROBLEM 1(c) — TrainValidationSplit (trainRatio=0.8)")
print("=" * 70)

# ---------------------------------------------------------------------------
# c(i)  Build TrainValidationSplit
#
#   Key differences vs CrossValidator:
#     • Uses a single random hold-out split (trainRatio=0.8) instead of k folds.
#     • Faster: only 1 fit per parameter combo → 18 Spark jobs total (vs 90).
#     • Higher variance in the metric estimates because only one split is used.
#
#   trainRatio=0.8  means 80% of train_df is used for internal training and
#   the remaining 20% is used for validation when ranking hyperparameter combos.
#   The outer 80/20 split (train_df / test_df) is preserved and used in c(iv).
# ---------------------------------------------------------------------------
tvs_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"      # same ranking criterion as CrossValidator in b(ii)
)

tvs = TrainValidationSplit(
    estimator=pipeline,            # same full Pipeline as in b(ii)
    estimatorParamMaps=param_grid, # same 18-combination grid as in b(ii)
    evaluator=tvs_evaluator,
    trainRatio=0.8,                # 80 % train / 20 % validation (internal split)
    seed=42                        # reproducibility
)

print(f"  trainRatio          : {tvs.getTrainRatio()}")
print(f"  Param grid size     : {len(tvs.getEstimatorParamMaps())}")
print(f"  Total fit jobs      : {len(param_grid)}  (1 split × 18 combos)")

# ---------------------------------------------------------------------------
# c(ii) / c(iii)  Fit TrainValidationSplit & inspect validation scores
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("WARNING: TrainValidationSplit.fit() is about to start.")
print("  -> 18 param combinations × 1 split = 18 Spark jobs.")
print("=" * 70 + "\n")

tvs_model = tvs.fit(train_df)

print("\n" + "=" * 70)
print("TrainValidationSplit Results (AUC-ROC on the internal validation split):")
print("=" * 70)
for params, metric in zip(tvs_model.getEstimatorParamMaps(), tvs_model.validationMetrics):
    readable = {p.name: v for p, v in params.items()}
    print(f"Metric: {metric:.6f} | Parameters: {readable}")

# Save best model
TVS_BEST_MODEL_PATH = "./best_dt_tvs_model"
tvs_model.bestModel.write().overwrite().save(TVS_BEST_MODEL_PATH)
print(f"\nBest TVS model saved to: {TVS_BEST_MODEL_PATH}")

# ---------------------------------------------------------------------------
# c(iv)  Evaluate best TVS model on the held-out test set
#        (same test_df as used in b(iv) – direct comparison is valid)
# ---------------------------------------------------------------------------
tvs_predictions_df = tvs_model.bestModel.transform(test_df)

tvs_score_and_labels = (
    tvs_predictions_df
    .select("prediction", "label")
    .rdd
    .map(lambda row: (float(row["prediction"]), float(row["label"])))
)

tvs_binary_metrics = BinaryClassificationMetrics(tvs_score_and_labels)

print("\n" + "=" * 70)
print("1(c) Metrics via BinaryClassificationMetrics API  [TVS best model]")
print("=" * 70)
print(f"  Area Under ROC (AUC-ROC) : {tvs_binary_metrics.areaUnderROC:.6f}")
print(f"  Area Under PR  (AUC-PR)  : {tvs_binary_metrics.areaUnderPR:.6f}")

# Manual confusion-matrix metrics (same approach as b(iv))
tvs_TP = tvs_score_and_labels.filter(lambda x: x[0] == 1.0 and x[1] == 1.0).count()
tvs_FP = tvs_score_and_labels.filter(lambda x: x[0] == 1.0 and x[1] == 0.0).count()
tvs_TN = tvs_score_and_labels.filter(lambda x: x[0] == 0.0 and x[1] == 0.0).count()
tvs_FN = tvs_score_and_labels.filter(lambda x: x[0] == 0.0 and x[1] == 1.0).count()

tvs_total = tvs_TP + tvs_FP + tvs_TN + tvs_FN
tvs_accuracy  = (tvs_TP + tvs_TN) / tvs_total if tvs_total > 0 else 0.0
tvs_precision = tvs_TP / (tvs_TP + tvs_FP) if (tvs_TP + tvs_FP) > 0 else 0.0
tvs_recall    = tvs_TP / (tvs_TP + tvs_FN) if (tvs_TP + tvs_FN) > 0 else 0.0

print("\n" + "=" * 70)
print("1(c) Manual Metrics for 'HeartDisease' (Label = 1.0)  [TVS best model]")
print("=" * 70)
print(f"  Accuracy  : {tvs_accuracy:.6f}")
print(f"  Precision : {tvs_precision:.6f}")
print(f"  Recall    : {tvs_recall:.6f}")

# ---------------------------------------------------------------------------
# Side-by-side comparison: CrossValidator (b) vs TrainValidationSplit (c)
#
# The variables cv_model, binary_metrics, accuracy, precision, recall are
# defined in the b(iii)/b(iv) block that precedes this section.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("COMPARISON — CrossValidator (b) vs TrainValidationSplit (c)")
print("=" * 70)
print(f"{'Metric':<25} {'CrossValidator':>18} {'TVS':>18}")
print("-" * 63)
print(f"{'AUC-ROC':<25} {binary_metrics.areaUnderROC:>18.6f} {tvs_binary_metrics.areaUnderROC:>18.6f}")
print(f"{'AUC-PR':<25} {binary_metrics.areaUnderPR:>18.6f}  {tvs_binary_metrics.areaUnderPR:>18.6f}")
print(f"{'Accuracy':<25} {accuracy:>18.6f} {tvs_accuracy:>18.6f}")
print(f"{'Precision':<25} {precision:>18.6f} {tvs_precision:>18.6f}")
print(f"{'Recall':<25} {recall:>18.6f} {tvs_recall:>18.6f}")
print("=" * 70)
print("1(c) complete.\n")

# Release cached plan from b; helps GC before 1(d) if running in one script
train_df.unpersist()
test_df.unpersist()


spark.stop()
