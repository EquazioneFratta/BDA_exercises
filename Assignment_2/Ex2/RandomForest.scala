/*
 * AI PROMPT (Claude.ai):
 * Following a similar approach to the provided RunDecisionTrees.scala notebook
 * from Moodle, how can I train a RandomForest regressor on the NOAA weather
 * dataset stored in noaaDF to predict air temperature from 12 input features
 * (year, month, day, hour, latitude, longitude, elevation, windDirection,
 * windSpeed, ceilingHeight, visibility, dewPointTemp)?
 * The model should use n=10 regression trees and automatic feature subset
 * selection mode ("auto"), matching the configuration used in the reference
 * script. Use the same year-based train/validation splits as the LinearRegression
 * script (training: 1949–2023, validation: 2024), and evaluate the model using
 * RMSE, MAE and R² via the RegressionEvaluator API.
 * Additionally, print the feature importances produced by the RandomForest model
 * in descending order, to help identify which attributes are most predictive of
 * air temperature and provide an early answer to Problem 2(e).
 * Make sure all method chains are on single lines to avoid spark-shell
 * parsing issues.
 */

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

val startTime = System.currentTimeMillis()

// ── 1. Assemble feature vector (same as LinearRegression) ─────────────────────
val featureCols = Array("year", "month", "day", "hour", "latitude", "longitude", "elevation", "windDirection", "windSpeed", "ceilingHeight", "visibility", "dewPointTemp")
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

// ── 2. Year-based splits (same as LinearRegression) ───────────────────────────
val trainRaw = noaaDF.filter("year >= 1949 AND year <= 2023")
val trainDF = assembler.transform(trainRaw).select("features", "airTemperature").cache()

val validRaw = noaaDF.filter("year = 2024")
val validationDF = assembler.transform(validRaw).select("features", "airTemperature").cache()

println(s"Training records:   ${trainDF.count()}")
println(s"Validation records: ${validationDF.count()}")

// ── 3. Train RandomForest regressor ───────────────────────────────────────────
// n=10 trees, "auto" feature subset selection mode (similar to reference script)
val rf = new RandomForestRegressor().setLabelCol("airTemperature").setFeaturesCol("features").setNumTrees(10).setFeatureSubsetStrategy("auto").setSeed(42)

val rfModel = rf.fit(trainDF)

// ── 4. Evaluate on validation set (2024) ──────────────────────────────────────
val evaluator = new RegressionEvaluator().setLabelCol("airTemperature").setPredictionCol("prediction")

val rfPredictions = rfModel.transform(validationDF)

val rmse = evaluator.setMetricName("rmse").evaluate(rfPredictions)
val mae  = evaluator.setMetricName("mae").evaluate(rfPredictions)
val r2   = evaluator.setMetricName("r2").evaluate(rfPredictions)

println(s"Validation RMSE: $rmse")
println(s"Validation MAE:  $mae")
println(s"Validation R2:   $r2")

// ── 5. Feature importances (bonus insight) ────────────────────────────────────
val importances = rfModel.featureImportances
featureCols.zip(importances.toArray).sortBy(-_._2).foreach { case (feat, imp) =>
  println(f"$feat%-20s $imp%.4f")
}

// ── 6. Show some predictions vs actuals ───────────────────────────────────────
rfPredictions.select("airTemperature", "prediction").show(20)




val endTime = System.currentTimeMillis()
val elapsedSeconds = (endTime - startTime) / 1000.0
println(f"Runtime: $elapsedSeconds%.2f seconds")