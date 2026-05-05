/*
 * AI PROMPT (Claude.ai):
 * I have two regression models already trained and available in my Spark session:
 * lrModel (LinearRegression) and rfModel (RandomForestRegressor), both trained
 * on NOAA weather data from 1949 to 2023 and validated on 2024 data.
 * How can I compare these two models by using the full year of 2025 as a test
 * set, filtering noaaDF for year = 2025? For each model, apply the same
 * VectorAssembler transformation used during training, generate predictions on
 * the 2025 test set, and compute the Mean Squared Error (MSE) and Root Mean
 * Squared Error (RMSE) using the RegressionEvaluator API with metricName="mse".
 * Print a side-by-side comparison table of both models showing MSE and RMSE,
 * and show a sample of predicted vs actual temperatures for each model to
 * visually inspect the quality of predictions.
 */

import org.apache.spark.ml.evaluation.RegressionEvaluator

val startTime = System.currentTimeMillis()

// ── 1. Build 2025 test set ─────────────────────────────────────────────────────
val testRaw = noaaDF.filter("year = 2025")
val testDF = assembler.transform(testRaw).select("features", "airTemperature").cache()
println(s"Test records (2025): ${testDF.count()}")

// ── 2. MSE evaluator ──────────────────────────────────────────────────────────
val mseEvaluator = new RegressionEvaluator().setLabelCol("airTemperature").setPredictionCol("prediction").setMetricName("mse")

// ── 3. Linear Regression on 2025 ──────────────────────────────────────────────
val lrTestPredictions = lrModel.transform(testDF)
val lrMSE = mseEvaluator.evaluate(lrTestPredictions)
println(s"Linear Regression  MSE on 2025: $lrMSE")

// ── 4. Random Forest on 2025 ──────────────────────────────────────────────────
val rfTestPredictions = rfModel.transform(testDF)
val rfMSE = mseEvaluator.evaluate(rfTestPredictions)
println(s"Random Forest      MSE on 2025: $rfMSE")

// ── 5. Side by side comparison ────────────────────────────────────────────────
println("\n=== Model Comparison on 2025 Test Set ===")
println(f"Linear Regression  MSE: $lrMSE%.4f  RMSE: ${math.sqrt(lrMSE)}%.4f")
println(f"Random Forest      MSE: $rfMSE%.4f  RMSE: ${math.sqrt(rfMSE)}%.4f")

// ── 6. Show sample predictions from both models ───────────────────────────────
println("\nLinear Regression predictions:")
lrTestPredictions.select("airTemperature", "prediction").show(10)
println("\nRandom Forest predictions:")
rfTestPredictions.select("airTemperature", "prediction").show(10)



val endTime = System.currentTimeMillis()
val elapsedSeconds = (endTime - startTime) / 1000.0
println(f"Runtime: $elapsedSeconds%.2f seconds")