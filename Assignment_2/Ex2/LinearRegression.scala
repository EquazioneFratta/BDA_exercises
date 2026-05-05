/*
 * AI PROMPT (Claude.ai):
 * I have a cached Spark DataFrame named noaaDF containing NOAA weather
 * observations for the Luxembourg Air Force station (USAF 065900) from
 * 1949 to 2025, with columns: year, month, day, hour, latitude, longitude,
 * elevation, windDirection, windSpeed, ceilingHeight, visibility, dewPointTemp
 * and airTemperature.
 * Adapting the provided LinearRegression.scala notebook from Moodle, how can
 * I train a LinearRegression model to predict air temperature from the other
 * 12 attributes? The dataset should be split by year rather than randomly:
 * a training set containing all records from 1949 to 2023, and a validation
 * set containing all records from 2024. I need to use VectorAssembler to
 * assemble the input features, and evaluate the resulting model on the
 * validation set using RMSE, MAE and R² via the RegressionEvaluator API.
 * Also print a sample of predicted vs actual temperatures for inspection.
 * Make sure all method chains are on single lines to avoid spark-shell
 * parsing issues.
 */

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator

val startTime = System.currentTimeMillis()

// ── 1. Assemble feature vector ────────────────────────────────────────────────
val featureCols = Array("year", "month", "day", "hour", "latitude", "longitude", "elevation", "windDirection", "windSpeed", "ceilingHeight", "visibility", "dewPointTemp")
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

// ── 2. Year-based splits ──────────────────────────────────────────────────────
val trainRaw = noaaDF.filter("year >= 1949 AND year <= 2023")
val trainAssembled = assembler.transform(trainRaw)
val trainDF = trainAssembled.select("features", "airTemperature").cache()

val validRaw = noaaDF.filter("year = 2024")
val validAssembled = assembler.transform(validRaw)
val validationDF = validAssembled.select("features", "airTemperature").cache()

println(s"Training records:   ${trainDF.count()}")
println(s"Validation records: ${validationDF.count()}")

// ── 3. Train LinearRegression ─────────────────────────────────────────────────
val lr = new LinearRegression().setLabelCol("airTemperature").setFeaturesCol("features").setMaxIter(100)
val lrModel = lr.fit(trainDF)

// ── 4. Print model summary on training data ───────────────────────────────────
val summary = lrModel.summary
println(s"Training RMSE: ${summary.rootMeanSquaredError}")
println(s"Training R2:   ${summary.r2}")

// ── 5. Evaluate on validation set (2024) ─────────────────────────────────────
val evaluator = new RegressionEvaluator().setLabelCol("airTemperature").setPredictionCol("prediction")
val predictions = lrModel.transform(validationDF)
val rmse = evaluator.setMetricName("rmse").evaluate(predictions)
val mae  = evaluator.setMetricName("mae").evaluate(predictions)
val r2   = evaluator.setMetricName("r2").evaluate(predictions)

println(s"Validation RMSE: $rmse")
println(s"Validation MAE:  $mae")
println(s"Validation R2:   $r2")

// ── 6. Show some predictions vs actuals ──────────────────────────────────────
predictions.select("airTemperature", "prediction").show(20)



val endTime = System.currentTimeMillis()
val elapsedSeconds = (endTime - startTime) / 1000.0
println(f"Runtime: $elapsedSeconds%.2f seconds")