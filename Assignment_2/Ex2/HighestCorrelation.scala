import org.apache.spark.sql.functions.corr

val startTime = System.currentTimeMillis()

val featureCols = Array("year", "month", "day", "hour", "latitude", "longitude", "elevation", "windDirection", "windSpeed", "ceilingHeight", "visibility", "dewPointTemp")

featureCols.foreach { col =>
  val correlation = noaaDF.stat.corr(col, "airTemperature")
  println(f"$col%-20s correlation: $correlation%.4f")
}

val endTime = System.currentTimeMillis()
val elapsedSeconds = (endTime - startTime) / 1000.0
println(f"Runtime: $elapsedSeconds%.2f seconds")