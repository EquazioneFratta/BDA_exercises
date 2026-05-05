/*
 * AI PROMPT (Claude.ai):
 * I have a cached RDD[Array[Any]] named parsedNOAA, produced by the provided
 * parseNOAA parsing function which extracts 13 fields from raw NOAA weather
 * observation files. The fields in order are: year, month, day, hour, latitude,
 * longitude, elevation, windDirection, windSpeed, ceilingHeight, visibility,
 * dewPointTemp and airTemperature.
 * How can I convert this RDD into a properly typed Spark DataFrame with named
 * columns and explicit data types (IntegerType for integer fields, DoubleType
 * for decimal fields), using a StructType schema and spark.createDataFrame()?
 * The resulting DataFrame should then be cached to avoid re-processing the
 * input files on every subsequent model run, and verified with count() and
 * show() to confirm the conversion was successful.
 */


import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

val startTime = System.currentTimeMillis()

val noaaSchema = StructType(Seq(
  StructField("year",             IntegerType, nullable = false),
  StructField("month",            IntegerType, nullable = false),
  StructField("day",              IntegerType, nullable = false),
  StructField("hour",             IntegerType, nullable = false),
  StructField("latitude",         DoubleType,  nullable = false),
  StructField("longitude",        DoubleType,  nullable = false),
  StructField("elevation",        IntegerType, nullable = false),
  StructField("windDirection",    IntegerType, nullable = false),
  StructField("windSpeed",        DoubleType,  nullable = false),
  StructField("ceilingHeight",    IntegerType, nullable = false),
  StructField("visibility",       IntegerType, nullable = false),
  StructField("dewPointTemp",     DoubleType,  nullable = false),
  StructField("airTemperature",   DoubleType,  nullable = false)
))

val noaaDF = spark.createDataFrame(
  parsedNOAA.map(arr => Row.fromSeq(arr.toSeq)),
  noaaSchema
)

noaaDF.cache()
println(s"Total records: ${noaaDF.count()}")
noaaDF.show(10)
noaaDF.printSchema()


val endTime = System.currentTimeMillis()
val elapsedSeconds = (endTime - startTime) / 1000.0
println(f"Runtime: $elapsedSeconds%.2f seconds")