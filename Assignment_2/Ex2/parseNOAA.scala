import org.apache.spark.rdd._

val startTime = System.currentTimeMillis()

def parseNOAA(rawData: RDD[String]): RDD[Array[Any]] = {

  rawData
    .filter(line => line.substring(87, 92) != "+9999") // filter out missing temperature labels
    .map { line =>
      val year = line.substring(15, 19).toInt
      val month = line.substring(19, 21).toInt
      val day = line.substring(21, 23).toInt
      val hour = line.substring(23, 25).toInt
      val latitude = line.substring(28, 34).toDouble / 1000
      val longitude = line.substring(34, 41).toDouble / 1000
      val elevationDimension = line.substring(46, 51).toInt
      val directionAngle = line.substring(60, 63).toInt
      val speedRate = line.substring(65, 69).toDouble / 10
      val ceilingHeightDimension = line.substring(70, 75).toInt
      val distanceDimension = line.substring(78, 84).toInt
      val dewPointTemperature = line.substring(93, 98).toDouble / 10
      val airTemperature = line.substring(87, 92).toDouble / 10

      Array(year, month, day, hour, latitude, longitude, elevationDimension, directionAngle, speedRate, ceilingHeightDimension, distanceDimension, dewPointTemperature, airTemperature)
    }
}

val rawNOAA = sc.textFile("./NOAA-065900/065900*")
val parsedNOAA = parseNOAA(rawNOAA)

parsedNOAA.cache()
parsedNOAA.count()

parsedNOAA.take(10) // take 10 samples
parsedNOAA.take(10)(0)(12) // access e.g. the airTemperature field of the first observation

val endTime = System.currentTimeMillis()
val elapsedSeconds = (endTime - startTime) / 1000.0
println(f"Runtime: $elapsedSeconds%.2f seconds")