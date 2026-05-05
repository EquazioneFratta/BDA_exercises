import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.expr

// ====================== Main Program Object ======================
object DataFrame {

  def main(args: Array[String]): Unit = {

    // Create SparkSession
    val spark = SparkSession.builder()
      .appName("Titanic Problem 3")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Header of CSV file is: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    val df = spark.read.option("inferSchema", true).option("header", true).csv("./titanic.csv")

    df.show()

    df.createOrReplaceTempView("titanic")

    // =============================================
    // Problem 3 (a) - MedianAge UDF
    // =============================================

    object MedianAge extends UserDefinedAggregateFunction {

      override def inputSchema: StructType = StructType(StructField("age", IntegerType, true) :: Nil)

      override def bufferSchema: StructType = StructType(StructField("ages", ArrayType(IntegerType)) :: Nil)

      override def dataType: DataType = DoubleType

      override def deterministic: Boolean = true

      override def initialize(buffer: MutableAggregationBuffer): Unit = {
        buffer(0) = Array.empty[Int]
      }

      override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
        val current = buffer.getAs[Seq[Int]](0)
        val age = input.getAs[Integer](0)
        val newAge = if (age == null) 23 else age
        buffer(0) = current :+ newAge
      }

      override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
        val ages1 = buffer1.getAs[Seq[Int]](0)
        val ages2 = buffer2.getAs[Seq[Int]](0)
        buffer1(0) = ages1 ++ ages2
      }

      override def evaluate(buffer: Row): Any = {
        val ages = buffer.getAs[Seq[Int]](0).sorted
        val n = ages.length
        if (n == 0) 23.0
        else if (n % 2 == 1) ages(n / 2).toDouble
        else (ages(n/2 - 1) + ages(n/2)) / 2.0
      }
    }

    // Register the UDF
    spark.udf.register("MedianAge", MedianAge)

    // Calculate the median age (null values are automatically replaced with 23)
    val medianResult = df.agg(expr("MedianAge(age) as median_age"))
    medianResult.show()

    // Verify using SQL
    spark.sql("SELECT MedianAge(age) as median_age FROM titanic").show()

    // Save the result to a CSV file in the current directory
    medianResult.write
      .mode("overwrite")
      .option("header", "true")
      .csv("./median_age_result")

    println("=== Problem 3 (a) Completed: Median Age has been calculated (nulls replaced with 23) ===")
    println("Result has been saved to: ./median_age_result")

    // =============================================
    // Problem 3 (b) - Query using built-in operations
    // =============================================

    println("=== Problem 3 (b) Starting ===")

    // Step 1: Calculate the median age once (reusing the UDF from part a)
    val medianDF = df.agg(expr("MedianAge(age) as median_age"))
    val medianAge = medianDF.collect()(0).getDouble(0)   // Get the scalar value

    println(s"Median Age used for filtering: $medianAge")

    // Step 2: Query using the median value
    val resultB = spark.sql(s"""
      SELECT name, fare 
      FROM titanic 
      WHERE age > $medianAge
    """)

    resultB.show(false)   // Show full name without truncation

    // Save result to CSV
    resultB.write
      .mode("overwrite")
      .option("header", "true")
      .csv("./problem3_b_result")

    println("Problem 3 (b) completed. Result saved to ./problem3_b_result")

    // =============================================
    // Problem 3 (c) - Hypothesis Testing with Simple Numeric Comparison
    // =============================================

    println("=== Problem 3 (c) Starting ===")

    // Calculate average fare for survived and not survived passengers
    val avgFares = spark.sql("""
      SELECT 
        Survived,
        AVG(fare) AS average_fare,
        COUNT(*) AS passenger_count
      FROM titanic 
      GROUP BY Survived
    """)

    avgFares.show()

    // Extract the two average values
    val survivedRow = avgFares.filter("Survived = 1").collect()(0)
    val notSurvivedRow = avgFares.filter("Survived = 0").collect()(0)

    val avgFareSurvived = survivedRow.getDouble(1)
    val avgFareNotSurvived = notSurvivedRow.getDouble(1)

    println(s"Average Fare (Survived)   : $avgFareSurvived")
    println(s"Average Fare (Not Survived): $avgFareNotSurvived")
    println(s"Difference                 : ${avgFareSurvived - avgFareNotSurvived}")

    // Simple conclusion
    if (Math.abs(avgFareSurvived - avgFareNotSurvived) < 10) {
      println("Conclusion: The average fares are quite similar. The hypothesis is supported.")
    } else {
      println("Conclusion: The average fare of survivors is significantly higher than non-survivors.")
      println("             The hypothesis is NOT strongly supported.")
    }

    // Save result to CSV for reference
    avgFares.write
      .mode("overwrite")
      .option("header", "true")
      .csv("./problem3_c_result")

    println("Problem 3 (c) completed. Result saved to ./problem3_c_result")

    spark.stop()
  }
}