import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

object SparkWordCountExtended {
  
  def main(args: Array[String]): Unit = {
    
    if (args.length < 2) {
      System.err.println("Correct arguments: <input-directory> <output-directory>")
      System.exit(1)
    }

    val sc = new SparkContext(new SparkConf().setAppName("SparkWordCountExtended"))
    
    // old Hadoop trick: make sure we can recursively scan directories for text files!
    sc.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive", "true")

    val textFile = sc.textFile(args(0))
    /* 
        Core code.
    */
    val pairCounts = textFile.flatMap { line =>
    val words = line.toLowerCase
      .replaceAll("""[^a-z0-9\s']""", " ")
      .split("\\s+")
      .filter(_.nonEmpty)
      .filter(_.length >= 2)

      if (words.length < 2) Seq.empty
      else {
        // zip(drop(n))：n-gram sliding window
        // Distance 1. case (a, b) is a key and 1 is the value.
        val dist1 = words.zip(words.drop(1)).map { case (a, b) => ((a, b), 1) }
        val dist2 = words.zip(words.drop(2)).map { case (a, b) => ((a, b), 1) } //Distance 2.
        dist1 ++ dist2   // 用 union() 也可以：dist1.union(dist2)
      }
    }
    .reduceByKey(_ + _)   // Combine the values under the same key.
    .sortBy(_._2, ascending = false)          // Sort by frequency in descending order



    // Save the result to the output directory.
    val outputDir = args(1)
    // Delete old directories
    import org.apache.hadoop.fs.{FileSystem, Path}
    val fs = FileSystem.get(sc.hadoopConfiguration)
    fs.delete(new Path(outputDir), true)

    // Result: ((word1,word2),count)
    pairCounts.saveAsTextFile(outputDir)
    sc.stop()
  }
}
