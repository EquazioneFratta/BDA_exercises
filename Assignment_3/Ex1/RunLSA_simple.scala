// ========================================================
// RunLSA_Simple.scala
// SIMPLE TOKENIZER VERSION for Problem 1(a)
// - Uses regex [a-zA-Z0-9]+ (exactly as introduced in Exercise Sheet #1)
// - Works in spark-shell (uses getOrCreate) AND can be compiled to JAR for spark-submit
// ========================================================

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import scala.collection.mutable.{ArrayBuffer, HashMap}
import org.apache.spark.rdd.RDD
import scala.util.matching.Regex
import org.apache.spark.mllib.linalg.{Vectors, Matrix, SingularValueDecomposition}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

object RunLSA_Simple {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    // ====================== Spark Context (works in shell + submit) =====
    // In spark-shell: reuses existing sc
    // When submitted via spark-submit: creates new one
    val sc = SparkContext.getOrCreate(new SparkConf().setAppName("RunLSA_Simple"))

    // ====================== Hyper-parameters (Problem 1(a)) =============
    val sampleSize = 1.0      // ← Change to 0.01 for quick testing
    val numTerms   = 5000     // number of frequent terms
    val k          = 25       // number of latent concepts (top-25)

    // ====================== 1. Parse Wikipedia Articles ==================
    def parseHeader(line: String): Array[String] = {
      try {
        var s = line.substring(line.indexOf("id=\"") + 4)
        val id = s.substring(0, s.indexOf("\""))
        s = s.substring(s.indexOf("url=\"") + 5)
        val url = s.substring(0, s.indexOf("\""))
        s = s.substring(s.indexOf("title=\"") + 7)
        val title = s.substring(0, s.indexOf("\""))
        Array(id, url, title)
      } catch { case _: Exception => Array("", "", "") }
    }

    def parse(lines: Array[String]): Array[(String, String)] = {
      var docs = ArrayBuffer.empty[(String, String)]
      var title = ""
      var content = ""
      for (line <- lines) {
        try {
          if (line.startsWith("<doc ")) {
            title = parseHeader(line)(2)
            content = ""
          } else if (line.startsWith("</doc>")) {
            if (title.nonEmpty && content.nonEmpty) docs += ((title, content))
          } else {
            content += line + "\n"
          }
        } catch { case _: Exception => content = "" }
      }
      docs.toArray
    }

    val textFiles = sc.wholeTextFiles("../Wikipedia-En-41784-Articles/*/*")
    val plainText: RDD[(String, String)] = textFiles
      .flatMap { case (_, text) => parse(text.split("\n")) }
      .sample(false, sampleSize)

    val numDocs = plainText.count()
    val bNumDocs = sc.broadcast(numDocs)

    import scala.io.Source
    val stopWords = Source.fromFile("../Data/stopwords.txt").getLines().toSet
    val bStopWords = sc.broadcast(stopWords)

    // ====================== 2. Simple Tokenizer (Exercise Sheet #1) ======
    def plainTextToWords(text: String): Seq[String] = {
      val regex: Regex = "[a-zA-Z0-9]+".r
      val stopwords = bStopWords.value
      regex.findAllIn(text.toLowerCase)
        .filter(token => token.length > 2 && !stopwords.contains(token))
        .toSeq
    }

    val tokenized: RDD[(String, Seq[String])] = plainText.map {
      case (title, contents) => (title, plainTextToWords(contents))
    }

    // ====================== 3. Build TF-IDF Matrix =======================
    val docTermFreqs = tokenized.map { case (title, terms) =>
      val termFreqs = terms.foldLeft(new HashMap[String, Int]()) { (map, term) =>
        map += term -> (map.getOrElse(term, 0) + 1)
        map
      }
      (title, termFreqs)
    }
    docTermFreqs.cache()
    docTermFreqs.count()

    val docIds = docTermFreqs.map(_._1).zipWithUniqueId().map(_.swap).collectAsMap()

    val docFreqs = docTermFreqs.flatMap(_._2.keySet).map((_, 1)).reduceByKey(_ + _, 24)

    val ordering = Ordering.by[(String, Int), Int](_._2)
    val topDocFreqs = docFreqs.top(numTerms)(ordering)

    val idfs = topDocFreqs.map { case (term, count) =>
      (term, math.log(bNumDocs.value.toDouble / count))
    }.toMap

    val idTerms = idfs.keys.zipWithIndex.toMap
    val bIdfs = sc.broadcast(idfs)
    val bIdTerms = sc.broadcast(idTerms)

    val vecs = docTermFreqs.map(_._2).map { termFreqs =>
      val docTotalTerms = termFreqs.values.sum.toDouble
      val termScores = termFreqs.filter { case (term, _) => bIdTerms.value.contains(term) }
        .map { case (term, freq) =>
          (bIdTerms.value(term), bIdfs.value(term) * freq.toDouble / docTotalTerms)
        }
        .toSeq
      Vectors.sparse(bIdTerms.value.size, termScores)
    }
    vecs.cache()
    vecs.count()

    // ====================== 4. Compute SVD ===============================
    val mat = new RowMatrix(vecs)
    val svd = mat.computeSVD(k, computeU = true)

    // ====================== 5. Top-25 Terms & Documents ==================
    def topTermsInTopConcepts(
      svd: SingularValueDecomposition[RowMatrix, Matrix],
      numConcepts: Int,
      numTermsPer: Int
    ): Seq[Seq[(String, Double)]] = {
      val v = svd.V
      val topTerms = new ArrayBuffer[Seq[(String, Double)]]()
      val arr = v.toArray
      for (i <- 0 until numConcepts) {
        val offs = i * v.numRows
        val termWeights = arr.slice(offs, offs + v.numRows).zipWithIndex
        val sorted = termWeights.sortBy(-_._1)
        topTerms += sorted.take(numTermsPer).map { case (score, id) =>
          (bIdTerms.value.find(_._2 == id).getOrElse(("", -1))._1, score)
        }
      }
      topTerms.toSeq
    }

    def topDocsInTopConcepts(
      svd: SingularValueDecomposition[RowMatrix, Matrix],
      numConcepts: Int,
      numDocsPer: Int
    ): Seq[Seq[(String, Double)]] = {
      val u = svd.U
      val topDocs = new ArrayBuffer[Seq[(String, Double)]]()
      for (i <- 0 until numConcepts) {
        val docWeights = u.rows.map(_.toArray(i)).zipWithUniqueId()
        topDocs += docWeights.top(numDocsPer).map { case (score, id) => (docIds(id), score) }
      }
      topDocs.toSeq
    }

    val topConceptTerms = topTermsInTopConcepts(svd, 25, 25)
    val topConceptDocs  = topDocsInTopConcepts(svd, 25, 25)

    // ====================== 6. Final Output (exactly as required) ========
    println("\n=== SIMPLE TOKENIZER - Top 25 Latent Concepts (Problem 1(a)) ===")
    for ((terms, docs) <- topConceptTerms.zip(topConceptDocs)) {
      println(s"Concept terms: ${terms.map(_._1).mkString(", ")}")
      println(s"Concept docs : ${docs.map(_._1).mkString(", ")}")
      println()
    }

    println(s"✅ Simple Tokenizer Version Completed")
    println(s"   Documents processed : $numDocs")
    println(s"   Frequent terms used : $numTerms")
    println(s"   Latent dimensions (k): $k")
    println(s"   Sample size         : $sampleSize")

    // sc.stop()   // Do NOT stop in spark-shell
  }
}