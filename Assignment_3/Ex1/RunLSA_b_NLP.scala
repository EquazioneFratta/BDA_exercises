// ========================================================
// RunLSA_b_NLP.scala
// Problem 1(b) - NLP Version Full Experiment
// Runs all 9 hyper-parameter combinations and records runtime
// Fixed serialization issue with named function
// ========================================================

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import scala.collection.mutable.{ArrayBuffer, HashMap}
import org.apache.spark.rdd.RDD
import java.util.Properties
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._
import org.apache.spark.mllib.linalg.{Vectors, Matrix, SingularValueDecomposition}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

object RunLSA_b_NLP {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setAppName("LSA-b-NLP-Full")
      .set("spark.local.dir", "/tmp/spark-" + System.getProperty("user.name"))
      .set("spark.shuffle.partitions", "120")

    val sc = SparkContext.getOrCreate(conf)

    val sampleSize = 1.0

    // ====================== HYPER-PARAMETERS (Problem 1(b)) ======================
    val numFreqList = Array(5000, 10000, 20000)
    val kList       = Array(25, 100, 250)
    // =============================================================================

    println("=== Problem 1(b) NLP Full Experiment Started ===")
    println(s"Total configurations: ${numFreqList.length * kList.length} | Dataset size: $sampleSize\n")

    // ====================== Parsing ======================
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
    val plainText = textFiles.flatMap { case (_, text) => parse(text.split("\n")) }.sample(false, sampleSize)

    val numDocs = plainText.count()
    val bNumDocs = sc.broadcast(numDocs)

    import scala.io.Source
    val stopWords = Source.fromFile("../Data/stopwords.txt").getLines().toSet
    val bStopWords = sc.broadcast(stopWords)

    // ====================== NLP Pipeline ======================
    def createNLPPipeline(): StanfordCoreNLP = {
      val props = new Properties()
      props.put("annotators", "tokenize, ssplit, pos, lemma")
      new StanfordCoreNLP(props)
    }

    def plainTextToLemmas(text: String, pipeline: StanfordCoreNLP): Seq[String] = {
      val doc = new Annotation(text)
      pipeline.annotate(doc)
      val lemmas = new ArrayBuffer[String]()
      val tokens = doc.get(classOf[TokensAnnotation])
      val stopwords = bStopWords.value
      tokens.forEach { token =>
        val lemma = token.lemma.toLowerCase
        if (lemma.length > 2 && lemma.forall(Character.isLetter) && !stopwords.contains(lemma))
          lemmas += lemma
      }
      lemmas.toSeq
    }

    // Fixed serialization issue with named function
    def processPartition(iter: Iterator[(String, String)]): Iterator[(String, Seq[String])] = {
      val pipeline = createNLPPipeline()
      iter.map { case (title, content) => (title, plainTextToLemmas(content, pipeline)) }
    }

    val lemmatized: RDD[(String, Seq[String])] = plainText.mapPartitions(processPartition)

    // ====================== Experiment Loop ======================
    for (numTerms <- numFreqList; k <- kList) {
      val startTime = System.currentTimeMillis()

      println(s">>> Running NLP | numFreq = $numTerms | k = $k")

      val docTermFreqs = lemmatized.map { case (title, terms) =>
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

      val mat = new RowMatrix(vecs)
      val svd = mat.computeSVD(k, computeU = true)

      val duration = (System.currentTimeMillis() - startTime) / 1000.0
      println(s"✅ Finished NLP: numFreq=$numTerms | k=$k | Runtime = $duration seconds\n")
    }

    println("=== Problem 1(b) NLP Full Experiment Completed! ===")
    sc.stop()
  }
}