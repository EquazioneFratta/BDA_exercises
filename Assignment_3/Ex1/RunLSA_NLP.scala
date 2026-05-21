// ========================================================
// RunLSA_NLP.scala
// Problem 1(a) - NLP Version 
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

object RunLSA_NLP {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val sc = SparkContext.getOrCreate(new SparkConf().setAppName("RunLSA_NLP"))

    val sampleSize = 1.0      
    val numTerms   = 5000
    val k          = 25

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
    val plainText: RDD[(String, String)] = textFiles
      .flatMap { case (_, text) => parse(text.split("\n")) }
      .sample(false, sampleSize)

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

    def lemmatizePartition(it: Iterator[(String, String)]): Iterator[(String, Seq[String])] = {
      val pipeline = createNLPPipeline()
      it.map { case (title, contents) =>
        (title, plainTextToLemmas(contents, pipeline))
      }
    }

    val lemmatized: RDD[(String, Seq[String])] = plainText.mapPartitions(lemmatizePartition)

    // ====================== TF-IDF + SVD ============================
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

    // ====================== Top-25 Results ===========================
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

    println("\n=== NLP (LEMMATIZER) - Top 25 Latent Concepts (Problem 1(a)) ===")
    for ((terms, docs) <- topConceptTerms.zip(topConceptDocs)) {
      println(s"Concept terms: ${terms.map(_._1).mkString(", ")}")
      println(s"Concept docs : ${docs.map(_._1).mkString(", ")}")
      println()
    }

    println(s"✅ NLP Version Completed")
    println(s"   Documents processed : $numDocs")
    println(s"   Frequent terms      : $numTerms")
    println(s"   Latent dimensions k : $k")
    println(s"   Sample size         : $sampleSize")
  }
}