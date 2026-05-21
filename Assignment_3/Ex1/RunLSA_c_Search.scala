// ========================================================
// RunLSA_c_Search.scala
// Problem 1(c) - NLP Version Search Engine 
// ========================================================

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import scala.collection.mutable.{ArrayBuffer, HashMap}
import org.apache.spark.rdd.RDD
import java.util.Properties
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._
import org.apache.spark.mllib.linalg.{Vectors, Matrix, SingularValueDecomposition, Vector => MLLibVector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg.{DenseMatrix => BDenseMatrix, SparseVector => BSparseVector}

object RunLSA_c_Search {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val sc = SparkContext.getOrCreate(new SparkConf().setAppName("LSA-c-Search-NLP"))

    // ======================  1(b) Best Hyper-Parameter ======================
    val numTerms   = 20000     
    val k          = 100       
    // =========================================================================

    val sampleSize = 1.0

    // ====================== Parsing + NLP ======================
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

    def processPartition(iter: Iterator[(String, String)]): Iterator[(String, Seq[String])] = {
      val pipeline = createNLPPipeline()
      iter.map { case (title, content) => (title, plainTextToLemmas(content, pipeline)) }
    }

    val lemmatized = plainText.mapPartitions(processPartition)

    // ====================== Build LSA ======================
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

    def multiplyByDiagonalRowMatrix(mat: RowMatrix, diag: MLLibVector): RowMatrix = {
      val sArr = diag.toArray
      new RowMatrix(mat.rows.map { vec =>
        val vecArr = vec.toArray
        val newArr = (0 until vec.size).map(i => vecArr(i) * sArr(i)).toArray
        Vectors.dense(newArr)
      })
    }

    val US = multiplyByDiagonalRowMatrix(svd.U, svd.s)

    // ====================== Search ======================
    def termsToQueryVector(terms: Seq[String]): BSparseVector[Double] = {
      val filtered = terms.filter(idTerms.contains)
      val indices = filtered.map(idTerms(_)).toArray
      val values = filtered.map(idfs(_)).toArray
      new BSparseVector[Double](indices, values, idTerms.size)
    }

    def topDocsForQuery(queryTerms: Seq[String], topN: Int = 5): Seq[(String, Double)] = {
      val queryVec = termsToQueryVector(queryTerms)
      val breezeV = new BDenseMatrix[Double](svd.V.numRows, svd.V.numCols, svd.V.toArray)
      val termRowArr = (breezeV.t * queryVec).toArray
      val termRowVec = org.apache.spark.mllib.linalg.Matrices.dense(termRowArr.length, 1, termRowArr)
      val docScores = US.multiply(termRowVec)
      val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId()
      allDocWeights.top(topN).map { case (score, id) => (docIds(id), score) }
    }

    val queries = List(
      List("computer", "algorithm", "software"),
      List("war", "battle", "military"),
      List("music", "album", "song"),
      List("movie", "film", "director", "actor"),
      List("space", "planet", "nasa"),
      List("artificial", "intelligence"),
      List("climate", "global", "warming"),
      List("economy", "market", "finance"),
      List("olympic", "game", "medal"),
      List("mythology", "norse", "greek")
    )

    println("\n=== NLP Search Engine Results (Top 5) ===")
    for (q <- queries) {
      val results = topDocsForQuery(q, 5)
      println(s"\nQuery: ${q.mkString(" ")}")
      results.foreach { case (title, score) =>
        println(f"  $title%-70s score = $score%.4f")
      }
    }

    //sc.stop()
  }
}