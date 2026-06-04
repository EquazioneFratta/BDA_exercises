// ============================================================
// RunMonteCarlo - Spark Shell Version
// Run with: :load RunMonteCarlo.scala
// ============================================================

import java.io.File
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import scala.collection.mutable.ArrayBuffer
import org.apache.commons.math3.stat.correlation.Covariance
import org.apache.commons.math3.random.MersenneTwister
import org.apache.commons.math3.distribution.{MultivariateNormalDistribution, NormalDistribution}
import org.apache.spark.sql.Dataset

sc.setLogLevel("ERROR")

// -------- Data Directory (relative path) --------
val dataDirectory = "."

// -------- Parse CSV in StockAnalysis.com Format --------
// Columns: Date, Open, High, Low, Close, Adj. Close, Change, Volume
// Date format: yyyy-MM-dd
def readStockAnalysisHistory(file: File): Array[(LocalDate, Double)] = {
  val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd")
  val lines = scala.io.Source.fromFile(file).getLines().toSeq
  lines.tail.flatMap { line =>
    try {
      val cols = line.split(',')
      val date  = LocalDate.parse(cols(0).trim, formatter)
      val value = cols(4).trim.toDouble  // Close price
      Some((date, value))
    } catch {
      case e: Exception => None
    }
  }.toArray
}

// -------- Trim to Date Range --------
def trimToRegion(
    history: Array[(LocalDate, Double)],
    start: LocalDate,
    end: LocalDate): Array[(LocalDate, Double)] = {
  var trimmed = history
    .dropWhile(_._1.isBefore(start))
    .takeWhile(x => x._1.isBefore(end) || x._1.isEqual(end))
  if (trimmed.isEmpty) return trimmed
  if (trimmed.head._1 != start)
    trimmed = Array((start, trimmed.head._2)) ++ trimmed
  if (trimmed.last._1 != end)
    trimmed = trimmed ++ Array((end, trimmed.last._2))
  trimmed
}

// -------- Fill Missing Trading Days --------
def fillInHistory(
    history: Array[(LocalDate, Double)],
    start: LocalDate,
    end: LocalDate): Array[(LocalDate, Double)] = {
  var cur = history
  val filled = new ArrayBuffer[(LocalDate, Double)]()
  var curDate = start
  while (curDate.isBefore(end)) {
    if (cur.tail.nonEmpty && cur.tail.head._1 == curDate)
      cur = cur.tail
    filled += ((curDate, cur.head._2))
    curDate = curDate.plusDays(1)
    if (curDate.getDayOfWeek.getValue > 5)
      curDate = curDate.plusDays(2)
  }
  filled.toArray
}

// -------- Read Data --------
val start = LocalDate.of(2021, 6, 3)
val end   = LocalDate.of(2026, 6, 3)

println("READING STOCKS: " + dataDirectory + "/stocks/")
val stocksDir = new File(dataDirectory + "/stocks/")
val stockFiles = stocksDir.listFiles().sortBy(_.getName)
val stocks = stockFiles.flatMap { file =>
  try {
    val raw = readStockAnalysisHistory(file)
    val trimmed = trimToRegion(raw, start, end)
    val filled = fillInHistory(trimmed, start, end)
    println(s"  Loaded: ${file.getName} (${filled.length} days)")
    Some(filled)
  } catch { case e: Exception => println(s"  Failed: ${file.getName}"); None }
}

println("READING FACTORS: " + dataDirectory + "/factors/")
val factorsDir = new File(dataDirectory + "/factors/")
val factorFiles = factorsDir.listFiles().sortBy(_.getName)
val factors = factorFiles.flatMap { file =>
  try {
    val raw = readStockAnalysisHistory(file)
    val trimmed = trimToRegion(raw, start, end)
    val filled = fillInHistory(trimmed, start, end)
    println(s"  Loaded: ${file.getName} (${filled.length} days)")
    Some(filled)
  } catch { case e: Exception => println(s"  Failed: ${file.getName}"); None }
}

println(s"DONE: ${stocks.length} stocks, ${factors.length} factors loaded.")

// -------- Two-Week Returns --------
def twoWeekReturns(history: Array[(LocalDate, Double)]): Array[Double] = {
  history.sliding(10).map { window =>
    val next = window.last._2
    val prev = window.head._2
    (next - prev) / prev
  }.toArray
}

val stockReturns  = stocks.map(twoWeekReturns).toSeq
val factorReturns = factors.map(twoWeekReturns).toSeq

println(s"${stockReturns.length} STOCK RETURNS, ${factorReturns.length} FACTOR RETURNS calculated.")
println(s"Each stock return series length: ${stockReturns.head.length}")
println(s"Each factor return series length: ${factorReturns.head.length}")

// -------- Factor Matrix: rows = time steps, cols = factors --------
def factorMatrix(histories: Seq[Array[Double]]): Array[Array[Double]] = {
  val mat = new Array[Array[Double]](histories.head.length)
  for (i <- histories.head.indices)
    mat(i) = histories.map(_(i)).toArray
  mat
}

val factorMat = factorMatrix(factorReturns)
println(s"factorMat: ${factorMat.length} rows x ${factorMat(0).length} cols")

// -------- Factor Statistics --------
val factorMeans = factorReturns.map(f => f.sum / f.length).toArray
val factorCov   = new Covariance(factorMat).getCovarianceMatrix().getData()
val factorVars  = factorCov.zipWithIndex.map { case (row, i) => row(i) }

// -------- Ridge Regression (solves SingularMatrixException) --------
// Adds small lambda to diagonal of X'X to ensure invertibility
def ridgeRegression(y: Array[Double], X: Array[Array[Double]], lambda: Double = 1e-6): Array[Double] = {
  val n = X.length       // number of observations
  val p = X(0).length    // number of features

  // Build X'X + lambda*I  and  X'y
  val XtX = Array.ofDim[Double](p + 1, p + 1)
  val Xty = Array.ofDim[Double](p + 1)

  // intercept column (column 0 = 1.0)
  for (i <- 0 until n) {
    Xty(0) += y(i)
    for (j <- 0 until p) {
      XtX(0)(j + 1) += X(i)(j)
      XtX(j + 1)(0) += X(i)(j)
      Xty(j + 1)    += X(i)(j) * y(i)
      for (k <- 0 until p)
        XtX(j + 1)(k + 1) += X(i)(j) * X(i)(k)
    }
    XtX(0)(0) += 1.0
  }

  // Add ridge penalty to diagonal (skip intercept at index 0)
  for (j <- 1 to p)
    XtX(j)(j) += lambda * n

  // Solve via Gaussian elimination with partial pivoting
  val aug = XtX.map(row => row ++ Array(Xty(row.indexOf(row.last))))  // won't work cleanly
  // Use simple Cholesky-free approach: org.apache.commons.math3 LU decomposition
  import org.apache.commons.math3.linear.{Array2DRowRealMatrix, ArrayRealVector, LUDecomposition}
  val matA = new Array2DRowRealMatrix(XtX)
  val vecB = new ArrayRealVector(Xty)
  val solver = new LUDecomposition(matA).getSolver
  solver.solve(vecB).toArray
}

def computeFactorWeightsRidge(
    stocksReturns:  Seq[Array[Double]],
    factorMatrix:   Array[Array[Double]],
    lambda: Double = 1e-4): Array[Array[Double]] = {
  stocksReturns.map { stockReturn =>
    ridgeRegression(stockReturn, factorMatrix, lambda)
  }.toArray
}

val factorWeights = computeFactorWeightsRidge(stockReturns, factorMat)
println(s"Factor weights computed. Shape: ${factorWeights.length} stocks x ${factorWeights(0).length} params")

// -------- Trial Return Calculation --------
def instrumentTrialReturn(instrument: Array[Double], trial: Array[Double]): Double = {
  var result = instrument(0)  // intercept
  var i = 0
  while (i < trial.length) {
    result += trial(i) * instrument(i + 1)
    i += 1
  }
  result
}

def trialReturn(trial: Array[Double], instruments: Seq[Array[Double]]): Double = {
  instruments.map(instrumentTrialReturn(_, trial)).sum / instruments.size
}

// -------- (c) & (d): Correlated Factors via Multivariate Normal --------
def trialReturns(
    seed:        Long,
    numTrials:   Int,
    instruments: Seq[Array[Double]],
    means:       Array[Double],
    covMatrix:   Array[Array[Double]]): Seq[Double] = {
  val rand  = new MersenneTwister(seed)
  val mvn   = new MultivariateNormalDistribution(rand, means, covMatrix)
  val results = new Array[Double](numTrials)
  for (i <- 0 until numTrials) {
    val sample   = mvn.sample()
    results(i)   = trialReturn(sample, instruments)
  }
  results
}

// -------- (e): Independent Factors --------
def trialReturnsIndependent(
    seed:        Long,
    numTrials:   Int,
    instruments: Seq[Array[Double]],
    means:       Array[Double],
    variances:   Array[Double]): Seq[Double] = {
  val rand  = new MersenneTwister(seed)
  val dists = means.zip(variances).map { case (m, v) =>
    new NormalDistribution(rand, m, math.sqrt(math.max(v, 1e-10)))
  }
  val results = new Array[Double](numTrials)
  for (i <- 0 until numTrials) {
    val sample = dists.map(_.sample())
    results(i) = trialReturn(sample, instruments)
  }
  results
}

// -------- VaR & CVaR --------
def fivePercentVaR(trialsDS: Dataset[Double]): Double = {
  trialsDS.stat.approxQuantile("value", Array(0.05), 0.0).head
}

def fivePercentCVaR(trialsDS: Dataset[Double]): Double = {
  val n = trialsDS.count().toInt
  val topLosses = trialsDS.orderBy("value").limit(math.max(n / 20, 1))
  topLosses.agg("value" -> "avg").first()(0).asInstanceOf[Double]
}

// -------- Simulation Parameters --------
val numTrials   = 1000000
val parallelism = 100
val baseSeed    = 1001L
val seeds       = (baseSeed until baseSeed + parallelism)

import spark.implicits._
val seedDS = seeds.toDS().repartition(parallelism)

// ============================================================
// (c) Portfolio VaR & CVaR — all 3 stocks, correlated factors
// ============================================================
println("\n========== (c) PORTFOLIO VaR & CVaR (Correlated Factors) ==========")
val t0 = System.currentTimeMillis()

val portfolioTrialsDS = seedDS.flatMap(
  trialReturns(_, numTrials / parallelism, factorWeights, factorMeans, factorCov)
)
portfolioTrialsDS.cache()
val portfolioCount = portfolioTrialsDS.count()
val portfolioVaR   = fivePercentVaR(portfolioTrialsDS)
val portfolioCVaR  = fivePercentCVaR(portfolioTrialsDS)
val t1 = System.currentTimeMillis()

println(f"Trials:        $portfolioCount")
println(f"Portfolio VaR  (5%%): $portfolioVaR%.6f")
println(f"Portfolio CVaR (5%%): $portfolioCVaR%.6f")
println(f"Runtime: ${(t1 - t0) / 1000.0}%.1f seconds")

// ============================================================
// (d) Individual Stock VaR & CVaR
// ============================================================
println("\n========== (d) INDIVIDUAL STOCK VaR & CVaR ==========")
val stockNames = stockFiles.map(_.getName.split("-").head)

stockNames.zipWithIndex.foreach { case (name, idx) =>
  val t2 = System.currentTimeMillis()
  val singleWeights = Seq(factorWeights(idx))
  val singleDS = seedDS.flatMap(
    trialReturns(_, numTrials / parallelism, singleWeights, factorMeans, factorCov)
  )
  singleDS.cache()
  val sVaR  = fivePercentVaR(singleDS)
  val sCVaR = fivePercentCVaR(singleDS)
  val t3 = System.currentTimeMillis()
  println(f"[$name] VaR (5%%): $sVaR%.6f  |  CVaR (5%%): $sCVaR%.6f  |  Runtime: ${(t3-t2)/1000.0}%.1f s")
  singleDS.unpersist()
}

// ============================================================
// (e) Portfolio VaR & CVaR — Independent Factors
// ============================================================
println("\n========== (e) PORTFOLIO VaR & CVaR (Independent Factors) ==========")
val t4 = System.currentTimeMillis()

val indepDS = seedDS.flatMap(
  trialReturnsIndependent(_, numTrials / parallelism, factorWeights, factorMeans, factorVars)
)
indepDS.cache()
val indepCount = indepDS.count()
val indepVaR   = fivePercentVaR(indepDS)
val indepCVaR  = fivePercentCVaR(indepDS)
val t5 = System.currentTimeMillis()

println(f"Trials:       $indepCount")
println(f"Indep VaR  (5%%): $indepVaR%.6f")
println(f"Indep CVaR (5%%): $indepCVaR%.6f")
println(f"Runtime: ${(t5 - t4) / 1000.0}%.1f seconds")

// ============================================================
// Summary
// ============================================================
println("\n========== SUMMARY ==========")
println(f"(c) Correlated  VaR: $portfolioVaR%.6f  CVaR: $portfolioCVaR%.6f")
println(f"(e) Independent VaR: $indepVaR%.6f  CVaR: $indepCVaR%.6f")
println("Done.")