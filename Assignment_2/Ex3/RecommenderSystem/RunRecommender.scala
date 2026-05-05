// Reduce some of the debugging output of Spark
import org.apache.log4j.Logger
import org.apache.log4j.Level
// Import the basic recommender libraries from Spark's MLlib package
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation._

object RunRecommenderObject {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)


    val sampleSize = 1.0 // again use 1 percent sample size for debugging!
    val rawArtistAlias = sc.textFile("../Data/audioscrobbler/artist_alias.txt")
    val rawArtistData = sc.textFile("../Data/audioscrobbler/artist_data.txt")
    val rawUserArtistData = sc.textFile("../Data/audioscrobbler/user_artist_data.txt").sample(false, sampleSize)

    rawUserArtistData.map(_.split(' ')(0).toDouble).stats()
    rawUserArtistData.map(_.split(' ')(1).toDouble).stats()

    val artistByID = rawArtistData.flatMap { 
      line => val (id, name) = line.span(_ != '\t')
      if (name.isEmpty) { 
        None 
      } else { 
        try { 
          Some((id.toInt, name.trim)) } 
        catch { case e: NumberFormatException => None } 
      } 
    }.collectAsMap()

    val artistAlias = rawArtistAlias.flatMap { line => 
      val tokens = line.split('\t') 
      if (tokens(0).isEmpty) { 
        None 
      } else {
        Some((tokens(0).toInt, tokens(1).toInt)) 
      } 
    }.collectAsMap()

    // Some lookups
    artistByID(6803336).head
    artistByID(1000010).head

    // Broadcast the local aliases map since it is going to be part of the closure of our training function
    val bArtistAlias = sc.broadcast(artistAlias)

    // Prepare and cache the training data
    val trainData = rawUserArtistData.map { 
      line => val Array(userID, artistID, count) = 
        line.split(' ').map(_.toInt) 
        val finalArtistID = artistAlias.getOrElse(artistID, artistID) 
        Rating(userID, finalArtistID, count) 
    }.repartition(100).cache() // repartitioning into smaller tasks plus caching reduces broadcasting overhead!

    trainData.first() // show first entry

    // And train the recommender model using Spark's ALS algorithm
    val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0) 

    // Inspect the model: show the first row in the X^T matrix of the resulting factorization
    model.userFeatures.mapValues(_.mkString(", ")).first()

    // Pick the first user from the training data
    val someUser = trainData.map(x => x.user).first()

    // Show actual artists played by some user
    def actualArtistsForUser(someUser: Int): Set[Int] = {
      val rawArtistsForUser = rawUserArtistData.map(_.split(' ')).
        filter { case Array(user,_,_) => user.toInt == someUser }
      rawArtistsForUser.map(x => x(1).toInt).collect().toSet
    }
    actualArtistsForUser(someUser)

    // Print the full artist name (after resolving aliases) for each artist this user listened to
    for (artistID <- actualArtistsForUser(someUser))
      println(artistByID(artistAlias.getOrElse(artistID, artistID)))

    // Next fetch the top 20 recommendations for this user from our model
    val recommendations = model.recommendProducts(someUser, 20)

    // Raw output
    recommendations.foreach(println)

    // And by resolving the aliases and actual artist names
    for (r <- recommendations)
      println(r.user + " -> " + r.product + ": " + artistByID(bArtistAlias.value.getOrElse(r.product, 
        r.product)) + " @ " + r.rating)

    // Compare the results to a baseline model which simply recommends the same most popular artists to each user
    val artistsTotalCount = trainData.map(r => (r.product, r.rating)).reduceByKey(_ + _).collect().sortBy(-_._2)
    def predictMostPopular(user: Int, numArtists: Int) = { 
      val topArtists = artistsTotalCount.take(numArtists)
      topArtists.map{case (artist, rating) => Rating(user, artist, rating)}
    }

    // Use the new prediction function instead of 'model.recommendProducts(someUser, 20)'
    val recommendations2 = predictMostPopular(someUser, 20)

    // Raw output
    recommendations2.foreach(println)

    // And by resolving the aliases and actual artist names
    for (r <- recommendations2)
      println(r.user + " -> " + artistByID(bArtistAlias.value.getOrElse(r.product, 
        r.product)) + " @ " + r.rating)

    // Batch recommendations for multiple users
    val someUsers = trainData.map(x => x.user).distinct().take(10)  
    val someRecommendations = someUsers.map(userID =>   
        model.recommendProducts(userID, 5))
    someRecommendations.map(recs => recs.head.user + " -> " +   
        recs.map(_.product).mkString(", ")).foreach(println)

    // Receiver-Operator-Characteristic (ROC) & Area-Under-the-Curve (AUC) (both averaged over some users)

    import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

    var AUC1 = 0.0
    var AUC2 = 0.0

    for (someUser <- someUsers) {
      
      val actualArtists = actualArtistsForUser(someUser)

      val recommendations1 = model.recommendProducts(someUser, 100)
      val predictionsAndLabels1 = recommendations1.map { 
        case Rating(user, artist, rating) =>
          if (actualArtists.contains(artist)) {
            (rating, 1.0)
          } else {
            (rating, 0.0)
          }
      }

      val metrics1 = new BinaryClassificationMetrics(sc.parallelize(predictionsAndLabels1))
      AUC1 += metrics1.areaUnderROC
      //println(metrics1.areaUnderROC)

      val recommendations2 = predictMostPopular(someUser, 100)
      val predictionsAndLabels2 = recommendations2.map { 
        case Rating(user, artist, rating) =>
          if (actualArtists.contains(artist)) {
            (rating, 1.0)
          } else {
            (rating, 0.0)
          }
      }

      val metrics2 = new BinaryClassificationMetrics(sc.parallelize(predictionsAndLabels2))
      AUC2 += metrics2.areaUnderROC
      //println(metrics2.areaUnderROC)
    }

    println("ALS-Recommender AUC: " + (AUC1/10.0))
    println("Most-Popular AUC:    " + (AUC2/10.0))


    // ====================== Problem 3 (a) - Full Implementation ======================

    println("=== Starting Problem 3 (a) ===")

    // (i) Filter heavy users: users who listened to at least 100 distinct artists
    val userArtistCount = trainData
      .map(r => (r.user, 1))
      .reduceByKey(_ + _)

    val heavyUserIds = userArtistCount
      .filter(_._2 >= 100)
      .map(_._1)
      .collect()
      .toSet

    println(s"Number of heavy users (>=100 artists): ${heavyUserIds.size}")  // Should be ~72,748

    val trainData100 = trainData
      .filter(r => heavyUserIds.contains(r.user))
      .repartition(200)
      .cache()

    // (ii) Per-user 80%/20% stratified split
    // Important: Both trainDataNew and testData contain the SAME users
    val userRatingsRDD = trainData100
      .map(r => (r.user, r))
      .groupByKey()

    val splits = userRatingsRDD.map { case (user, ratings) =>
      val shuffled = scala.util.Random.shuffle(ratings.toList)
      val trainSize = (shuffled.size * 0.8).toInt.max(1)   // at least 1 record per user
      (shuffled.take(trainSize), shuffled.drop(trainSize))
    }

    val trainDataNew = splits.flatMap(_._1).repartition(200).cache()
    val testData     = splits.flatMap(_._2).repartition(200).cache()

    println(s"After split -> Train records: ${trainDataNew.count()}, Test records: ${testData.count()}")

    // Helper function to compute AUC for one user (top-50 recommendations)
    def computeUserAUC(model: MatrixFactorizationModel, user: Int, testRDD: RDD[Rating], topK: Int = 50): Double = {
      val actualArtists = testRDD
        .filter(_.user == user)
        .map(_.product)
        .collect()
        .toSet

      if (actualArtists.isEmpty) return 0.0

      val recommendations = model.recommendProducts(user, topK)

      val predictionsAndLabels = recommendations.map { rec =>
        val label = if (actualArtists.contains(rec.product)) 1.0 else 0.0
        (rec.rating, label)
      }

      if (predictionsAndLabels.isEmpty) 0.0
      else {
        val metrics = new BinaryClassificationMetrics(sc.parallelize(predictionsAndLabels))
        metrics.areaUnderROC
      }
    }

    // Train final model on trainDataNew
    val modelNew = ALS.trainImplicit(trainDataNew, rank = 50, iterations = 10, lambda = 0.01, alpha = 10.0)
    println("New ALS model trained on trainDataNew")

    // Use 1000 test users for evaluation (good balance between accuracy and runtime)
    val numTestUsers = 200
    val testUsers = testData.map(_.user).distinct().collect().take(numTestUsers)
    println(s"Evaluating on ${testUsers.length} test users...")

    var totalAUC = 0.0
    testUsers.foreach { user =>
      totalAUC += computeUserAUC(modelNew, user, testData)
    }

    val avgAUC = totalAUC / testUsers.length
    println(s"Average AUC (ALS, top-50) on test set: $avgAUC")

    // ====================== Save results to log file ======================
    import java.io._
    val logFile = new PrintWriter(new File("Problem3_a.log"))

    logFile.println("=== Problem 3 (a) Results ===")
    logFile.println(s"Timestamp: ${new java.util.Date()}")
    logFile.println(s"Heavy users (>= 100 artists): ${heavyUserIds.size}")
    logFile.println(s"Train records: ${trainDataNew.count()}")
    logFile.println(s"Test records: ${testData.count()}")
    logFile.println(s"Test users used for AUC evaluation: ${testUsers.length}")
    logFile.println(s"Average AUC (ALS model, top-50): $avgAUC")
    logFile.println("=== Problem 3 (a) Completed ===")
    logFile.close()

    println("✅ Results saved to Problem3_a.log")
    println("=== Problem 3 (a) Completed ===")

    // ====================== Problem 3 (b) - Grid Search + 5-fold CV ======================

    println("\n=== Starting Problem 3 (b) - Hyperparameter Tuning ===")

    val ranks   = Array(10, 25, 50)
    val lambdas = Array(1.0, 0.1, 0.01)
    val alphas  = Array(1.0, 10.0, 100.0)

    var bestAUC = 0.0
    var bestRank = 0
    var bestLambda = 0.0
    var bestAlpha = 0.0
    var bestModel: MatrixFactorizationModel = null

    val numFolds = 5
    println(s"Starting grid search with $numFolds-fold CV...")

    for (rank <- ranks; lambda <- lambdas; alpha <- alphas) {
      println(s"\nTesting: rank=$rank, lambda=$lambda, alpha=$alpha")

      var foldAUCs = List[Double]()

      for (fold <- 0 until numFolds) {
        val Array(trainFold, valFold) = trainDataNew.randomSplit(Array(0.8, 0.2), seed = fold.toLong)

        val tmpModel = ALS.trainImplicit(trainFold, rank, 8, lambda, alpha)

        val valUsers = valFold.map(_.user).distinct().collect().take(250)  // 加速CV
        var foldAUC = 0.0
        valUsers.foreach { u =>
          foldAUC += computeUserAUC(tmpModel, u, valFold)
        }
        val avgFold = if (valUsers.nonEmpty) foldAUC / valUsers.length else 0.0
        foldAUCs = avgFold :: foldAUCs
      }

      val avgCVAUC = foldAUCs.sum / numFolds
      println(s"  CV Average AUC = $avgCVAUC")

      if (avgCVAUC > bestAUC) {
        bestAUC = avgCVAUC
        bestRank = rank
        bestLambda = lambda
        bestAlpha = alpha
        println(s"  >>> New Best Parameters!")
      }
    }

    // Train final model with best parameters
    println(s"\nBest parameters found: rank=$bestRank, lambda=$bestLambda, alpha=$bestAlpha (CV AUC=$bestAUC)")

    val finalModel = ALS.trainImplicit(trainDataNew, bestRank, 12, bestLambda, bestAlpha)
    println("Final model trained with best parameters")

    // Final evaluation on testData
    val finalTestUsers = testData.map(_.user).distinct().collect().take(800)
    var finalTotalAUC = 0.0
    finalTestUsers.foreach { u =>
      finalTotalAUC += computeUserAUC(finalModel, u, testData)
    }
    val finalAUC = finalTotalAUC / finalTestUsers.length
    println(s"Final Test AUC: $finalAUC")

    // ====================== Save to log file ======================
    import java.io._
    val pw = new PrintWriter(new File("Problem3_b.log"))
    pw.println("=== Problem 3 (b) Results ===")
    pw.println(s"Timestamp: ${new java.util.Date()}")
    pw.println(s"Best rank: $bestRank")
    pw.println(s"Best lambda: $bestLambda")
    pw.println(s"Best alpha: $bestAlpha")
    pw.println(s"Best CV AUC: $bestAUC")
    pw.println(s"Final Test AUC: $finalAUC")
    pw.println(s"Test users used: ${finalTestUsers.length}")
    pw.println("=== Problem 3 (b) Completed ===")
    pw.close()

    println("✅ Problem 3 (b) results saved to Problem3_b.log")
    println("=== Problem 3 (b) Completed ===")

    // ====================== Problem 3 (c) - New User Recommendation ======================

    println("\n=== Starting Problem 3 (c) - New User Cold Start ===")

    // 1. Create a new user ID 
    val newUserId = 99999999

    // 2. Choose your 8 favorite artists
    val myFavoriteArtistIds = Array(
      1000010,   // The Beatles
      1000002,   // Radiohead
      1000030,   // Pink Floyd
      1000150,   // Nirvana
      1000261,   // Metallica
      1000325,   // Muse
      1000404,   // Linkin Park
      1000764    // Coldplay
    )

    // Give these artists high ratings (implicit feedback: the higher the score, the more you like them).
    val newUserRatings = myFavoriteArtistIds.map { artistId =>
      Rating(newUserId, artistId, 500)   // 500 shows preferrence
    }

    println(s"Added ${newUserRatings.length} ratings for new user $newUserId")

    // 3. Add new users' ratings to the original trainData
    val augmentedTrainData = trainData.union(sc.parallelize(newUserRatings)).repartition(200).cache()

    // 4. Train the new model using the optimal parameters found in (b).
    val newUserModel = ALS.trainImplicit(
      augmentedTrainData, 
      rank = bestRank, 
      iterations = 12, 
      lambda = bestLambda, 
      alpha = bestAlpha
    )

    println("New model trained with new user ratings")

    // 5. Recommend Top 25 Artists for New Users
    val top25Recs = newUserModel.recommendProducts(newUserId, 25)

    println("\n=== Top 25 Recommendations for New User ===")
    top25Recs.foreach { rec =>
      val artistName = artistByID.getOrElse(
        bArtistAlias.value.getOrElse(rec.product, rec.product), 
        s"Unknown Artist (${rec.product})"
      )
      println(f"${rec.product}%8d -> $artistName @ ${rec.rating}%.2f")
    }

    // 6. Save recommendation results to log
    val pw2 = new PrintWriter(new File("Problem3_c.log"))
    pw2.println("=== Problem 3 (c) - New User Recommendations ===")
    pw2.println(s"New User ID: $newUserId")
    pw2.println(s"Number of ratings added: ${newUserRatings.length}")
    pw2.println(s"Used best parameters: rank=$bestRank, lambda=$bestLambda, alpha=$bestAlpha")
    pw2.println("=== Top 25 Recommendations ===")
    top25Recs.foreach { rec =>
      val artistName = artistByID.getOrElse(bArtistAlias.value.getOrElse(rec.product, rec.product), "Unknown")
      pw2.println(f"${rec.product}%8d -> $artistName @ ${rec.rating}%.2f")
    }
    pw2.close()

    println("\n✅ Problem 3 (c) results saved to Problem3_c.log")
    println("=== Problem 3 (c) Completed ===")


  }
}