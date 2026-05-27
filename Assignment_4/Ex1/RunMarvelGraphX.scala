import org.apache.log4j.{Logger, Level}
Logger.getLogger("org").setLevel(Level.ERROR)
Logger.getLogger("akka").setLevel(Level.ERROR)

import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._

val DATA_PATH = "."

def time[R](label: String)(block: => R): R = {
  val t0 = System.currentTimeMillis()
  val result = block
  val t1 = System.currentTimeMillis()
  println(f"[time] $label: ${(t1 - t0) / 1000.0}%.2f s")
  result
}

// =============================================================================
// (a) Load & Parse CSV files --> Vertex RDD + Edge RDD --> Build Graph
// =============================================================================

val (edgesParsed, marvelGraph, allVertexNames, nodesParsed) = time("(a) Graph construction") {

  val edgesRaw = sc.textFile(DATA_PATH + "/edges.csv")
  val edgesHeader = edgesRaw.first()
  val edgesParsed = edgesRaw.
    filter(_ != edgesHeader).
    map(line => { val c = line.split(",", -1); (c(0).trim, c(1).trim) })
  edgesParsed.cache()

  val heroNames  = edgesParsed.map(_._1).distinct()
  val comicNames = edgesParsed.map(_._2).distinct()
  val allVertexNames = heroNames.union(comicNames).distinct()

  val vertexNameToId = allVertexNames.zipWithUniqueId().collectAsMap().toMap
  val bVertexNameToId = sc.broadcast(vertexNameToId)

  val vertexRDD: RDD[(VertexId, String)] = allVertexNames.
    map(name => (bVertexNameToId.value(name), name))
  val edgeRDD: RDD[Edge[Int]] = edgesParsed.
    map { case (hero, comic) =>
      Edge(bVertexNameToId.value(hero), bVertexNameToId.value(comic), 1)
    }

  val marvelGraph = Graph(vertexRDD, edgeRDD)
  marvelGraph.cache()

  val nodesRaw = sc.textFile(DATA_PATH + "/nodes.csv")
  val nodesHeader = nodesRaw.first()
  val nodesParsed = nodesRaw.
    filter(_ != nodesHeader).
    map(line => { val c = line.split(",", -1); (c(0).trim, c(1).trim) })

  println(s"[a] Number of (hero, comic) edges : ${edgesParsed.count()}")
  println(s"[a] Distinct heroes  from edges.csv: ${heroNames.count()}")
  println(s"[a] Distinct comics  from edges.csv: ${comicNames.count()}")
  println(s"[a] Distinct vertices from edges.csv: ${allVertexNames.count()}")
  println(s"[a] Graph vertices : ${marvelGraph.vertices.count()}")
  println(s"[a] Graph edges    : ${marvelGraph.edges.count()}")

  val nodesFromFile = nodesParsed.map(_._1).distinct()
  val onlyInNodes = nodesFromFile.subtract(allVertexNames)
  val onlyInEdges = allVertexNames.subtract(nodesFromFile)
  println(s"[a] Vertices in nodes.csv only: ${onlyInNodes.count()}")
  println(s"[a] Vertices in edges.csv only: ${onlyInEdges.count()}")
  println("[a] Sample vertices only in nodes.csv:") ; onlyInNodes.take(10).foreach(println)
  println("[a] Sample vertices only in edges.csv:") ; onlyInEdges.take(10).foreach(println)

  (edgesParsed, marvelGraph, allVertexNames, nodesParsed)
}

// =============================================================================
// (b) Connected Components
// =============================================================================

val componentSizes = time("(b) Connected components") {
  val ccGraph = marvelGraph.connectedComponents()
  val componentSizes = ccGraph.vertices.map(_._2).countByValue()
  println(s"[b] Number of connected components: ${componentSizes.size}")
  println("[b] Top-10 component sizes (largest first):")
  componentSizes.toSeq.sortBy(-_._2).take(10).foreach { case (cid, sz) =>
    println(s"    Component id=$cid  size=$sz")
  }
  componentSizes
}

// =============================================================================
// (c) Degree Distribution, Clustering Coefficient, Average Path Length
// =============================================================================

time("(c) Degree distribution") {
  val degrees = marvelGraph.degrees.cache()
  println(s"[c] Degree stats: ${degrees.map(_._2).stats()}")
  val topDegrees = degrees.innerJoin(marvelGraph.vertices) { (_, deg, name) => (name, deg) }
  val topDegOrd = Ordering.by[(String, Int), Int](_._2)
  println("[c] Top-10 vertices by degree:")
  topDegrees.map(_._2).top(10)(topDegOrd).foreach(println)
}

val avgCC = time("(c) Clustering coefficient") {
  val reverseEdges = marvelGraph.edges.map(e => Edge(e.dstId, e.srcId, e.attr))
  val undirectedEdges = marvelGraph.edges.union(reverseEdges)
  val undirectedGraph = Graph(marvelGraph.vertices, undirectedEdges).cache()
  val triGraph = undirectedGraph.triangleCount()
  triGraph.cache()
  val maxTriGraph = undirectedGraph.degrees.mapValues(d => d * (d - 1) / 2.0)
  val clusterCoeffRDD = triGraph.vertices.
    innerJoin(maxTriGraph) { (_, triCount, maxTris) =>
      if (maxTris == 0.0) 0.0 else triCount / maxTris
    }
  val avgCC = clusterCoeffRDD.map(_._2).sum() / marvelGraph.vertices.count()
  println(f"[c] Average clustering coefficient: $avgCC%.6f")
  avgCC
}

val undirectedGraph = {
  val reverseEdges = marvelGraph.edges.map(e => Edge(e.dstId, e.srcId, e.attr))
  Graph(marvelGraph.vertices, marvelGraph.edges.union(reverseEdges)).cache()
}

time("(c) Average path length (Pregel SSSP)") {
  def mergeMaps(m1: Map[VertexId, Int], m2: Map[VertexId, Int]): Map[VertexId, Int] = {
    def minVal(k: VertexId) =
      math.min(m1.getOrElse(k, Int.MaxValue), m2.getOrElse(k, Int.MaxValue))
    (m1.keySet ++ m2.keySet).map(k => k -> minVal(k)).toMap
  }
  def updateState(id: VertexId, state: Map[VertexId, Int],
                  msg: Map[VertexId, Int]): Map[VertexId, Int] = mergeMaps(state, msg)
  def sendMessages(e: EdgeTriplet[Map[VertexId, Int], _])
      : Iterator[(VertexId, Map[VertexId, Int])] = {
    def checkInc(a: Map[VertexId, Int], b: Map[VertexId, Int],
                 bid: VertexId): Iterator[(VertexId, Map[VertexId, Int])] = {
      val aPlus = a.map { case (v, d) => v -> (d + 1) }
      if (b != mergeMaps(aPlus, b)) Iterator((bid, aPlus)) else Iterator.empty
    }
    checkInc(e.srcAttr, e.dstAttr, e.dstId) ++ checkInc(e.dstAttr, e.srcAttr, e.srcId)
  }
  val sampleIds = marvelGraph.vertices.map(_._1).sample(false, 0.01).collect().toSet
  println(s"[c] Running SSSP from ${sampleIds.size} sampled source vertices ...")
  val mapGraph = undirectedGraph.
    mapVertices((id, _) =>
      if (sampleIds.contains(id)) Map(id -> 0) else Map.empty[VertexId, Int]
    )
  val ssspResult = mapGraph.pregel(Map.empty[VertexId, Int])(updateState, sendMessages, mergeMaps)
  val paths = ssspResult.vertices.
    flatMap { case (id, m) =>
      m.map { case (k, v) => if (id < k) (id, k, v) else (k, id, v) }
    }.distinct()
  paths.cache()
  val pathStats = paths.map(_._3).filter(_ > 0).stats()
  println(s"[c] Path-length stats (sampled): $pathStats")
  println(f"[c] Estimated average path length: ${pathStats.mean}%.4f")
  val pathHist = paths.map(_._3).countByValue()
  println("[c] Path-length histogram:")
  pathHist.toSeq.sorted.foreach { case (len, cnt) => println(s"    length=$len  pairs=$cnt") }
}

// =============================================================================
// (d) PageRank – top-25 heroes and top-25 comics
// =============================================================================

time("(d) PageRank") {
  val heroSet  = nodesParsed.filter(_._2 == "hero" ).map(_._1).collect().toSet
  val comicSet = nodesParsed.filter(_._2 == "comic").map(_._1).collect().toSet
  val bHeroSet  = sc.broadcast(heroSet)
  val bComicSet = sc.broadcast(comicSet)

  val prGraph = marvelGraph.pageRank(0.001, 0.15)
  val namesAndRanks = prGraph.vertices.
    innerJoin(marvelGraph.vertices) { (_, rank, name) => (name, rank) }.
    map(_._2)
  namesAndRanks.cache()

  val prOrd = Ordering.by[(String, Double), Double](_._2)
  val top25HeroesPR = namesAndRanks.filter { case (n, _) => bHeroSet.value.contains(n) }.top(25)(prOrd)
  val top25ComicsPR = namesAndRanks.filter { case (n, _) => bComicSet.value.contains(n) }.top(25)(prOrd)

  println("[d] Top-25 heroes by PageRank:")
  top25HeroesPR.foreach { case (name, rank) => println(f"    $name%-40s PR=$rank%.4f") }
  println("[d] Top-25 comics by PageRank:")
  top25ComicsPR.foreach { case (name, rank) => println(f"    $name%-40s PR=$rank%.4f") }

  val inDeg  = marvelGraph.inDegrees
  val outDeg = marvelGraph.outDegrees
  val totalDeg = inDeg.fullOuterJoin(outDeg).
    map { case (vid, (in, out)) => (vid, in.getOrElse(0) + out.getOrElse(0)) }
  val namesAndDeg = totalDeg.join(marvelGraph.vertices).map { case (_, (deg, name)) => (name, deg) }
  namesAndDeg.cache()

  val degOrd = Ordering.by[(String, Int), Int](_._2)
  val top25HeroesDeg = namesAndDeg.filter { case (n, _) => bHeroSet.value.contains(n) }.top(25)(degOrd)
  val top25ComicsDeg = namesAndDeg.filter { case (n, _) => bComicSet.value.contains(n) }.top(25)(degOrd)

  println("[d] Top-25 heroes by degree:")
  top25HeroesDeg.foreach { case (name, deg) => println(f"    $name%-40s degree=$deg") }
  println("[d] Top-25 comics by degree:")
  top25ComicsDeg.foreach { case (name, deg) => println(f"    $name%-40s degree=$deg") }

  val overlapHeroes = top25HeroesPR.map(_._1).toSet & top25HeroesDeg.map(_._1).toSet
  val overlapComics = top25ComicsPR.map(_._1).toSet & top25ComicsDeg.map(_._1).toSet
  println(s"[d] Heroes in both top-25 lists: ${overlapHeroes.size}/25")
  println(s"[d] Comics  in both top-25 lists: ${overlapComics.size}/25")
}

// =============================================================================
// (e) Hero co-occurrence pairs vs hero-network.csv
// =============================================================================

time("(e) Hero co-occurrence pairs") {
  val heroesByComic = edgesParsed.
    map { case (hero, comic) => (comic, hero) }.
    groupByKey().
    mapValues(_.toSeq.sorted)
  val heroPairsFromEdges = heroesByComic.
    flatMap { case (_, heroes) => heroes.combinations(2).map(p => (p(0), p(1))) }.
    distinct()
  heroPairsFromEdges.cache()
  println(s"[e] Hero co-occurrence pairs from edges.csv: ${heroPairsFromEdges.count()}")

  val heroEdgeRaw = sc.textFile(DATA_PATH + "/hero-network.csv")
  val heroEdgeHeader = heroEdgeRaw.first()
  val heroPairsFromFile = heroEdgeRaw.
    filter(_ != heroEdgeHeader).
    filter(line => line.split(",", -1).length >= 2).
    map(line => { val c = line.split(",", -1); val a = c(0).trim.replaceAll("\"", ""); val b = c(1).trim.replaceAll("\"", ""); if (a < b) (a, b) else (b, a) }).
    distinct()
  heroPairsFromFile.cache()
  println(s"[e] Hero pairs in hero-network.csv         : ${heroPairsFromFile.count()}")

  val onlyInDerived = heroPairsFromEdges.subtract(heroPairsFromFile)
  val onlyInFile    = heroPairsFromFile.subtract(heroPairsFromEdges)
  println(s"[e] Pairs only in derived (edges.csv) set  : ${onlyInDerived.count()}")
  println(s"[e] Pairs only in hero-network.csv         : ${onlyInFile.count()}")
  println("[e] Sample pairs only in derived set:") ; onlyInDerived.take(5).foreach(println)
  println("[e] Sample pairs only in hero-network.csv:") ; onlyInFile.take(5).foreach(println)
}

println("\n=== Problem 1 complete ===")
