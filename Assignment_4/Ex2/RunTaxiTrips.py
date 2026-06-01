# ==============================================================================
# RunTaxiTrips.py
# Big Data Analytics – Exercise Sheet #4, Problem 2
# Geo-Spatial & Temporal Data Analysis
# ==============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# LOCAL_TEST FLAG
#   True  → local[*] SparkSession, small data sample, no cluster config
#   False → no master (provided by spark-submit), full dataset
# ──────────────────────────────────────────────────────────────────────────────
LOCAL_TEST = False

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import json
import logging
import math
import time
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.statcounter import StatCounter

# Sedona imports (packages injected by spark-submit --packages)
from sedona.spark import SedonaContext

# Silence all noisy loggers (Py4J, PySpark internals, Sedona) at Python level
for _noisy in ("py4j", "pyspark", "sedona", "org.apache", "org.spark"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


# ==============================================================================
# REPORT BUFFER
# All task result lines go through report.add() – they print to stdout AND
# accumulate in memory.  At the end, report.save() flushes only those clean
# lines (no Spark/JVM warnings) to the output file.
# ==============================================================================

class Report:
    def __init__(self):
        self._lines: list[str] = []

    def add(self, text: str = "") -> None:
        """Print a line to stdout and buffer it for the output file."""
        print(text)
        self._lines.append(text)

    def save(self, path: str, runtime_secs: float) -> None:
        footer = (
            "\n" + "=" * 70 + "\n"
            f"Total wall-clock time: {runtime_secs:.1f}s "
            f"({runtime_secs / 60:.1f} min)\n"
            + "=" * 70 + "\n"
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self._lines))
            fh.write(footer)
        print(f"\n[INFO] Clean results written to: {path}")

# ──────────────────────────────────────────────────────────────────────────────
# DATA PATHS
# ──────────────────────────────────────────────────────────────────────────────
BOROUGHS_GEOJSON_PATH = "data/nyc-borough-boundaries-polygon.geojson"
TAXI_CSV_PATH         = "data/trip_data-part*.csv"
#TAXI_CSV_PATH         = "data/trip_data_sample.csv"
# Sample fraction used in LOCAL_TEST mode
LOCAL_SAMPLE_FRACTION = 0.005   # 0.5 % of data – adjust as needed
LOCAL_SAMPLE_SEED     = 42


# ==============================================================================
# SPARK SESSION SETUP
# ==============================================================================
def build_spark_session() -> SparkSession:
    """Create and return a SparkSession (local or cluster depending on flag)."""
    builder = (
        SparkSession.builder
        .appName("BDA-Ex4-TaxiTrips")
        # Sedona serialiser – must match the packages injected via spark-submit
        .config("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryo.registrator",
                "org.apache.sedona.core.serde.SedonaKryoRegistrator")
    )
    if LOCAL_TEST:
        builder = builder.master("local[*]")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ==============================================================================
# HELPER UTILITIES
# ==============================================================================

def haversine_miles(lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
    """
    Return the great-circle distance in miles between two (lat, lon) points.
    Returns 0.0 when either coordinate is obviously invalid (0,0) or when
    the two points are identical (distance would be 0 → division by zero).
    """
    if (lat1 == 0.0 and lon1 == 0.0) or (lat2 == 0.0 and lon2 == 0.0):
        return 0.0
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi        = math.radians(lat2 - lat1)
    dlambda     = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_datetime(s: str):
    """Parse '2013-01-01 15:11:48' → datetime.  Returns None on failure."""
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return None


def statcounter_mean(sc: StatCounter) -> float:
    """Safe mean from a StatCounter (avoid division by zero)."""
    return sc.mean() if sc.count() > 0 else float("nan")


# ==============================================================================
# BOROUGH SPATIAL LOOK-UP
# A lightweight broadcaster: load the small GeoJSON with Shapely (available
# in Python environments that have geopandas/shapely) and broadcast the
# list of (borough_name, shapely_polygon) pairs so every executor can call
# point-in-polygon without network I/O.
# ==============================================================================

def load_borough_polygons(geojson_path: str):
    """
    Read the borough GeoJSON and return a list of
    (borough_name, shapely.geometry.shape) tuples.
    Requires: shapely  (pre-installed or pip-installed in pylibs).
    """
    from shapely.geometry import shape as shapely_shape

    with open(geojson_path, "r") as fh:
        gj = json.load(fh)

    polygons = []
    for feat in gj["features"]:
        name = (feat.get("properties", {}).get("boro_name")
                or feat.get("properties", {}).get("borough")
                or feat.get("properties", {}).get("name")
                or "Unknown")
        geom = shapely_shape(feat["geometry"])
        polygons.append((name, geom))
    return polygons


def make_point_to_borough_fn(bc_polygons):
    """
    Return a closure that maps (lon, lat) -> borough name (str).
    Uses bounding-box pre-filtering to avoid 1.5 billion expensive contains() checks.
    """
    from shapely.geometry import Point
    
    
    polys_with_bounds = [(name, poly, poly.bounds) for name, poly in bc_polygons.value]

    def point_to_borough(lon: float, lat: float) -> str:
        try:
            pt = Point(lon, lat)
            # Iteriamo sulla lista ottimizzata
            for name, poly, (minx, miny, maxx, maxy) in polys_with_bounds:
                # 1. Bounding Box Check (Istantaneo)
                if minx <= lon <= maxx and miny <= lat <= maxy:
                    # 2. Polygon Check (Solo se necessario)
                    if poly.contains(pt):
                        return name
        except Exception:
            pass
        return "Unknown"

    return point_to_borough


# ==============================================================================
# DATA LOADING & PARSING
# ==============================================================================

# Each row in the RDD will be a dict with these keys:
#   hack_license, pickup_dt, dropoff_dt, duration_secs,
#   pickup_lon, pickup_lat, dropoff_lon, dropoff_lat

def parse_csv_line(line: str):
    """
    Parse a single CSV line into a trip dict.
    Returns None for the header row or malformed lines.
    """
    parts = line.split(",")
    if len(parts) < 14:
        return None
    if parts[0].strip() == "medallion":   # header row
        return None
    try:
        pickup_dt  = parse_datetime(parts[5])
        dropoff_dt = parse_datetime(parts[6])
        if pickup_dt is None or dropoff_dt is None:
            return None

        pickup_lon  = float(parts[10])
        pickup_lat  = float(parts[11])
        dropoff_lon = float(parts[12])
        dropoff_lat = float(parts[13])

        duration = (dropoff_dt - pickup_dt).total_seconds()
        if duration <= 0:
            return None

        return {
            "hack_license": parts[1].strip(),
            "pickup_dt":    pickup_dt,
            "dropoff_dt":   dropoff_dt,
            "duration":     duration,           # seconds
            "pickup_lon":   pickup_lon,
            "pickup_lat":   pickup_lat,
            "dropoff_lon":  dropoff_lon,
            "dropoff_lat":  dropoff_lat,
        }
    except (ValueError, IndexError):
        return None


def load_trips_rdd(spark: SparkSession, path: str, sample: bool):
    """
    Load the CSV, parse each line, optionally sample.
    Returns a *cached* RDD of trip dicts (malformed rows filtered out).
    """
    raw_rdd = spark.sparkContext.textFile(path)

    if sample:
        raw_rdd = raw_rdd.sample(False, LOCAL_SAMPLE_FRACTION, LOCAL_SAMPLE_SEED)

    trips_rdd = (
        raw_rdd
        .map(parse_csv_line)
        .filter(lambda x: x is not None)
    )
    trips_rdd.cache()
    return trips_rdd


# ==============================================================================
# ENRICH RDD WITH BOROUGH LABELS
# ==============================================================================

def enrich_with_boroughs(trips_rdd, bc_polygons):
    """
    Add pickup_borough and dropoff_borough keys to each trip dict.
    Runs the point-in-polygon check on each executor via the broadcast var.
    """
    p2b = make_point_to_borough_fn(bc_polygons)

    def _enrich(trip):
        t = dict(trip)   # shallow copy – avoid mutating cached RDD
        t["pickup_borough"]  = p2b(t["pickup_lon"],  t["pickup_lat"])
        t["dropoff_borough"] = p2b(t["dropoff_lon"], t["dropoff_lat"])
        return t

    return trips_rdd.map(_enrich)


# ==============================================================================
# TASK (a) – Same-borough trips: count, sum, average duration
# ==============================================================================

def task_a(enriched_rdd, report: Report):
    """
    Aggregate statistics for trips where pickup_borough == dropoff_borough.
    Uses aggregateByKey with StatCounter, keyed on borough name.
    """
    report.add("\n" + "=" * 70)
    report.add("TASK (a) – Trips that started AND ended in the SAME borough")
    report.add("=" * 70)

    # Filter to same-borough trips; key = borough name
    same_rdd = (
        enriched_rdd
        .filter(lambda t: (t["pickup_borough"] == t["dropoff_borough"]
                           and t["pickup_borough"] != "Unknown"))
        .map(lambda t: (t["pickup_borough"], t["duration"]))
    )

    # aggregateByKey with StatCounter:
    #   zero value    → empty StatCounter
    #   seq function  → add one value to counter
    #   comb function → merge two counters
    zero   = StatCounter()
    seq_fn = lambda sc, v: sc.merge(v)
    comb   = lambda a, b: a.mergeStats(b)

    result = (
        same_rdd
        .aggregateByKey(zero, seq_fn, comb)
        .mapValues(lambda sc: {
            "count":    sc.count(),
            "sum_secs": sc.sum(),
            "avg_secs": statcounter_mean(sc),
        })
        .sortBy(lambda kv: kv[1]["count"], ascending=False)
        .collect()
    )

    for borough, stats in result:
        report.add(f"  {borough:<20}  count={stats['count']:>8,}  "
                   f"sum={stats['sum_secs']:>14,.1f}s  "
                   f"avg={stats['avg_secs']:>8.1f}s")
    return result


# ==============================================================================
# TASK (b) – Different-borough trips: count, sum, average duration
# ==============================================================================

def task_b(enriched_rdd, report: Report):
    """
    Aggregate statistics for trips where pickup_borough != dropoff_borough.
    Key = (pickup_borough, dropoff_borough) for detailed breakdown,
    plus a global summary.
    """
    report.add("\n" + "=" * 70)
    report.add("TASK (b) – Trips that started and ended in DIFFERENT boroughs")
    report.add("=" * 70)

    diff_rdd = (
        enriched_rdd
        .filter(lambda t: (t["pickup_borough"] != t["dropoff_borough"]
                           and t["pickup_borough"] != "Unknown"
                           and t["dropoff_borough"] != "Unknown"))
        .map(lambda t: ((t["pickup_borough"], t["dropoff_borough"]),
                        t["duration"]))
    )

    zero   = StatCounter()
    seq_fn = lambda sc, v: sc.merge(v)
    comb   = lambda a, b: a.mergeStats(b)

    result = (
        diff_rdd
        .aggregateByKey(zero, seq_fn, comb)
        .mapValues(lambda sc: {
            "count":    sc.count(),
            "sum_secs": sc.sum(),
            "avg_secs": statcounter_mean(sc),
        })
        .sortBy(lambda kv: kv[1]["count"], ascending=False)
        .collect()
    )

    for (src, dst), stats in result:
        report.add(f"  {src:<20} -> {dst:<20}  count={stats['count']:>8,}  "
                   f"sum={stats['sum_secs']:>14,.1f}s  "
                   f"avg={stats['avg_secs']:>8.1f}s")

    # Global summary across all inter-borough trips
    total_count = sum(s["count"]    for _, s in result)
    total_sum   = sum(s["sum_secs"] for _, s in result)
    total_avg   = total_sum / total_count if total_count > 0 else float("nan")
    report.add(f"\n  GLOBAL  count={total_count:>8,}  "
               f"sum={total_sum:>14,.1f}s  avg={total_avg:>8.1f}s")
    return result


# ==============================================================================
# TASK (c) – Average gap between consecutive trips by same driver,
#             grouped by (pickup_borough of trip i, hour_of_day of trip i)
# ==============================================================================

def task_c(enriched_rdd, report: Report):
    """
    For each driver (hack_license):
      1. Sort trips chronologically.
      2. Form consecutive pairs (trip_i, trip_{i+1}).
      3. Gap = trip_{i+1}.pickup_dt  −  trip_i.dropoff_dt   (seconds).
      4. Key  = (trip_i.pickup_borough, hour(trip_i.pickup_dt))
      5. Aggregate gap statistics with StatCounter.
    """
    report.add("\n" + "=" * 70)
    report.add("TASK (c) – Avg gap between consecutive trips per "
               "(borough, hour-of-day)")
    report.add("=" * 70)

    # key = driver id, value = trip record
    by_driver = (
        enriched_rdd
        .filter(lambda t: t["pickup_borough"] != "Unknown")
        .map(lambda t: (t["hack_license"], t))
    )

    def make_pairs(kv):
        """
        For one driver, sort trips and yield
        ((borough_i, hour_i), gap_seconds) for each consecutive pair.
        """
        _, trips = kv
        # Sort by pickup datetime
        sorted_trips = sorted(trips, key=lambda t: t["pickup_dt"])
        pairs = []
        for i in range(len(sorted_trips) - 1):
            trip_i   = sorted_trips[i]
            trip_ip1 = sorted_trips[i + 1]
            gap = (trip_ip1["pickup_dt"] - trip_i["dropoff_dt"]).total_seconds()
            if gap < 0:
                continue   # overlapping trips – skip
            borough_i = trip_i["pickup_borough"]
            hour_i    = trip_i["pickup_dt"].hour
            pairs.append(((borough_i, hour_i), gap))
        return pairs

    # Group all trips per driver first (groupByKey), then flatMap pairs.
    # groupByKey is appropriate here because we need the full sorted sequence
    # per driver to identify consecutive trips.
    gap_rdd = (
        by_driver
        .groupByKey()
        .flatMap(make_pairs)
    )

    zero   = StatCounter()
    seq_fn = lambda sc, v: sc.merge(v)
    comb   = lambda a, b: a.mergeStats(b)

    result = (
        gap_rdd
        .aggregateByKey(zero, seq_fn, comb)
        .mapValues(lambda sc: {
            "count":   sc.count(),
            "avg_gap": statcounter_mean(sc),
        })
        .sortBy(lambda kv: (kv[0][0], kv[0][1]))   # sort by borough, hour
        .collect()
    )

    report.add(f"  {'Borough':<20}  {'Hour':>4}  {'Pairs':>8}  {'Avg gap (s)':>12}")
    report.add("  " + "-" * 50)
    for (borough, hour), stats in result:
        report.add(f"  {borough:<20}  {hour:>4}  "
                   f"{stats['count']:>8,}  {stats['avg_gap']:>12.1f}")
    return result


# ==============================================================================
# TASK (d) – Outlier detection via normalised duration (duration / distance)
#             at 95th-percentile cut-off, per starting borough
# ==============================================================================

def task_d(enriched_rdd, report: Report):
    """
    Steps:
      1. Compute Haversine distance between pickup and dropoff coords.
      2. Compute ratio = duration_secs / distance_miles (skip ~0-dist trips).
      3. Group ratios by pickup_borough.
      4. For each borough, find the 95th-percentile threshold.
      5. Report trips that exceed the threshold (outliers).
    """
    report.add("\n" + "=" * 70)
    report.add("TASK (d) – Outlier detection (normalized duration > 95th pct "
               "per borough)")
    report.add("=" * 70)

    def compute_ratio(trip):
        dist = haversine_miles(trip["pickup_lat"],  trip["pickup_lon"],
                               trip["dropoff_lat"], trip["dropoff_lon"])
        if dist < 0.01:   # effectively zero distance – skip
            return None
        ratio = trip["duration"] / dist
        return (trip["pickup_borough"], ratio)

    ratio_rdd = (
        enriched_rdd
        .filter(lambda t: t["pickup_borough"] != "Unknown")
        .map(compute_ratio)
        .filter(lambda x: x is not None)
    )
    ratio_rdd.cache()

    # Collect all ratios per borough to compute exact 95th percentile.
    # Only 5 NYC boroughs → collecting per-borough sorted lists is safe.
    borough_ratios = (
        ratio_rdd
        .groupByKey()
        .mapValues(lambda vs: sorted(vs))
        .collectAsMap()
    )

    # Compute 95th-percentile threshold per borough
    thresholds = {}
    for borough, ratios in borough_ratios.items():
        n   = len(ratios)
        idx = max(0, int(math.ceil(0.95 * n)) - 1)
        thresholds[borough] = ratios[idx]

    report.add("\n  95th-percentile thresholds (sec/mile):")
    for b, thr in sorted(thresholds.items()):
        n = len(borough_ratios[b])
        report.add(f"    {b:<20}  threshold={thr:>10.2f}  total_trips={n:>8,}")

    # Broadcast thresholds so they are available on executors
    bc_thresholds = ratio_rdd.context.broadcast(thresholds)

    # Find outliers
    outliers_rdd = (
        ratio_rdd
        .filter(lambda kv: kv[1] > bc_thresholds.value.get(kv[0], float("inf")))
    )

    outlier_counts = (
        outliers_rdd
        .aggregateByKey(
            StatCounter(),
            lambda sc, v: sc.merge(v),
            lambda a, b: a.mergeStats(b),
        )
        .mapValues(lambda sc: {
            "outlier_count": sc.count(),
            "avg_ratio":     statcounter_mean(sc),
        })
        .sortBy(lambda kv: kv[0])
        .collect()
    )

    report.add("\n  Outlier summary (trips > 95th pct):")
    for borough, stats in outlier_counts:
        pct = (stats["outlier_count"] / len(borough_ratios[borough]) * 100
               if borough in borough_ratios else 0.0)
        report.add(f"    {borough:<20}  outliers={stats['outlier_count']:>8,}  "
                   f"({pct:.1f}%)  avg_ratio={stats['avg_ratio']:>10.2f} s/mile")

    ratio_rdd.unpersist()
    return outlier_counts


# ==============================================================================
# TASK (e) – Rush-hour detection: average normalised duration per hour-of-day
# ==============================================================================

def task_e(enriched_rdd, report: Report):
    """
    Steps:
      1. Compute Haversine distance between pickup and dropoff.
      2. ratio = duration / distance.
      3. Key  = hour_of_day (0–23).
      4. Aggregate ratio with StatCounter → compute average per hour.
      5. Sort descending by average ratio (peak hours first).
    """
    report.add("\n" + "=" * 70)
    report.add("TASK (e) – Rush-hour detection (avg normalised duration "
               "per hour-of-day, descending)")
    report.add("=" * 70)

    def hour_ratio(trip):
        dist = haversine_miles(trip["pickup_lat"],  trip["pickup_lon"],
                               trip["dropoff_lat"], trip["dropoff_lon"])
        if dist < 0.01:
            return None
        return (trip["pickup_dt"].hour, trip["duration"] / dist)

    hourly_rdd = (
        enriched_rdd
        .map(hour_ratio)
        .filter(lambda x: x is not None)
    )

    zero   = StatCounter()
    seq_fn = lambda sc, v: sc.merge(v)
    comb   = lambda a, b: a.mergeStats(b)

    result = (
        hourly_rdd
        .aggregateByKey(zero, seq_fn, comb)
        .mapValues(lambda sc: {
            "trip_count": sc.count(),
            "avg_ratio":  statcounter_mean(sc),
        })
        .sortBy(lambda kv: kv[1]["avg_ratio"], ascending=False)
        .collect()
    )

    report.add(f"\n  {'Hour':>4}  {'Trips':>10}  {'Avg sec/mile':>14}")
    report.add("  " + "-" * 34)
    for hour, stats in result:
        report.add(f"  {hour:>4}  {stats['trip_count']:>10,}  "
                   f"{stats['avg_ratio']:>14.2f}")
    return result


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    t0 = time.time()

    # ── 1. SparkSession ────────────────────────────────────────────────────────
    spark = build_spark_session()
    # Wrap with SedonaContext so spatial UDFs are registered (even if we use
    # Shapely for the broadcast approach, Sedona may still be needed for
    # the cluster environment).
    try:
        sc_sedona = SedonaContext.create(spark)   # noqa: F841
    except Exception as ex:
        print(f"[WARN] SedonaContext.create failed ({ex}); "
              "continuing without Sedona SQL functions.")

    sc = spark.sparkContext

    print(f"Spark version : {spark.version}")
    print(f"LOCAL_TEST    : {LOCAL_TEST}")
    print(f"Taxi data     : {TAXI_CSV_PATH}")
    print(f"Boroughs      : {BOROUGHS_GEOJSON_PATH}")

    # ── 2. Load borough polygons & broadcast ──────────────────────────────────
    print("\n[INFO] Loading borough polygons …")
    borough_polygons = load_borough_polygons(BOROUGHS_GEOJSON_PATH)
    bc_polygons      = sc.broadcast(borough_polygons)
    print(f"[INFO] {len(borough_polygons)} borough polygon(s) loaded: "
          f"{[b for b, _ in borough_polygons]}")

    # ── 3. Load & parse taxi trips RDD ────────────────────────────────────────
    print("\n[INFO] Loading taxi trips …")
    trips_rdd = load_trips_rdd(spark, TAXI_CSV_PATH, sample=LOCAL_TEST)

    # Force evaluation so we can report counts
    n_trips = trips_rdd.count()
    print(f"[INFO] Parsed trips : {n_trips:,}")

    # ── 4. Enrich with borough labels & cache ─────────────────────────────────
    print("[INFO] Running spatial join (point-in-polygon) …")
    enriched_rdd = enrich_with_boroughs(trips_rdd, bc_polygons)
    enriched_rdd.cache()
    n_enriched = enriched_rdd.count()
    print(f"[INFO] Enriched trips: {n_enriched:,}")

    # ── 5. Run tasks (all results go into report) ─────────────────────────────
    report = Report()

    task_a(enriched_rdd, report)
    task_b(enriched_rdd, report)
    task_c(enriched_rdd, report)
    task_d(enriched_rdd, report)
    task_e(enriched_rdd, report)

    # ── 6. Write clean output file ────────────────────────────────────────────
    total_runtime = time.time() - t0
    report.save("Problem_2.txt", total_runtime)

    # ── 7. Clean-up ───────────────────────────────────────────────────────────
    enriched_rdd.unpersist()
    trips_rdd.unpersist()
    bc_polygons.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()
