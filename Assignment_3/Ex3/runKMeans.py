import warnings
warnings.filterwarnings('ignore')

import os, sys, math, time
import numpy as np
from pyspark.sql import SparkSession
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import Vectors

try:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    print("matplotlib / sklearn not available — sample will be saved to CSV for local plotting.", flush=True)

def log(msg=""):
    print(msg, flush=True)

# ── 1. Spark session ──────────────────────────────────────────────────────────
log("=== Initialising Spark ===")
spark = SparkSession.builder.appName("GearboxOutlierDetection").getOrCreate()
sc    = spark.sparkContext

# ── 2. Load and parse data ────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.getcwd(), "data/gearbox-fault-detection-dataset-phm-2009-nasa/*.csv")
log(f"\n=== Loading data from: {DATA_PATH} ===")

def parse_line(line):
    try:
        features = [float(x) for x in line.split(',')]
        return Vectors.dense(features) if len(features) == 3 else None
    except ValueError:
        return None

t0      = time.time()
rawData = sc.textFile(DATA_PATH)
data    = rawData.map(parse_line).filter(lambda x: x is not None).cache()
n       = data.count()
log(f"Total readings loaded: {n:,}  (runtime: {time.time()-t0:.1f}s)")

# ── 3. Normalisation ──────────────────────────────────────────────────────────
log("\n=== Normalising data ===")

def normalize(data_rdd):
    data_as_array = data_rdd.map(lambda x: x.toArray())
    num_cols = len(data_as_array.first())
    n        = data_as_array.count()
    sums     = data_as_array.reduce(lambda a, b: [x + y for x, y in zip(a, b)])
    sum_sq   = data_as_array.aggregate(
        [0.0] * num_cols,
        lambda a, b: [x + y * y for x, y in zip(a, b)],
        lambda a, b: [x + y     for x, y in zip(a, b)]
    )
    stdevs = [math.sqrt(max(0, n * ssq - s * s)) / n for ssq, s in zip(sum_sq, sums)]
    means  = [s / n for s in sums]

    def normalize_vector(vector):
        return Vectors.dense([
            (v - m) / sd if sd > 0 else (v - m)
            for v, m, sd in zip(vector.toArray(), means, stdevs)
        ])
    return normalize_vector

t0              = time.time()
normalize_func  = normalize(data)
normalized_data = data.map(normalize_func).cache()
normalized_data.count()   # trigger caching now so it isn't charged to k=2
log(f"Normalisation done  (runtime: {time.time()-t0:.1f}s)")

# ── 4. Distance helpers ───────────────────────────────────────────────────────
def distance(a, b):
    return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))

def distToCentroid(vector, model):
    cluster  = model.predict(vector)
    centroid = model.clusterCenters[cluster]
    return distance(centroid, vector)

# ── 5. Outlier detection loop k = 2 … 12 ─────────────────────────────────────
log("\n=== Outlier detection loop (k = 2 … 12) ===")
last_model = None

for k in range(2, 13):
    log(f"\n{'─'*55}")
    log(f"  k = {k}")
    log(f"{'─'*55}")

    t0         = time.time()
    model      = KMeans.train(normalized_data, k=k, maxIterations=20)
    log(f"  Model trained  (runtime: {time.time()-t0:.1f}s)")
    last_model = model

    # Cluster sizes
    t0 = time.time()
    cluster_sizes = (normalized_data
                     .map(lambda v: (model.predict(v), 1))
                     .reduceByKey(lambda a, b: a + b)
                     .sortByKey()
                     .collect())
    log(f"  Cluster sizes  (runtime: {time.time()-t0:.1f}s):")
    for cid, cnt in cluster_sizes:
        log(f"    cluster {cid:2d}: {cnt:>10,} readings")

    # Top-25 outliers
    t0             = time.time()
    scored         = data.map(lambda v: (distToCentroid(normalize_func(v), model), v))
    top25_outliers = scored.takeOrdered(25, key=lambda x: -x[0])
    log(f"  Top-25 outliers  (runtime: {time.time()-t0:.1f}s):")
    log(f"  {'Rank':>4}  {'Distance':>10}  {'vib_hs':>12}  {'vib_ls':>12}  {'speed':>12}")
    for rank, (dist, vec) in enumerate(top25_outliers, 1):
        v = vec.toArray()
        log(f"  {rank:>4}  {dist:>10.4f}  {v[0]:>12.6f}  {v[1]:>12.6f}  {v[2]:>12.6f}")

# ── 6. Visualisation (one plot, k=12) ────────────────────────────────────────
log("\n=== Generating visualisation (k=12, small sample) ===")
t0          = time.time()
sample_vecs = normalized_data.sample(False, 0.001, seed=42).collect()
log(f"  Sample collected: {len(sample_vecs)} points  (runtime: {time.time()-t0:.1f}s)")

if sample_vecs:
    X_sample = np.array([v.toArray() for v in sample_vecs])
    labels   = [last_model.predict(v) for v in sample_vecs]

    if VIZ_AVAILABLE:
        pca   = PCA(n_components=2)
        X_pca = pca.fit_transform(X_sample)
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6, s=5)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title('Gearbox K-Means (k=12) – PCA 2-D projection')
        plt.xlabel("PC 1"); plt.ylabel("PC 2")
        plt.tight_layout()
        plt.savefig('gearbox_clusters_pca.png', dpi=120)
        log("  Plot saved → gearbox_clusters_pca.png")
    else:
        import csv
        csv_path = 'gearbox_k12.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['vib_hs', 'vib_ls', 'speed', 'cluster'])
            for vec, lbl in zip(X_sample, labels):
                writer.writerow([*vec, lbl])
        log(f"  Sample saved → {csv_path}  (run plot_sample.py locally to generate the plot)")

# ── 7. Cleanup ────────────────────────────────────────────────────────────────
data.unpersist()
normalized_data.unpersist()
spark.stop()
log("\n=== Done ===")
