"""
RunLSA_MoviePlots.py
====================
Latent Semantic Analysis on the "Wikipedia Movie Plots" dataset.
Adapted from the professor's RDD-based mllib pipeline (RunLSA.ipynb).

Run with:
  spark-submit RunLSA_MoviePlots.py

Dataset expected at: ./wiki_movie_plots_deduped.csv
  (Kaggle: "Wikipedia Movie Plots" – 134k rows)

Key adaptations vs. the original RunLSA.ipynb
----------------------------------------------
- Data source  : CSV (multiLine, escaped quotes) instead of wholeTextFiles
- NLP function : plainTextToLemmas((title, genre, plot)) → ((title, genre), lemmas)
- docIds dict  : maps numeric row-ID → (Title, Genre)  (not just Title)
- Hyperparams  : numFreq=5000, k=25  (assignment spec)
- Search engine: 8 example keyword queries, top-5 results printed with Title + Genre
"""

# ---------------------------------------------------------------------------
# 0.  Imports & configuration
# ---------------------------------------------------------------------------
import sys, io, os
os.environ["NLTK_DATA"] = "/home/users/mfrattarola/nltk_data"
#sys.path.append("/mnt/aiongpfs/users/mfrattarola/.local/lib/python3.6/site-packages")
import nltk
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import math
import numpy as np
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType
)
from pyspark.mllib.linalg import Vectors, Matrices
from pyspark.mllib.linalg.distributed import RowMatrix

from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# 1.  Hyper-parameters & local-test flag
# ---------------------------------------------------------------------------

IS_LOCAL_TEST = False    # ← set False for the full 134 k-row run
SAMPLE_FRAC   = 0.01    # fraction used when IS_LOCAL_TEST is True

NUM_FREQ = 5000         # vocabulary size (top-N terms by document frequency)
K        = 25           # number of latent concepts

NUM_TOP_TERMS = 25      # terms to print per concept
NUM_TOP_DOCS  = 25      # docs  to print per concept
NUM_TOP_QUERY = 5       # results to return per search query

QUERIES = [
    ["alien", "space", "ship"],
    ["magic", "wizard", "school"],
    ["robot", "future", "war"],
    ["vampire", "blood", "love"],
    ["detective", "murder", "mystery"],
    ["shark", "ocean", "attack"]
]

CSV_PATH = "wiki_movie_plots_deduped.csv"

# ---------------------------------------------------------------------------
# 2.  SparkSession
# ---------------------------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("LSA-MoviePlots")
    #.config("spark.executorEnv.PYTHONPATH", "/mnt/aiongpfs/users/mfrattarola/.local/lib/python3.6/site-packages")
    .getOrCreate()
)
sc = spark.sparkContext
sc.setLogLevel("WARN")

print("="*70)
print("LSA on Wikipedia Movie Plots")
print(f"IS_LOCAL_TEST={IS_LOCAL_TEST}  |  numFreq={NUM_FREQ}  |  k={K}")
print("="*70)

# ---------------------------------------------------------------------------
# 3.  Data loading & safe parsing
#     Explicit schema prevents Spark from mis-inferring column boundaries
#     when Plot cells contain embedded commas / newlines / escaped quotes.
# ---------------------------------------------------------------------------

schema = StructType([
    StructField("Release Year",      IntegerType(), True),
    StructField("Title",             StringType(),  True),
    StructField("Origin/Ethnicity",  StringType(),  True),
    StructField("Director",          StringType(),  True),
    StructField("Cast",              StringType(),  True),
    StructField("Genre",             StringType(),  True),
    StructField("Wiki Page",         StringType(),  True),
    StructField("Plot",              StringType(),  True),
])

raw_df = (
    spark.read
    .option("header",    "true")
    .option("multiLine", "true")     # Plot fields contain embedded newlines
    .option("escape",    "\"")       # handles "" escaped quotes inside fields
    .option("quote",     "\"")
    .schema(schema)
    .csv(CSV_PATH)
)

# Keep only what we need and drop rows without a plot
movies_df = (
    raw_df
    .select("Title", "Genre", "Plot")
    .dropna(subset=["Plot"])
)

# Optional 1 % sample to prevent OOM on a laptop
if IS_LOCAL_TEST:
    movies_df = movies_df.sample(fraction=SAMPLE_FRAC, seed=42)
    print(f"[LOCAL TEST] Sampled {movies_df.count()} rows")
else:
    print(f"[FULL RUN] Total rows: {movies_df.count()}")

# Convert DataFrame → RDD of (Title, Genre, Plot) tuples
# Using rdd.map keeps us in the RDD world that the professor's pipeline expects.
plainText = movies_df.rdd.map(lambda row: (row["Title"], row["Genre"], row["Plot"]))
plainText.cache()

numDocs = plainText.count()
bNumDocs = sc.broadcast(numDocs)
print(f"Documents after filtering: {numDocs}")

# ---------------------------------------------------------------------------
# 4.  Stopwords – load from NLTK corpus (no external file dependency)
#     If you have the professor's stopwords.txt, replace the two lines
#     below with:
#         stopwords = set(sc.textFile("stopwords.txt").collect())
# ---------------------------------------------------------------------------

from nltk.corpus import stopwords as nltk_sw
try:
    _sw_set = set(nltk_sw.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords", quiet=True)
    _sw_set = set(nltk_sw.words("english"))

bStopWords = sc.broadcast(_sw_set)
print(f"Stopwords loaded: {len(bStopWords.value)}")

# ---------------------------------------------------------------------------
# 5.  NLP pipeline
#     Adapted from professor's plainTextToLemmas(title, text).
#     New signature: input  = (title, genre, plot)
#                    output = ((title, genre), [lemmas])
# ---------------------------------------------------------------------------

def plainTextToLemmas(title_genre_text):
    """
    Tokenise, lemmatise, and filter a movie plot.

    Parameters
    ----------
    title_genre_text : tuple  (title: str, genre: str, plot: str)

    Returns
    -------
    tuple  ((title, genre), lemmas: list[str])
    """
    title, genre, text = title_genre_text

    # Guard against None plots (shouldn't happen after dropna, but be safe)
    if not text:
        return ((title, genre), [])

    lemmatizer = WordNetLemmatizer()
    lemmas = []

    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [text]

    for sentence in sentences:
        try:
            tokens = word_tokenize(sentence)
        except Exception:
            tokens = sentence.split()

        for token in tokens:
            lemma = lemmatizer.lemmatize(token.lower())
            if (len(lemma) > 2
                    and lemma not in bStopWords.value
                    and lemma.isalpha()):
                lemmas.append(lemma)

    return ((title, genre), lemmas)


# Run NLP once per partition (mirrors professor's mapPartitions call)
lemmatized = plainText.mapPartitions(lambda rows: map(plainTextToLemmas, rows))

# ---------------------------------------------------------------------------
# 6.  Term Frequencies
#     Adapted from professor's calculateTermFreqs.
#     key is now (title, genre) instead of title.
# ---------------------------------------------------------------------------

def calculateTermFreqs(key_lemmas):
    """((title, genre), [lemmas])  →  ((title, genre), {term: count})"""
    key, lemmas = key_lemmas
    termFreqs = {}
    for term in lemmas:
        termFreqs[term] = termFreqs.get(term, 0) + 1
    return (key, termFreqs)


docTermFreqs = lemmatized.map(calculateTermFreqs)
docTermFreqs.cache()

# ---------------------------------------------------------------------------
# 7.  docIds: numeric row-ID → (Title, Genre)
#     Professor's original: docIds maps ID → Title.
#     We extend it to map ID → (Title, Genre) tuple.
# ---------------------------------------------------------------------------

# RDD[(Title,Genre)] → zipWithUniqueId → RDD[(id, (Title,Genre))]
docIds = (
    docTermFreqs
    .map(lambda x: x[0])            # extract (title, genre) key
    .zipWithUniqueId()
    .map(lambda x: (x[1], x[0]))    # (id, (title, genre))
    .collectAsMap()
)
print(f"docIds built: {len(docIds)} entries")

# ---------------------------------------------------------------------------
# 8.  Document-frequency counts & IDF (mirrors professor's pipeline exactly)
# ---------------------------------------------------------------------------

docFreqs = (
    docTermFreqs
    .flatMap(lambda x: x[1].keys())
    .map(lambda term: (term, 1))
    .reduceByKey(lambda a, b: a + b, numPartitions=24)
)

topDocFreqs = docFreqs.top(NUM_FREQ, key=lambda x: x[1])

idfs     = {term: math.log(numDocs / count) for term, count in topDocFreqs}
idTerms  = dict(zip(idfs.keys(), range(len(idfs))))   # term → int index
termIds  = {v: k for k, v in idTerms.items()}         # int index → term

bIdfs    = sc.broadcast(idfs)
bIdTerms = sc.broadcast(idTerms)

print(f"Vocabulary size: {len(idfs)}")

# ---------------------------------------------------------------------------
# 9.  TF-IDF sparse row vectors & SVD
#     Identical to professor's code; only the upstream RDD key changed.
# ---------------------------------------------------------------------------

rowVectors = docTermFreqs.map(lambda x: x[1]).map(
    lambda termFreqs: Vectors.sparse(
        len(bIdTerms.value),
        [(bIdTerms.value[term],
          bIdfs.value[term] * termFreqs[term] / max(sum(termFreqs.values()), 1))
         for term in termFreqs
         if term in bIdTerms.value]
    )
)
rowVectors.cache()

mat = RowMatrix(rowVectors)
svd = mat.computeSVD(K, computeU=True)

print(f"\nSVD complete  |  U: ({mat.numRows()} × {K})  |  V: ({NUM_FREQ} × {K})")
print(f"Singular values: {svd.s.toArray().round(4)}")

# ---------------------------------------------------------------------------
# 10.  US = U × diag(S)  – needed for query projection
# ---------------------------------------------------------------------------

def multiplyByDiagonalRowMatrix(mat, diag):
    """Scale each row of a RowMatrix by the diagonal vector."""
    s_arr = diag.toArray()
    return RowMatrix(
        mat.rows.map(lambda vec: Vectors.dense(np.multiply(vec.toArray(), s_arr)))
    )


US = multiplyByDiagonalRowMatrix(svd.U, svd.s)

# ---------------------------------------------------------------------------
# 11.  topTermsInTopConcepts
#      Adapted from professor's function.  Unchanged logic; uses termIds.
# ---------------------------------------------------------------------------

def topTermsInTopConcepts(svd_result, numConcepts, numTerms):
    """
    Returns list of lists: for each concept, the top-numTerms (term, score) pairs.
    """
    top_terms = []
    v = svd_result.V
    arr = v.toArray().T          # shape: (numConcepts, numRows/vocab)

    for i in range(numConcepts):
        term_weights = [(arr[i][tid], tid) for tid in range(v.numRows)]
        sorted_terms = sorted(term_weights, key=lambda x: -x[0])
        top_terms.append(
            [(termIds.get(idx, str(idx)), score)
             for score, idx in sorted_terms[:numTerms]]
        )
    return top_terms


# ---------------------------------------------------------------------------
# 12.  topDocsInTopConcepts
#      Key change: docIds now returns (Title, Genre) tuples.
#      We format the display as "Title [Genre]".
# ---------------------------------------------------------------------------

def topDocsInTopConcepts(svd_result, numConcepts, numDocs_to_show):
    """
    Returns list of lists: for each concept, the top-numDocs_to_show
    (doc_label, score) pairs where doc_label = "Title [Genre]".
    """
    top_docs = []
    u = svd_result.U

    # Collect (score_in_concept_i, row_index) for every row
    # We must do this per-concept from the U matrix rows.
    u_collected = (
        u.rows
        .zipWithUniqueId()
        .map(lambda row_id: (row_id[1], row_id[0].toArray()))   # (id, array)
        .collect()
    )

    for i in range(numConcepts):
        doc_weights = [(arr[i], doc_id) for doc_id, arr in u_collected]
        sorted_docs = sorted(doc_weights, key=lambda x: -x[0])

        top_docs.append([
            (_format_doc(docIds.get(idx)), score)
            for score, idx in sorted_docs[:numDocs_to_show]
        ])
    return top_docs


def _format_doc(title_genre):
    if title_genre is None:
        return "Unknown"
    title, genre = title_genre
    genre_str = genre if genre else "unknown"
    label = f"{title}  [{genre_str}]"
    # Safe fallback for terminals that can't handle full Unicode
    return label.encode("utf-8", errors="replace").decode("utf-8")

# ---------------------------------------------------------------------------
# 12b. Helper functions for Search Engine & Queries
#      Recuperate dalla cella 14 del notebook RunLSA.ipynb
# ---------------------------------------------------------------------------

def termsToQueryVector(terms, idTerms, idfs):
    indices = [idTerms.get(term) for term in terms if term in idTerms]
    values = [idfs[term] for term in terms if term in idfs]
    
    
    if not indices:
        return None
        
    return csr_matrix((values, (indices, [0]*len(indices))), shape=(len(idTerms), 1))

def topDocsForTermQuery(US, V, query, n=10):
    term_row_arr = np.dot(V.toArray().T, query.toarray()).flatten()
    term_row_vec = Matrices.dense(len(term_row_arr), 1, term_row_arr)
    doc_scores = US.multiply(term_row_vec)
    
    all_doc_weights = doc_scores.rows.zipWithUniqueId().map(
        lambda x: (x[0].toArray()[0], docIds.get(x[1], ("Unknown", "Unknown")))
    )
    return sorted(all_doc_weights.collect(), key=lambda x: -x[0])[:n]

# ---------------------------------------------------------------------------
# 13. Save Results to Output File (Replaces old Sections 13, 14, 15)
# ---------------------------------------------------------------------------
# We redirect the output to a text file as requested by the assignment
OUTPUT_FILE = "Problem_2.txt"

# Open the file in write mode
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    
    # Helper function to print to the console AND write to the file simultaneously
    def log_print(text):
        print(text)
        f.write(text + "\n")

    log_print("\n" + "="*70)
    log_print(f"TOP {NUM_TOP_TERMS} TERMS & TOP {NUM_TOP_DOCS} DOCS PER CONCEPT  (k={K})")
    log_print("="*70)

    # Calculate top terms and documents
    top_concept_terms = topTermsInTopConcepts(svd, K, NUM_TOP_TERMS)
    top_concept_docs  = topDocsInTopConcepts(svd, K, NUM_TOP_DOCS)

    # Iterate through concepts and log them
    for concept_idx, (terms, docs) in enumerate(zip(top_concept_terms, top_concept_docs)):
        log_print(f"\n--- Concept {concept_idx + 1} ---")
        log_print("Terms: " + ",  ".join(f"{t}" for t, _ in terms))
        log_print("Docs:")
        for doc_label, score in docs:
            log_print(f"  {score:+.4f}  {doc_label}")

    log_print("\n" + "="*70)
    log_print("SEARCH ENGINE  - Keyword Query Results")
    log_print("="*70)

    # Run and log the keyword queries
    for query_terms in QUERIES:
        qvec = termsToQueryVector(query_terms, idTerms, idfs)
        
        if qvec is None:
            log_print(f"\nQuery {query_terms}: No terms found in vocabulary - skipping")
            continue

        results = topDocsForTermQuery(US, svd.V, qvec, n=NUM_TOP_QUERY)

        log_print(f"\nQuery: {query_terms}")
        for rank, (score, (title, genre)) in enumerate(results, 1):
            log_print(f"  {rank}. [{genre:>20}]  {title}  (score={score:.4f})")

    log_print("\n" + "="*70)
    log_print(f"LSA complete. Results successfully saved to {OUTPUT_FILE}")
    log_print("="*70)

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
spark.stop()
