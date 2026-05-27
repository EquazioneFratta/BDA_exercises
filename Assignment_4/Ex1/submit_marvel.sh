#!/bin/bash -l
#SBATCH --job-name=marvel-graphx
#SBATCH --output=logs/marvel-graphx_%j.out
#SBATCH --error=logs/marvel-graphx_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --qos=normal

echo "Job started at: $(date) on $SLURM_NODELIST (ID: $SLURM_JOB_ID)"
mkdir -p "$SLURM_SUBMIT_DIR/logs"

module purge
module load env/development/2024a
module load devel/Spark/4.0.1-foss-2024a-Java-21

SCRIPT_ORIG="$SLURM_SUBMIT_DIR/RunMarvelGraphX.scala"
SCRIPT_RUN="$SLURM_SUBMIT_DIR/RunMarvelGraphX_run.scala"
DATA_DIR="$SLURM_SUBMIT_DIR/kaggle-marvel-universe"

echo "Submit dir : $SLURM_SUBMIT_DIR"
echo "Script     : $SCRIPT_RUN"
echo "Data dir   : $DATA_DIR"

if [ ! -f "$SCRIPT_ORIG" ]; then
  echo "ERROR: Script not found — aborting."; exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: Data directory not found — aborting."; exit 1
fi

cp "$SCRIPT_ORIG" "$SCRIPT_RUN"
sed -i "s|val DATA_PATH = \".*\"|val DATA_PATH = \"$DATA_DIR\"|" "$SCRIPT_RUN"

spark-shell \
  --master local[$SLURM_CPUS_PER_TASK] \
  --driver-memory 8g \
  --executor-memory 16g \
  --executor-cores 4 \
  --conf spark.driver.maxResultSize=4g \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  < "$SCRIPT_RUN"

echo "Job finished at: $(date) with exit code $?"