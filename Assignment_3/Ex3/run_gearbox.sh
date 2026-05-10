#!/bin/bash
#SBATCH --job-name=NASA_KMeans
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results.out

# Load the core 2024a environment and Spark
module load env/development/2024a
module load devel/Spark/4.0.1-foss-2024a-Java-21

# Run the script
spark-submit --master local[*] RunKMeans_Ex3.py
