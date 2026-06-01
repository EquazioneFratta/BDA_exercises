#!/bin/bash -l
#SBATCH -J bda-job
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=12:00:00
#SBATCH -p batch
#SBATCH --mem=0
#SBATCH -o %x-%j.log

### Load latest available Spark module
module load lang/Python/3.11.5-GCCcore-13.2.0
module load env/development/2024a
module load devel/Spark/4.0.1-foss-2024a-Java-21
export PYSPARK_PYTHON=/opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/broadwell/software/Python/3.11.5-GCCcore-13.2.0/bin/python3
export PYTHONPATH="/mnt/aiongpfs/users/mfrattarola/Ex4/pylibs:$PYTHONPATH"
export PYSPARK_DRIVER_PYTHON=/opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/broadwell/software/Python/3.11.5-GCCcore-13.2.0/bin/python3
### If you do not wish tmp dirs to be cleaned
### at the job end, set below to 0
export SPARK_CLEAN_TEMP=1

### INTERNAL SPARK CONFIGURATION

## CPU and memory settings
export SPARK_WORKER_CORES=${SLURM_CPUS_PER_TASK}

export DAEMON_MEM=4096  # MB
export NODE_MEM=$((${SLURM_MEM_PER_NODE}-${DAEMON_MEM}))

export SPARK_DAEMON_MEMORY=${DAEMON_MEM}m
export SPARK_NODE_MEM=${NODE_MEM}m
export SPARK_SUBMIT_OPTIONS="--packages org.apache.sedona:sedona-spark-shaded-3.5_2.12:1.9.0,org.datasyslab:geotools-wrapper:1.9.0-33.5 \
  --conf spark.executor.memory=${SPARK_NODE_MEM} \
  --conf spark.worker.memory=${SPARK_NODE_MEM} \
  --conf spark.driver.memory=20g \
  --conf spark.driver.maxResultSize=8g \
  --conf spark.kryoserializer.buffer.max=512m \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.kryo.registrator=org.apache.sedona.core.serde.SedonaKryoRegistrator"

## Set up job directories and environment variables
export SPARK_JOB_DIR="$HOME/spark-jobs"
export SPARK_JOB="$HOME/spark-jobs/${SLURM_JOBID}"
mkdir -p "${SPARK_JOB}"

export SPARK_HOME=$EBROOTSPARK
export SPARK_WORKER_DIR=${SPARK_JOB}
export SPARK_LOCAL_DIRS=${SPARK_JOB}
export SPARK_MASTER_PORT=7077
export SPARK_MASTER_WEBUI_PORT=9080
export SPARK_WORKER_WEBUI_PORT=9081
export SPARK_INNER_LAUNCHER=${SPARK_JOB}/spark-start-all.sh
export SPARK_MASTER_FILE=${SPARK_JOB}/spark_master

export HADOOP_HOME_WARN_SUPPRESS=1
export HADOOP_ROOT_LOGGER="WARN,DRFA"

## Generate spark starter-script
cat << 'EOF' > ${SPARK_INNER_LAUNCHER}
#!/bin/bash
## Load configuration and environment

source "$SPARK_HOME/sbin/spark-config.sh"
source "$SPARK_HOME/bin/load-spark-env.sh"

if [[ ${SLURM_PROCID} -eq 0 ]]; then
    ## Start master in background
    echo "MASTER USING SPARK_HOME="$SPARK_HOME
    export SPARK_MASTER_HOST=$(hostname)
    MASTER_NODE=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

    echo "spark://${SPARK_MASTER_HOST}:${SPARK_MASTER_PORT}" > "${SPARK_MASTER_FILE}"

    "${SPARK_HOME}/bin/spark-class" org.apache.spark.deploy.master.Master \
        --host $SPARK_MASTER_HOST                                           \
        --port $SPARK_MASTER_PORT                                         \
        --webui-port $SPARK_MASTER_WEBUI_PORT &

    echo "${SPARK_HOME}/bin/spark-class" org.apache.spark.deploy.master.Master \
        --host $SPARK_MASTER_HOST                                           \
        --port $SPARK_MASTER_PORT                                         \
        --webui-port $SPARK_MASTER_WEBUI_PORT > ${SPARK_JOB}/master_${SLURM_PROCID}.log

    ## Start one worker on same node as master
    echo "WORKER USING SPARK_HOME="$SPARK_HOME
    export SPARK_WORKER_CORES=$((${SLURM_CPUS_PER_TASK}))
    "${SPARK_HOME}/bin/spark-class" org.apache.spark.deploy.worker.Worker \
       --webui-port ${SPARK_WORKER_WEBUI_PORT}                             \
       spark://${SPARK_MASTER_HOST}:${SPARK_MASTER_PORT} &

    echo "${SPARK_HOME}/bin/spark-class" org.apache.spark.deploy.worker.Worker \
       --webui-port ${SPARK_WORKER_WEBUI_PORT}                             \
       spark://${SPARK_MASTER_HOST}:${SPARK_MASTER_PORT} > ${SPARK_JOB}/worker_${SLURM_PROCID}.log

    ## Wait for background tasks to complete
    wait
else
    ## Start one worker on each other node
    echo "WORKER USING SPARK_HOME="$SPARK_HOME
    export SPARK_MASTER_HOST=spark://$(scontrol show hostname ${SLURM_NODELIST} | head -n 1):${SPARK_MASTER_PORT}
    "${SPARK_HOME}/bin/spark-class" org.apache.spark.deploy.worker.Worker \
       --webui-port ${SPARK_WORKER_WEBUI_PORT}                             \
       ${SPARK_MASTER_HOST} &

    echo "${SPARK_HOME}/bin/spark-class" org.apache.spark.deploy.worker.Worker \
       --webui-port ${SPARK_WORKER_WEBUI_PORT}                             \
       ${SPARK_MASTER_HOST} > ${SPARK_JOB}/worker_${SLURM_PROCID}.log

    ## Wait for background tasks to complete
    wait
fi
EOF
chmod +x ${SPARK_INNER_LAUNCHER}

echo "=============================="
echo "SLURM NODE LIST:" ${SLURM_NODELIST}
echo "SLURM CPUs PER TASK:" ${SLURM_CPUS_PER_TASK}
echo "USING SPARK_HOME:" ${SPARK_HOME}
echo "RUNNING INNER LAUNCHER:" ${SPARK_INNER_LAUNCHER}
echo "=============================="

## Launch SPARK master and worker processes and wait for them to start
for i in $(scontrol show hostname ${SLURM_NODELIST});
do
    srun ${SPARK_INNER_LAUNCHER} &
done
while [ -z "$MASTER" ]; do
	sleep 5
	MASTER=$(cat "${SPARK_MASTER_FILE}")
done
### END OF INTERNAL CONFIGURATION

### ACTUAL USER CODE EXECUTION STARTS HERE
echo "=============================="
echo "LAUNCHING SPARK SUBMIT:" $SPARK_HOME/bin/spark-submit $SPARK_SUBMIT_OPTIONS --master $MASTER "${@:1}"
echo "=============================="

SPARK_PRINT_LAUNCH_COMMAND=true $SPARK_HOME/bin/spark-submit $SPARK_SUBMIT_OPTIONS --master $MASTER "${@:1}" # append any cmd args to the launcher!

### FINAL CLEANUP
if [[ -n "${SPARK_CLEAN_TEMP}" && ${SPARK_CLEAN_TEMP} -eq 1 ]]; then
    echo "====== Cleaning up: SPARK_CLEAN_TEMP=${SPARK_CLEAN_TEMP}"
    rm -rf ${SPARK_JOB_DIR}
else
    echo "====== Not cleaning up: SPARK_CLEAN_TEMP=${SPARK_CLEAN_TEMP}"
fi
