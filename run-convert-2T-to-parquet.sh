#!/bin/bash
#SBATCH -q premium
#SBATCH -N 78
#SBATCH -t 02:00:00
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out
#SBATCH -C haswell
#module load collectl
#start-collectl.sh 

source setup/cori-start-alchemist.sh 39 2
sleep 15

infname=/global/cscratch1/sd/gittens/large-datasets/rda_ds093.0_dataset/outputs/ocean2T.h5
outfname=/global/cscratch1/sd/gittens/large-datasets/ocean2T.parquet
varname=rows

spark-submit --verbose\
  --driver-memory 120G\
  --executor-memory 120G\
  --executor-cores 32 \
  --driver-cores 32  \
  --num-executors 38 \
  --conf spark.driver.extraLibraryPath=$SCRATCH/alchemistSHELL/alchemist/lib\
  --conf spark.executor.extraLibraryPath=$SCRATCH/alchemistSHELL/alchemist/lib\
  --conf spark.eventLog.enabled=true\
  --conf spark.eventLog.dir=$SCRATCH/spark/event_logs\
  --class amplab.alchemist.ConvertH5ToParquet \
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar $infname $outfname $varname | tee convert-h5-to-parquet.log

stop-all.sh
exit
#stop-collectl.sh
