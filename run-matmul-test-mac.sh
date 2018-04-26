#!/bin/bash
# Assumes that spark-submit is in your PATH

export NUM_ALCHEMIST_RANKS=4
export TMPDIR=/tmp

method=MATMUL
m=200 
n=1000
k=500

partitions=0

spark-submit --verbose\
  --master local[2] \
  --driver-memory 4G\
  --class amplab.alchemist.BasicSuite\
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar $method $m $n $k $partitions 2>&1 | tee test-matmul.log
exit
