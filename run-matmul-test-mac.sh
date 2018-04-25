#!/bin/bash
# Assumes that spark-submit is in your PATH

export NUM_ALCHEMIST_RANKS=2
export TMPDIR=/tmp

method=MATMUL
m=10000 
n=10000
k=10000

partitions=0

spark-submit --verbose\
  --master local[1] \
  --driver-memory 4G\
  --class amplab.alchemist.BasicSuite\
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar $method $m $n $k $partitions 2>&1 | tee test-matmul.log
exit
