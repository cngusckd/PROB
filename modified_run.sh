#!/bin/bash

GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 configs/modified_M_OWOD_BENCHMARK.sh | tee -a 20250620_1.log