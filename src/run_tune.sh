#!/usr/bin/env bash

export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export XLA_USE_BF16=1

python3 tune.py "$@"