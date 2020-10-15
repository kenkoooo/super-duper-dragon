#!/bin/bash

cd $(dirname $0)

cargo run --bin policy_player --release -- --model-filepath ./train_policy_2020_check_point.bin
