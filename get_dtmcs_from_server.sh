#!/bin/bash
FOLDER_NAME=only50_random_30-50_3full
#mkdir data/$FOLDER_NAME
#mkdir data/$FOLDER_NAME/raw
#scp giorgi@192.168.3.47:generator/dtmc_network/data/dtmc_config_only50* data/$FOLDER_NAME/
#scp -r giorgi@192.168.3.47:generator/dtmc_network/data/dtmcs data/$FOLDER_NAME/raw
#conda activate dtmc_env
#python preprocess.py data/$FOLDER_NAME
cd linear || return
python main.py ../data/$FOLDER_NAME