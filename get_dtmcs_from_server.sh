#!/bin/bash
FOLDER_NAME=max50_random_3
mkdir data/$FOLDER_NAME
mkdir data/$FOLDER_NAME/raw
scp papavero.dinfo.unifi.it:tesi/generator/dtmc_config* data/$FOLDER_NAME/
scp -r papavero.dinfo.unifi.it:tesi/generator/dtmcs data/$FOLDER_NAME/raw/
scp -r papavero.dinfo.unifi.it:tesi/generator/labels data/$FOLDER_NAME/raw/
#conda activate dtmc_env
#python preprocess.py data/$FOLDER_NAME
#cd linear || return
#python main.py train ../data/$FOLDER_NAME