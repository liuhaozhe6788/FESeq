#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/TiSASRec 


echo "=== Training TiSASRec_eleme ===" && python run_expid.py --gpu 0 --expid TiSASRec_eleme

echo "=== Training TiSASRec_bundle ===" && python run_expid.py --gpu 0 --expid TiSASRec_bundle

echo "=== Training TiSASRec_ml10m ===" && python run_expid.py --gpu 0 --expid TiSASRec_ml10m