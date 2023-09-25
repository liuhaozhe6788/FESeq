#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/TiSASRec 


echo "=== Training TiSASRec ===" && python run_expid.py --gpu 0 --expid TiSASRec_eleme

echo "=== Training TiSASRec ===" && python run_expid.py --gpu 0 --expid TiSASRec_bundle
