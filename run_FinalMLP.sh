#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}
cd $home/FinalMLP 

echo "=== Training DualMLP_eleme ===" && python run_expid.py --gpu 0 --expid DualMLP_eleme

echo "=== Training DualMLP_bundle ===" && python run_expid.py --gpu 0 --expid DualMLP_bundle

echo "=== Training DualMLP_ml10m ===" && python run_expid.py --gpu 0 --expid DualMLP_ml10m 

echo "=== Training FinalMLP_eleme ===" && python run_expid.py --gpu 0 --expid FinalMLP_eleme

echo "=== Training FinalMLP_bundle ===" && python run_expid.py --gpu 0 --expid FinalMLP_bundle

echo "=== Training FinalMLP_ml10m ===" && python run_expid.py --gpu 0 --expid FinalMLP_ml10m 