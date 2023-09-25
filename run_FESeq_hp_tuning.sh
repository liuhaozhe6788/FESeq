#! /bin/sh
home="model_zoo"

export PYTHONPATH=${PWD}
echo ${PYTHONPATH}

cd $home/FESeq

echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_10_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_11_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_20_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_21_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_30_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_31_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_32_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_33_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_8_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_24_eleme
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_32_eleme

echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_10_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_11_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_20_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_21_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_30_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_31_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_32_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_33_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_8_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_24_bundle
echo "=== Training FESeq===" && python run_expid.py --gpu 0 --expid FESeq_3_32_bundle
