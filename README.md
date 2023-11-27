# Feature Interaction-Enhanced Sequential Transformer for Click-Through Rate Prediction

This is the PyTorch implementation for the paper Feature Interaction-Enhanced Sequential Transformer for Click-Through Rate Prediction based on [FuxiCTR](https://github.com/xue-pai/FuxiCTR). 

![alt text](docs/img/FESeq.png?raw=true)

## Datasets

### Ele.me Dataset

The original Ele.me dataset is from this [link](https://tianchi.aliyun.com/dataset/131047). However, many data files are corrupted. We extract and sample from D1_0, D3_1, and D5_0 csv files only. The sampled Ele.me dataset can be downloaded from this [link](https://drive.google.com/drive/folders/1azJt4ZbKOYeO8wDT-M06rUx09WRckWaG). 

### Bundle Dataset

It cannot be shared openly but is available on request from authors.

### Data Preprocessing

To preprocess data for feature interaction models:

```
cd data 
python preprocess_<dataset_name>.py
```
To preprocess data for sequence recommendation and unified models:

```
cd data 
python preprocess_<dataset_name>seq.py
```
## Model Training
To train a baseline:
```
./run_<baseline_name>.sh
```
To reproduce the ablation studies results:
```
./run_FESeq_abl_study.sh
```
To reproduce the hyperparameter tuning results:
```
./run_FESeq_hp_tuning.sh
```
