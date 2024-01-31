import os.path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils import split_train_val_test_by_time, reduce_mem, create_sequences, create_targets
import swifter
from tqdm import tqdm


def gen_ele_data():
    tqdm.pandas(desc="power DataFrame of Ele.me dataset!")
    path = "elemeseq/"
    if not os.path.exists(path):
        os.mkdir(path)

    sequence_cols = ['shop_id', 'item_id', 'city_id', 'district_id', 'shop_aoi_id', 'shop_geohash_6', 'shop_geohash_12',
         'category_1_id', 'rank_7', 'rank_30', 'rank_90', 'times', 'hours', 'time_type', 'weekdays', 'label'
    ]

    sequence_list_cols = [col+"_list" for col in sequence_cols]
    
    default_cols = ['user_id', 'gender', 'visit_city', 'avg_price', 'is_super_vip', 'ctr_30', 'ord_30', 'total_amt_30', 'geohash12'
    ]

    df = pd.read_csv("./raw_data/Ele_me/eleme.csv", sep=',', nrows=None)
    print(f"raw data rows: {df.shape[0]}")

    print("drop nan...")
    df = df[df.isna().sum(axis=1) == 0]
    print(f"remain rows: {df.shape[0]}")
    df = df[~(df["gender"]==-99)]
    df = df[~(df["gender"]=="-99")]
    print(f"remain rows: {df.shape[0]}")
    df = df[~(df["city_id"]==-1)]
    df = df[~(df["city_id"]=="-1")]
    print(f"remain rows: {df.shape[0]}")
    df = df[~(df["district_id"]==-1)]
    df = df[~(df["district_id"]=="-1")]
    print(f"remain rows: {df.shape[0]}")
    df = df[~(df["shop_aoi_id"]==-1)]
    df = df[~(df["shop_aoi_id"]=="-1")]
    print(f"remain rows: {df.shape[0]}")
    df = df[~(df["shop_geohash_6"]==-1)]
    df = df[~(df["shop_geohash_6"]=="-1")]
    print(f"remain rows: {df.shape[0]}")
    df = df[~(df["shop_geohash_12"]==-1)]
    df = df[~(df["shop_geohash_12"]=="-1")]
    print(f"remain rows: {df.shape[0]}")
    df = df[~(df["rank_7"]==-1)]
    df = df[~(df["rank_7"]=="-1")]
    print(f"remain rows: {df.shape[0]}")
    df = df[~(df["rank_30"]==-1)]
    df = df[~(df["rank_30"]=="-1")]
    print(f"remain rows: {df.shape[0]}")
    df = df[~(df["rank_90"]==-1)]
    df = df[~(df["rank_90"]=="-1")]
    print(f"remain rows: {df.shape[0]}")
    df = df[~(df["geohash12"]==-1)]
    df = df[~(df["geohash12"]=="-1")]
    print(f"remain rows: {df.shape[0]}")

    df.drop(['brand_id', 'merge_standard_food_id'], axis=1, inplace=True)

    grouped = df.sort_values(by=['times']).groupby('user_id')

    df = pd.DataFrame(
        data={
            "user_id": list(grouped.groups.keys()),
            **{col_name: grouped[col_name].apply(lambda x: x.iloc[0]) for col_name in default_cols[1:]},
            **{col_name: grouped[col_name].apply(list) for col_name in sequence_cols},
        }
    )
    del grouped

    sequence_length = 10 + 1
    for col_name in sequence_cols:
        df[col_name+ "_list"] = df[col_name].swifter.apply(lambda x: create_sequences(x, window_size=sequence_length))
        df[col_name] = df[col_name].swifter.apply(lambda x: create_targets(x, window_size=sequence_length))

    df = df.explode(column=sequence_cols+sequence_list_cols, ignore_index=True)

    df.drop(["label_list"], axis=1,inplace=True)

    df.sort_values(by=["times"], inplace=True)
    df["pos"] = list(range(len(df)))[::]
    df = reduce_mem(df) 
    train, val, test = split_train_val_test_by_time(df, 'user_id', 'pos')
    train.drop(["pos"], axis=1, inplace=True)
    val.drop(["pos"], axis=1, inplace=True)
    test.drop(["pos"], axis=1, inplace=True)

    train.to_csv(path+"train_data.csv", index=False, sep=',')
    val.to_csv(path+"val_data.csv", index=False, sep=',')
    test.to_csv(path+"test_data.csv", index=False, sep=',')


if __name__ == '__main__':
    gen_ele_data()