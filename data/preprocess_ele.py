import os.path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils import split_train_val_test_by_time, reduce_mem, plot_user_click_count_his
import swifter

def gen_ele_data():
    path = "eleme/"
    if not os.path.exists(path):
        os.mkdir(path)
    cols = ['label', 'user_id', 'gender', 'visit_city', 'avg_price', 'is_super_vip', 'ctr_30', 'ord_30', 'total_amt_30',

            'shop_id', 'item_id', 'city_id', 'district_id', 'shop_aoi_id', 'shop_geohash_6', 'shop_geohash_12',
            'brand_id', 'category_1_id', 'merge_standard_food_id', 'rank_7', 'rank_30', 'rank_90',

            'shop_id_list', 'item_id_list', 'category_1_id_list', 'merge_standard_food_id_list', 'brand_id_list',
            'price_list', 'shop_aoi_id_list', 'shop_geohash_6_list', 'timediff_list', 'hours_list', 'time_type_list',
            'weekdays_list',

            'times', 'hours', 'time_type', 'weekdays', 'geohash12']

    sparse_features = ['user_id', 'gender', 'visit_city', 'is_super_vip',

                      'shop_id', 'item_id', 'city_id', 'district_id', 'shop_aoi_id', 'shop_geohash_6', 'shop_geohash_12',
                      'brand_id', 'category_1_id', 'merge_standard_food_id','times','hours', 'time_type', 'weekdays', 'geohash12']

    dense_features = ['avg_price', 'ctr_30', 'ord_30', 'total_amt_30', 'rank_7', 'rank_30', 'rank_90']


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

    df = reduce_mem(df)

    num_users = len(df.user_id.unique())
    num_items = len(df.item_id.unique())
    num_interactions = df[['user_id', 'item_id']].drop_duplicates().shape[0]
    num_rows = df.shape[0]
    sparsity = (1-num_interactions/(num_users*num_items))*100
    print(f"number of rows:{num_rows}")
    print(f"number of interactions:{num_interactions}")
    print(f"number of users:{num_users}")
    print(f"number of items:{num_items}")
    print(f"dataset sparcity:{sparsity}%")
    plot_user_click_count_his(df, 'user_id', 'eleme', 30)

    train, val, test = split_train_val_test_by_time(df, 'user_id', 'times')
    train.to_csv(path+"train_data.csv", index=False, sep=',')
    val.to_csv(path+"val_data.csv", index=False, sep=',')
    test.to_csv(path+"test_data.csv", index=False, sep=',')


if __name__ == '__main__':
    gen_ele_data()