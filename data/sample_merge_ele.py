import os.path
import pickle

import numpy as np
import pandas as pd

cols = ['label', 'user_id', 'gender', 'visit_city', 'avg_price', 'is_super_vip', 'ctr_30', 'ord_30', 'total_amt_30',

        'shop_id', 'item_id', 'city_id', 'district_id', 'shop_aoi_id', 'shop_geohash_6', 'shop_geohash_12',
        'brand_id', 'category_1_id', 'merge_standard_food_id', 'rank_7', 'rank_30', 'rank_90',

        'shop_id_list', 'item_id_list', 'category_1_id_list', 'merge_standard_food_id_list', 'brand_id_list',
        'price_list', 'shop_aoi_id_list', 'shop_geohash_6_list', 'timediff_list', 'hours_list', 'time_type_list',
        'weekdays_list',

        'times', 'hours', 'time_type', 'weekdays', 'geohash12']
list_cols = ['shop_id_list', 'item_id_list', 'category_1_id_list', 'merge_standard_food_id_list', 'brand_id_list',
        'price_list', 'shop_aoi_id_list', 'shop_geohash_6_list', 'timediff_list', 'hours_list', 'time_type_list',
        'weekdays_list']

df_1 = pd.read_csv("./raw_data/Ele_me/D1_0.csv", names=cols, sep=',',nrows=None)
df_1.drop(list_cols, axis=1, inplace=True)
print("read df_1 finished!")
df_2 = pd.read_csv("./raw_data/Ele_me/D3_1.csv", names=cols, sep=',',nrows=None)
df_2.drop(list_cols, axis=1, inplace=True)
print("read df_2 finished!")
df_3 = pd.read_csv("./raw_data/Ele_me/D5_0.csv", names=cols, sep=',',nrows=None)
df_3.drop(list_cols, axis=1, inplace=True)
print("read df_3 finished!")
sampled_df = pd.concat([df_1.sample(int(len(df_1)/3)), df_2.sample(int(len(df_2)/3)), df_3.sample(int(len(df_3)/3))], ignore_index=True)
print("merge dfs finished!")

sampled_df.to_csv("./raw_data/Ele_me/eleme.csv", index=False, sep=',')