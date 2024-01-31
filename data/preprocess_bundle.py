import os.path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
import swifter

from utils import split_train_val_test_by_time, reduce_mem, plot_user_click_count_his

def read_processed_bundle_data(path: str,
                               to_path: str,
                               sparse_features: list,
                               dense_features: list,
                               usecols: list = None,
                               check_cols: list = None,
                               sep=','):
    """
    :param dense_features: 连续特征
    :param sparse_features: 类别特征
    :param path: data_path
    :param user: user_col_name
    :param time: time_col_name
    :param usecols: Use None to read all columns.
    :param check_cols: delete null rows
    :param sep:
    :return: Returns a DataFrameGroupy by uid.
    """
    if check_cols is None:
        check_cols = []
    df = pd.read_csv(path, sep=sep, usecols=usecols, nrows=None)
    print("loaded data of {} rows".format(df.shape[0]))
    for col in usecols:
        null_num = df[col].isnull().sum()
        if null_num > 0:
            print("There are {} nulls in {}.".format(null_num, col))
            df = df[~df[col].isnull()]
    print("buy:{}, unbuy:{}".format(df[df['impression_result'] == 1].shape[0], df[df['impression_result'] == 0].shape[0]))
    df['register_time'] = pd.to_datetime(df["register_time"]).swifter.apply(lambda x: int(x.timestamp()))

    df.sort_values(by=['ftime'], inplace=True)
    first_time = df.ftime[0]
    df["ftime"] = df["ftime"].swifter.apply(lambda x: x - first_time)

    # lbes = {feature: LabelEncoder() for feature in sparse_features}
    # for feature in sparse_features:
    #     df[feature] = lbes[feature].fit_transform(df[feature])
    # with open(to_path+"lbes.pkl", 'wb') as file:
    #     pickle.dump(lbes, file)
    # mms = MinMaxScaler()
    # df[dense_features] = mms.fit_transform(df[dense_features])
    # with open(to_path+"mms.pkl", 'wb') as file:
    #     pickle.dump(mms, file)

    # grouped = df.sort_values(by=[time]).groupby(user)
    # return grouped
    return df


def transform2sequences(grouped, default_col, sequence_col):
    """
    :param grouped:
    :param default_col:
    :param sequence_col: columns needed to generate sequences.
    :return: DataFrame
    """
    # TODO: to be updated
    df = pd.DataFrame(
        data={
            "uid": list(grouped.groups.keys()),
            **{col_name: grouped[col_name].apply(lambda x: x.iloc[0]) for col_name in default_col[1:]},
            **{col_name: grouped[col_name].apply(list) for col_name in sequence_col},
        }
    )
    return df



def gen_bundle_data():
    path = "./bundle/"
    if not os.path.exists(path):
        os.mkdir(path)
    user_default_col = ['uid', 'register_country', 'register_time', 'is_visitor', 'register_device',
                        'register_device_brand', 'register_os', 'gender']
    user_changeable_col = ['ftime', 'bundle_id', 'bundle_price', 'impression_result', 'island_no', 'spin_stock',
                           'coin_stock', 'diamond_stock', 'island_complete_gap_coin',
                           'island_complete_gap_building_cnt', 'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d',
                           'register_country_arpu', 'life_time', 'star', 'battle_pass_exp', 'is_up_to_date_version',
                           'pet_heart_stock', 'pet_exp_stock', 'friend_cnt', 'social_friend_cnt', 'pay_amt', 'pay_cnt',
                           'pay_per_day', 'pay_mean']
    spare_features = ['uid', 'is_visitor', 'gender', 'is_up_to_date_version', 'register_country', 'register_device',
                      'register_device_brand', 'register_os', 'bundle_id', 'ftime', 'register_time']
    dense_features = ['bundle_price', 'island_no', 'spin_stock', 'coin_stock',
                      'diamond_stock', 'island_complete_gap_coin', 'island_complete_gap_building_cnt',
                      'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d', 'register_country_arpu',
                      'life_time', 'star', 'battle_pass_exp', 'pet_heart_stock', 'pet_exp_stock', 'friend_cnt',
                      'social_friend_cnt', 'pay_amt', 'pay_cnt', 'pay_per_day', 'pay_mean']

    df = read_processed_bundle_data(path="./raw_data/ik/brs_daily_20211101_20211230.csv",
                                         sparse_features=spare_features,
                                         dense_features=dense_features,
                                         usecols=user_default_col + user_changeable_col,
                                         check_cols=['bundle_id'],
                                         to_path=path)

    # df = transform2sequences(grouped, user_default_col, user_changeable_col)

    # sequence_length = 8
    # for col_name in user_changeable_col:
    #     df[col_name] = df[col_name].swifter.apply(lambda x: create_fixed_sequences(x, sequence_length=sequence_length))

    # df = df.explode(column=user_changeable_col, ignore_index=True)
    # df['cur_time'] = df['ftime'].swifter.apply(lambda x: x[-1])

    df["pos"] = list(range(len(df)))[::]
    df = reduce_mem(df)
    for col in ["register_country", "register_device", "register_device_brand", "register_os"]:
        df[col] = df[col].astype(str)
    num_users = len(df.uid.unique())
    num_items = len(df.bundle_id.unique())
    num_interactions = df[['uid', 'bundle_id']].drop_duplicates().shape[0]
    num_rows = df.shape[0]
    sparsity = (1-num_interactions/(num_users*num_items))*100
    print(f"number of rows:{num_rows}")
    print(f"number of interactions:{num_interactions}")
    print(f"number of users:{num_users}")
    print(f"number of items:{num_items}")
    print(f"dataset sparcity:{sparsity}%")
    plot_user_click_count_his(df, 'uid', 'bundle', 200)
    train, val, test = split_train_val_test_by_time(df, 'uid', 'pos')
    train.drop(["pos"], axis=1, inplace=True)
    val.drop(["pos"], axis=1, inplace=True)
    test.drop(["pos"], axis=1, inplace=True)
    train.to_csv(path+"train_data.csv", index=False, sep=',')
    val.to_csv(path+"val_data.csv", index=False, sep=',')
    test.to_csv(path+"test_data.csv", index=False, sep=',')
    print("Done")


if __name__ == '__main__':
    gen_bundle_data()

