import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import gc

def create_sequences(row, window_size):
    sequences = []
    for i in range(0, min(window_size, len(row))):
        sequences.append('^'.join(map(str, row[: i])))
    while i+1<len(row):
        i+=1
        sequences.append('^'.join(map(str, row[i-window_size+1: i])))

    return sequences

def create_targets(row, window_size):
    targets = []
    for i in range(0, min(window_size, len(row))):
        targets.append(row[i])
    while i+1<len(row):
        i+=1
        targets.append(row[i])

    return targets

def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem, 100 * (start_mem - end_mem) / start_mem, (time.time() - starttime) / 60))
    gc.collect()
    return df

def split_train_val_test_by_time(df: pd.DataFrame, user: str, time: str):
    """
    Use last user record as test data.
    :param df:.
    :param user: user_col_name
    :param time: time_col_name
    :param seq: if the data should be processed as sequence
    :return: DataFrame
    """
    df = df.sort_values(by=time)
    rows = df.shape[0]
    th = int(rows * 0.8)
    th2 = int(rows * 0.9)
    train = df[:th]
    val = df[th:th2]
    test = df[th2:]
    return train, val, test


def split_train_val_test_by_user(df: pd.DataFrame, user: str, time: str):
    """
    Use last user record as test data.
    :param df:
    :param user: user_col_name
    :param time: time_col_name
    :return: DataFrame
    """
    grouped = df.sort_values(by=time).groupby(user)
    # 过滤用户行为数量小于等于3的用户
    grouped = grouped.filter(lambda x: x.shape[0] > 2)
    train = grouped.swifter.apply(lambda x: x[: -2])
    val = grouped.swifter.apply(lambda x: x[-2: -1])
    test = grouped.swifter.apply(lambda x: x[-1:])
    return train, val, test

def plot_user_click_count_his(df: pd.DataFrame, user: str, dataset_name: str, max_xlim: int):
    grouped = df.groupby(user)
    shapes = grouped.apply(lambda x: x.shape[0])
    from matplotlib import pyplot as plt
    plt.hist(shapes.values, bins='auto', edgecolor="r", histtype="step")
    plt.xlim(0, max_xlim)
    plt.savefig(f"{dataset_name}_user_click_count_his.png", dpi=500)