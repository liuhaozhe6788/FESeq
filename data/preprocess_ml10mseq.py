import os
import pandas as pd
from utils import split_train_val_test_by_time, reduce_mem, create_sequences, create_targets
import swifter

def gen_ml10mseq_data():
    path = "ml_10mseq/"
    if not os.path.exists(path):
        os.mkdir(path)
    root_path = './raw_data/'

    ratings = pd.read_csv(
        root_path + "ml_10m/ratings.dat",
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        nrows=None
    )

    movies = pd.read_csv(
        root_path + "ml_10m/movies.dat",
        sep="::", 
        names=["movie_id", "title", "genres"], 
        nrows=None
    )

    ## Movies
    movies["year"] = movies["title"].swifter.apply(lambda x: x[-5:-1])

    genres = [
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film_Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci_Fi",
        "Thriller",
        "War",
        "Western",
    ]
    for genre in genres:
        movies[genre] = movies["genres"].apply(
            lambda values: int(genre in values.split("|"))
        )

    data = pd.merge(ratings, movies, on='movie_id')
    data['label'] = (data.rating > 3).astype('int')

    data.drop(['rating', "title", "genres"], axis=1, inplace=True)
    data = reduce_mem(data)

    sequence_col = ["movie_id", "year", "timestamp", "label"] + genres

    sequence_list_col = [col+"_list" for col in sequence_col]

    grouped = data.sort_values(by=['timestamp']).groupby('user_id')

    df = pd.DataFrame(
        data={
            "user_id": list(grouped.groups.keys()),
            **{col_name: grouped[col_name].apply(list) for col_name in sequence_col},
        }
    )
    del grouped

    sequence_length = 10 + 1
    for col_name in sequence_col:
        df[col_name+ "_list"] = df[col_name].swifter.apply(lambda x: create_sequences(x, window_size=sequence_length))
        df[col_name] = df[col_name].swifter.apply(lambda x: create_targets(x, window_size=sequence_length))

    df = df.explode(column=sequence_col+sequence_list_col, ignore_index=True)

    df.drop(["label_list"], axis=1,inplace=True)

    df.sort_values(by=["timestamp"], inplace=True)

    train, val, test = split_train_val_test_by_time(df, 'user_id', 'timestamp')

    train.to_csv(path+"train_data.csv", index=False, sep=',')
    val.to_csv(path+"val_data.csv", index=False, sep=',')
    test.to_csv(path+"test_data.csv", index=False, sep=',')


if __name__ == '__main__':
    gen_ml10mseq_data()