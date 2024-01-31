import os
import pandas as pd
from utils import split_train_val_test_by_time, reduce_mem, plot_user_click_count_his
import swifter

def gen_ml10m_data():
    path = "ml_10m/"
    if not os.path.exists(path):
        os.mkdir(path)
    root_path = './raw_data/'

    ratings = pd.read_csv(
        root_path + "ml_10m/ratings.dat",
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        root_path + "ml_10m/movies.dat",
        sep="::", 
        names=["movie_id", "title", "genres"], 
        encoding="ISO-8859-1"
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

    num_users = len(data.user_id.unique())
    num_items = len(data.movie_id.unique())
    num_interactions = data[['user_id', 'movie_id']].drop_duplicates().shape[0]
    num_rows = data.shape[0]
    # sparsity = (1-num_interactions/(num_users*num_items))*100
    print(f"number of rows:{num_rows}")
    print(f"number of interactions:{num_interactions}")
    print(f"number of users:{num_users}")
    print(f"number of items:{num_items}")
    # print(f"dataset sparcity:{sparsity}%")
    plot_user_click_count_his(data, 'user_id', 'movielens-10m', 1000)

    train, val, test = split_train_val_test_by_time(data, 'user_id', 'timestamp')

    train.to_csv(path+"train_data.csv", index=False, sep=',')
    val.to_csv(path+"val_data.csv", index=False, sep=',')
    test.to_csv(path+"test_data.csv", index=False, sep=',')


if __name__ == '__main__':
    gen_ml10m_data()