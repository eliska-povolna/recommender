import pickle
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# Load the same dataset used by the EasyStudy fastcompare plugin
from EasyStudy.server.plugins.utils.data_loading import load_ml_dataset

if __name__ == "__main__":
    loader = load_ml_dataset()
    df = loader.ratings_df

    # Map original ids to contiguous indices
    user_to_index = loader.user_to_user_index
    item_to_index = loader.movie_id_to_index

    rows = df['userId'].map(user_to_index).values
    cols = df['movieId'].map(item_to_index).values
    data = (df['rating'] >= 4.0).astype('float32').values

    num_users = len(user_to_index)
    num_items = len(item_to_index)
    X = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

    all_users = list(range(num_users))
    train_users, test_users = train_test_split(all_users, test_size=0.2, random_state=42)

    X_train = X[train_users]
    X_test = X[test_users]

    with open('data/processed_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('data/processed_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open('data/item2index.pkl', 'wb') as f:
        pickle.dump(item_to_index, f)
