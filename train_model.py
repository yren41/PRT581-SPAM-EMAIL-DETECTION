from core_classify import *
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_model(filename):
    raw_df = read_data_from_csv_with_Y(filename)
    data = data_cleaning(raw_df)
    # data['spam'] = pd.to_numeric(data['spam'], errors='coerce')
    # data = data.dropna()
    one_hot_data = count_vectorize_with_vocab(data["text"])
    pca_data = pca_extract_feature(one_hot_data)

    X = pca_data[:4000]
    Y = data["spam"].tolist()[:4000]

    lg_model = LogisticRegression()
    lg_model.fit(X, Y)
    pred_y = lg_model.predict(X)

    print(accuracy_score(pred_y, Y))

    return lg_model


if __name__ == '__main__':
    model = train_model("emails.csv")
    with open("lg_model.pkl", "wb") as f:
        pickle.dump(model, f)
