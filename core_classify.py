import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import re
import pickle


def load_dict(filename):
    with open(filename, "r") as f:
        return f.read().split("\n")


def load_lg_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def read_data_from_csv_with_Y(file):
    df = pd.read_csv(file)
    df = df.loc[:, ["text", "spam"]]
    return df


def read_data_from_csv(file):
    df = pd.read_csv(file)
    df = df.loc[:, ["text"]]
    return df


def data_cleaning(df):
    # convert all to str
    df = df.applymap(str)

    # Clear head text
    df["text"] = df["text"].apply(lambda x: x.replace("Subject: ", ""))

    # convert all alphabet to lower
    df["text"] = df["text"].apply(lambda x: x.lower())

    # replace URL related symbol to space
    # pat1 = re.compile(r'[:\\]+')
    # df["text"] = df["text"].apply(lambda x: re.sub(pat1, ' ', x))

    # delete everything but alphabet and space
    pat2 = re.compile(r'[^A-Za-z\s]+')
    df["text"] = df["text"].apply(lambda x: re.sub(pat2, '', x))

    df = df.dropna()

    return df


def count_vectorize_with_vocab(df):
    # Separate words into one-hot encoding
    spam_dict = load_dict("spam_dict.txt")
    vz = CountVectorizer(vocabulary=spam_dict)
    X = vz.fit_transform(list(df.to_numpy()))
    one_hot_df = pd.DataFrame(X.toarray(), columns=vz.get_feature_names())

    return one_hot_df


def count_vectorize(df):
    # Separate words into one-hot encoding
    vz = CountVectorizer()
    X = vz.fit_transform(list(df.to_numpy()))
    one_hot_df = pd.DataFrame(X.toarray(), columns=vz.get_feature_names())

    return one_hot_df


def pca_extract_feature(one_hot_df):
    # Feature Extraction
    one_hot_df = StandardScaler().fit_transform(one_hot_df)
    pca = PCA(n_components=50, random_state=42, svd_solver="full")
    pca_data = pca.fit_transform(one_hot_df)
    return pca_data


def hierarchical_clustering(pca_data):
    # Clustering
    clustering = AgglomerativeClustering(linkage="ward").fit(pca_data)
    return clustering.labels_


def gen_comparison_df(text_data, predict_data, actual_data):
    comp_df = pd.DataFrame()
    comp_df["text"] = text_data
    comp_df["pred_y"] = predict_data
    comp_df["actual"] = pd.to_numeric(actual_data, errors='coerce')
    return comp_df


def gen_acc_metric(comp_df):
    total = comp_df.shape[0]
    correct_count = len(comp_df[comp_df["pred_y"] == comp_df["actual"]])
    print(f"acc:{correct_count / total}")
    print(len(comp_df[comp_df["pred_y"] == 0]))
    print(len(comp_df[comp_df["pred_y"] == 1]))


def gen_result_df(pca_data: np.ndarray, text, pred_y):
    limit = 20

    rf_model: LogisticRegression
    rf_model = load_lg_model("lg_model.pkl")

    temp_df = pd.DataFrame()
    # temp_df["pca_data"] = pca_data
    temp_df["text"] = text
    temp_df["pred_y"] = pred_y

    # supervised learning based model classify cluster
    # cls0 = temp_df[temp_df["pred_y"] == 0][:limit]
    # cls0["rf_pred_y"] = rf_model.predict(np.array(cls0["pca_data"].tolist()))
    # actual_0 = len(cls0[cls0["rf_pred_y"] == 0])
    #
    # if actual_0 < (limit // 2):
    #     temp_df["pred_y"] = np.logical_xor(temp_df["pred_y"], 1)
    #
    # result_df = pd.DataFrame({"text": temp_df["text"], "spam": temp_df["pred_y"]})

    temp_df["rf_pred_y"] = rf_model.predict(pca_data.tolist())
    result_df = pd.DataFrame({"text": temp_df["text"], "spam": temp_df["rf_pred_y"]})

    return result_df


def classify_spam_with_Y(data_file):
    raw_df = read_data_from_csv_with_Y(data_file)
    data = data_cleaning(raw_df)
    # data['spam'] = pd.to_numeric(data['spam'], errors='coerce')
    # data = data.dropna()
    one_hot_data = count_vectorize_with_vocab(data["text"])
    pca_data = pca_extract_feature(one_hot_data)
    predict_y = hierarchical_clustering(pca_data)
    result_df = gen_result_df(pca_data, data["text"], predict_y)
    return result_df


def classify_spam(data_file):
    raw_df = read_data_from_csv(data_file)
    data = data_cleaning(raw_df)
    data = data.dropna()
    one_hot_data = count_vectorize_with_vocab(data["text"])
    pca_data = pca_extract_feature(one_hot_data)
    predict_y = hierarchical_clustering(pca_data)
    result_df = gen_result_df(pca_data, raw_df["text"], predict_y)
    return result_df


def acc_test():
    raw_df = read_data_from_csv_with_Y("emails.csv")
    result_df = classify_spam_with_Y("emails.csv")
    comp_df = gen_comparison_df(result_df["text"], result_df["spam"], raw_df["spam"])
    gen_acc_metric(comp_df)


if __name__ == '__main__':
    # acc_test()
    result_df: pd.DataFrame
    result_df = classify_spam("test.csv")
    result_df.to_csv("result.csv", index=False)
