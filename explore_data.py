from core_classify import *

with open("stopwords.txt", "r") as f:
    stopwords = f.read().split("\n")

raw_df = read_data_from_csv_with_Y("emails.csv")
data = data_cleaning(raw_df)
data['spam'] = pd.to_numeric(data['spam'], errors='coerce')
# data = data.loc[data["spam"] == 1]
data = data.dropna()
one_hot_data = count_vectorize(data["text"])

found_stop_words = np.intersect1d(stopwords, one_hot_data.columns)
hot_spam_words = one_hot_data.drop(found_stop_words, axis=1)

hot_word = hot_spam_words.sum().sort_values(ascending=False)
dict = hot_word.head(1000).index.tolist()

with open("spam_dict.txt", "w+") as f:
    f.write("\n".join(dict))
