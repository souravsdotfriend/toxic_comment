import pandas as pd, numpy as np, re
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

data_train = pd.read_csv('data/train.csv', sep=',')
data_test = pd.read_csv('data/test.csv', sep=',')
print(data_train.shape)
print(data_test.shape)

# data_train.append(data_test)
# data_train = pd.concat([data_train, data_test], axis=0)
# print(data_train.shape)

# nb_validation_samples = int(0.2 * data_train.shape[0])
# data_val = data_train[-nb_validation_samples:]
# data_train = data_train[:-nb_validation_samples]
# # pd.concat([data_train, data_val])
# print(data_train.shape)
# print(data_val.shape)
# data_train.to_csv('data/train1.csv', sep=',')
# data_val.to_csv('data/val1.csv', sep=',')

# print(train.head())
print(data_train.columns)
print(data_train.shape)
columnList = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def clean_str(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

ids = []
texts = []
toxic_labels = []
severe_toxic_labels = []
obscene_labels = []
threat_labels = []
insult_labels = []
identity_hate_labels = []

for idx in range(data_train.id.shape[0]):
    ids.append(data_train.id[idx])
    texts.append(clean_str(data_train.comment_text[idx]))
    toxic_labels.append(data_train.toxic[idx])
    severe_toxic_labels.append(data_train.severe_toxic[idx])
    obscene_labels.append(data_train.obscene[idx])
    threat_labels.append(data_train.threat[idx])
    insult_labels.append(data_train.insult[idx])
    identity_hate_labels.append(data_train.identity_hate[idx])


# toxic_labels = to_categorical(np.asarray(toxic_labels))
# print('^^^^^^^^^^^^ ', toxic_labels.sum(axis=0))
# print(toxic_labels.shape)

for idx in range(data_test.id.shape[0]):
    ids.append(data_test.id[idx])
    texts.append(clean_str(data_test.comment_text[idx]))

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=1000)

toxic_labels = to_categorical(np.asarray(toxic_labels))
severe_toxic_labels = to_categorical(np.asarray(severe_toxic_labels))
obscene_labels = to_categorical(np.asarray(obscene_labels))
threat_labels = to_categorical(np.asarray(threat_labels))
insult_labels = to_categorical(np.asarray(insult_labels))
identity_hate_labels = to_categorical(np.asarray(identity_hate_labels))
print('Shape of data tensor:', data.shape)
print('Shape of toxic label tensor:', toxic_labels.shape)

# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# toxic_labels = toxic_labels[indices]
# nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
# x_train = data[:-nb_validation_samples]
# y_train = labels[:-nb_validation_samples]
# x_val = data[-nb_validation_samples:]
# y_val = labels[-nb_validation_samples:]

x_train, x_val, x_test = np.split(data, [int(0.8 * 159571), 159571])
id_train, id_val, id_test = np.split(ids, [int(0.8 * 159571), 159571])
y_train_toxic, y_val_toxic = np.split(toxic_labels, [int(0.8 * 159571)])

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

print(y_train_toxic.shape)
print(y_val_toxic.shape)
# print(y_test_toxic.shape)


# y_train_severe_toxic, y_val_severe_toxic, y_test_severe_toxic = np.split(severe_toxic_labels, [int(0.6 * len(data)),
#                                                                                                int(0.8 * len(
#                                                                                                    data))])
# y_train_obscene, y_val_obscene, y_test_obscene = np.split(obscene_labels,
#                                                           [int(0.6 * len(data)), int(0.8 * len(data))])
# y_train_threat, y_val_threat, y_test_threat = np.split(threat_labels, [int(0.6 * len(data)), int(0.8 * len(data))])
# y_train_insult, y_val_insult, y_test_insult = np.split(insult_labels, [int(0.6 * len(data)), int(0.8 * len(data))])
# y_train_identity_hate, y_val_identity_hate, y_test_identity_hate = np.split(identity_hate_labels,
#                                                                             [int(0.6 * len(data)),
#                                                                              int(0.8 * len(data))])



    # x_train, x_val, x_test = np.split(data, [int(0.6*len(data)), int(0.8*len(data))])
# y_train, y_val, y_test = np.split(labels, [int(0.6*len(data)), int(0.8*len(data))])
# id_train, id_val, id_test = np.split(ids, [int(0.6*len(data)), int(0.8*len(data))])