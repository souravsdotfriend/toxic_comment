# author - Richard Liao
# Dec 26 2016
import os
import re
import pickle
import numpy as np
import pandas as pd
from keras.utils import to_categorical

os.environ['KERAS_BACKEND'] = 'theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

def clean_str(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

data_train = pd.read_csv('data/train.csv', sep=',')
print(data_train.shape)

data_test = pd.read_csv('data/test.csv', sep=',')
print(data_test.shape)

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

for idx in range(data_test.id.shape[0]):
    ids.append(data_test.id[idx])
    texts.append(clean_str(data_test.comment_text[idx]))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

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
y_train_severe_toxic, y_val_severe_toxic = np.split(severe_toxic_labels, [int(0.8 * 159571)])
y_train_obscene, y_val_obscene = np.split(obscene_labels, [int(0.8 * 159571)])
y_train_threat, y_val_threat = np.split(threat_labels, [int(0.8 * 159571)])
y_train_insult, y_val_insult = np.split(insult_labels, [int(0.8 * 159571)])
y_train_identity_hate, y_val_identity_hate = np.split(identity_hate_labels, [int(0.8 * 159571)])

print('Number of toxic text in training and validation set ')
print(y_train_toxic.sum(axis=0))
print(y_val_toxic.sum(axis=0))
print('Number of severe toxic text in training and validation set ')
print(y_train_severe_toxic.sum(axis=0))
print(y_val_severe_toxic.sum(axis=0))
print('Number of obscene text in training and validation set ')
print(y_train_obscene.sum(axis=0))
print(y_val_obscene.sum(axis=0))
print('Number of threat text in training and validation set ')
print(y_train_threat.sum(axis=0))
print(y_val_threat.sum(axis=0))
print('Number of insult text in training and validation set ')
print(y_train_insult.sum(axis=0))
print(y_val_insult.sum(axis=0))
print('Number of identity hate text in training and validation set ')
print(y_train_identity_hate.sum(axis=0))
print(y_val_identity_hate.sum(axis=0))

GLOVE_DIR = "glove.6B"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in Glove 6B 200d.' % len(embeddings_index))
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(5)(l_cov2)
l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
l_flat = Flatten()(l_pool3)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

print("model fitting - simplified convolutional neural network")
model.summary()

# MODEL_P = 'models/cnn_toxic.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_toxic, validation_data=(x_val, y_val_toxic),
#           epochs=3, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# # scores = model.evaluate(x_test, y_test_toxic)
# # print("\ntoxic test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/cnn_toxic.out", "wb")
# pickle.dump(predictions, file)

MODEL_P = 'models/cnn_severe_toxic.h5'
model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
model.fit(x_train, y_train_severe_toxic, validation_data=(x_val, y_val_severe_toxic),
          epochs=3, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
model = load_model(filepath=MODEL_P)
# scores = model.evaluate(x_test, y_test_severe_toxic)
# print("\n severe toxictest-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
predictions = model.predict(data)
file = open("output/cnn_severe_toxic.out", "wb")
pickle.dump(predictions, file)

# MODEL_P = 'models/cnn_obscene.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_obscene, validation_data=(x_val, y_val_obscene),
#           epochs=3, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# # scores = model.evaluate(x_test, y_test_obscene)
# # print("\n obscene test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/cnn_obscene.out", "wb")
# pickle.dump(predictions, file)
#
# MODEL_P = 'models/cnn_threat.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_threat, validation_data=(x_val, y_val_threat),
#           epochs=3, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# # scores = model.evaluate(x_test, y_test_threat)
# # print("\n threat test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/cnn_threat.out", "wb")
# pickle.dump(predictions, file)
#
# MODEL_P = 'models/cnn_insult.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_insult, validation_data=(x_val, y_val_insult),
#           epochs=3, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# # scores = model.evaluate(x_test, y_test_insult)
# # print("\n insult test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/cnn_insult.out", "wb")
# pickle.dump(predictions, file)
#
# MODEL_P = 'models/cnn_identity_hate.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_identity_hate, validation_data=(x_val, y_val_identity_hate),
#           epochs=3, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# # scores = model.evaluate(x_test, y_test_identity_hate)
# # print("\n identity hate test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/cnn_identity_hate.out", "wb")
# pickle.dump(predictions, file)

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

# applying a more complex convolutional approach
convs = []
filter_sizes = [3, 4, 5]

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

for fsz in filter_sizes:
    l_conv = Conv1D(nb_filter=128, filter_length=fsz, activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    convs.append(l_pool)

l_merge = Merge(mode='concat', concat_axis=1)(convs)
l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(30)(l_cov2)
l_flat = Flatten()(l_pool2)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - more complex convolutional neural network")
model.summary()

# MODEL_P = 'models/cnn_complex_toxic.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_toxic, validation_data=(x_val, y_val_toxic),
#           nb_epoch=3, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# # scores = model.evaluate(x_test, y_test_toxic)
# # print("\n toxic test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/cnn_complex_toxic.out", "wb")
# pickle.dump(predictions, file)

MODEL_P = 'models/cnn_complex_severe_toxic.h5'
model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
model.fit(x_train, y_train_severe_toxic, validation_data=(x_val, y_val_severe_toxic),
          nb_epoch=3, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
model = load_model(filepath=MODEL_P)
# scores = model.evaluate(x_test, y_test_severe_toxic)
# print("\n severe toxic test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
predictions = model.predict(data)
file = open("output/cnn_complex_severe_toxic.out", "wb")
pickle.dump(predictions, file)

# MODEL_P = 'models/cnn_complex_obscene.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_obscene, validation_data=(x_val, y_val_obscene),
#           nb_epoch=3, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# # scores = model.evaluate(x_test, y_test_obscene)
# # print("\n obscene test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/cnn_complex_obscene.out", "wb")
# pickle.dump(predictions, file)
#
# MODEL_P = 'models/cnn_complex_threat.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_threat, validation_data=(x_val, y_val_threat),
#           nb_epoch=3, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# # scores = model.evaluate(x_test, y_test_threat)
# # print("\n threat test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/cnn_complex_threat.out", "wb")
# pickle.dump(predictions, file)
#
# MODEL_P = 'models/cnn_complex_insult.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_insult, validation_data=(x_val, y_val_insult),
#           nb_epoch=3, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# # scores = model.evaluate(x_test, y_test_insult)
# # print("\n threat test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/cnn_complex_insult.out", "wb")
# pickle.dump(predictions, file)
#
# MODEL_P = 'models/cnn_complex_identity_hate.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_identity_hate, validation_data=(x_val, y_val_identity_hate),
#           nb_epoch=3, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# # scores = model.evaluate(x_test, y_test_identity_hate)
# # print("identity hate test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/cnn_complex_identity_hate.out", "wb")
# pickle.dump(predictions, file)