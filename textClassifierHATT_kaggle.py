# author - Richard Liao 
# Dec 26 2016
import os
import re
import pickle
import numpy as np
import pandas as pd
from keras.engine import Layer
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.layers import Dense, Input, GRU, initializers, K
from keras.layers import Embedding, LSTM, Bidirectional, TimeDistributed
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from nltk import tokenize

os.environ['KERAS_BACKEND'] = 'theano'


MAX_SENT_LENGTH = 100
MAX_SENTS = 15
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
reviews = []
toxic_labels = []
severe_toxic_labels = []
obscene_labels = []
threat_labels = []
insult_labels = []
identity_hate_labels = []

for idx in range(data_train.id.shape[0]):
    ids.append(data_train.id[idx])
    text = clean_str(data_train.comment_text[idx])
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)
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
data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                try:
                    if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                        data[i, j, k] = tokenizer.word_index[word]
                        k = k + 1
                except:
                    print(word)

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

toxic_labels = to_categorical(np.asarray(toxic_labels))
severe_toxic_labels = to_categorical(np.asarray(severe_toxic_labels))
obscene_labels = to_categorical(np.asarray(obscene_labels))
threat_labels = to_categorical(np.asarray(threat_labels))
insult_labels = to_categorical(np.asarray(insult_labels))
identity_hate_labels = to_categorical(np.asarray(identity_hate_labels))
print('Shape of data tensor:', data.shape)
print('Shape of toxic label tensor:', toxic_labels.shape)
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

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

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(LSTM(200, dropout=0.3))(embedded_sequences)
sentEncoder = Model(sentence_input, l_lstm)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(LSTM(100, dropout=0.4))(review_encoder)
temp1 = Dense(100, activation='relu')(l_lstm_sent)
temp2 = Dense(100, activation='relu')(temp1)
preds = Dense(2, activation='softmax')(temp2)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# print("model fitting - Hierachical LSTM")
# print(model.summary())
#
# MODEL_P = 'models/hier_lstm_toxic.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_toxic, validation_data=(x_val, y_val_toxic),
#           epochs=1, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# # scores = model.evaluate(x_test, y_test_toxic)
# # print("test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/hier_lstm_toxic.out", "wb")
# pickle.dump(predictions, file)


MODEL_P = 'models/hier_lstm_severe_toxic.h5'
model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
model.fit(x_train, y_train_severe_toxic, validation_data=(x_val, y_val_severe_toxic),
          epochs=1, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
model = load_model(filepath=MODEL_P)
# scores = model.evaluate(x_test, y_test_severe_toxic)
# print("test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
predictions = model.predict(data)
file = open("output/hier_lstm_severe_toxic.out", "wb")
pickle.dump(predictions, file)

# MODEL_P = 'models/hier_lstm_obscene.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_obscene, validation_data=(x_val, y_val_obscene),
#           epochs=1, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# scores = model.evaluate(x_test, y_test_obscene)
# print("test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/hier_lstm_obscene.out", "wb")
# pickle.dump(predictions, file)
#
# MODEL_P = 'models/hier_lstm_threat.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_threat, validation_data=(x_val, y_val_threat),
#           epochs=1, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# scores = model.evaluate(x_test, y_test_threat)
# print("test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/hier_lstm_threat.out", "wb")
# pickle.dump(predictions, file)
#
# MODEL_P = 'models/hier_lstm_insult.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_insult, validation_data=(x_val, y_val_insult),
#           epochs=1, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# scores = model.evaluate(x_test, y_test_insult)
# print("test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/hier_lstm_insult.out", "wb")
# pickle.dump(predictions, file)
#
# MODEL_P = 'models/hier_lstm_identity_hate.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_identity_hate, validation_data=(x_val, y_val_identity_hate),
#           epochs=1, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P)
# scores = model.evaluate(x_test, y_test_identity_hate)
# print("test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/hier_lstm_identity_hate.out", "wb")
# pickle.dump(predictions, file)


# building Hierachical Attention network
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        self.W = K.variable(self.init((input_shape[-1],)))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(200, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(200))(l_lstm)
l_att = AttLayer()(l_dense)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
l_att_sent = AttLayer()(l_dense_sent)
preds = Dense(2, activation='softmax')(l_att_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical attention network")
model.summary()

MODEL_P = 'models/hier_att_toxic.h5'
model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
model.fit(x_train, y_train_toxic, validation_data=(x_val, y_val_toxic),
          epochs=1, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
model = load_model(filepath=MODEL_P, custom_objects={'AttLayer': AttLayer})
# scores = model.evaluate(x_test, y_test_toxic)
# print("test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
predictions = model.predict(data)
file = open("output/hier_att.out", "wb")
pickle.dump(predictions, file)

# MODEL_P = 'models/hier_att_severe_toxic.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_severe_toxic, validation_data=(x_val, y_val_severe_toxic),
#           epochs=1, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P, custom_objects={'AttLayer': AttLayer})
# scores = model.evaluate(x_test, y_test_severe_toxic)
# print("test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/hier_att_severe_toxic.out", "wb")
# pickle.dump(predictions, file)

MODEL_P = 'models/hier_att_obscene.h5'
model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
model.fit(x_train, y_train_obscene, validation_data=(x_val, y_val_obscene),
          epochs=1, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
model = load_model(filepath=MODEL_P, custom_objects={'AttLayer': AttLayer})
# scores = model.evaluate(x_test, y_test_obscene)
# print("test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
predictions = model.predict(data)
file = open("output/hier_att_obscene.out", "wb")
pickle.dump(predictions, file)

# MODEL_P = 'models/hier_att_threat.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_threat, validation_data=(x_val, y_val_threat),
#           epochs=1, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P, custom_objects={'AttLayer': AttLayer})
# scores = model.evaluate(x_test, y_test_threat)
# print("test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/hier_att_threat.out", "wb")
# pickle.dump(predictions, file)

MODEL_P = 'models/hier_att_insult.h5'
model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
model.fit(x_train, y_train_insult, validation_data=(x_val, y_val_insult),
          epochs=1, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
model = load_model(filepath=MODEL_P, custom_objects={'AttLayer': AttLayer})
# scores = model.evaluate(x_test, y_test_insult)
# print("test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
predictions = model.predict(data)
file = open("output/hier_att_insult.out", "wb")
pickle.dump(predictions, file)

# MODEL_P = 'models/hier_att_identity_hate.h5'
# model_chk = ModelCheckpoint(filepath=MODEL_P, monitor='val_acc', save_best_only=True, verbose=1)
# model.fit(x_train, y_train_identity_hate, validation_data=(x_val, y_val_identity_hate),
#           epochs=1, batch_size=64, callbacks=[model_chk], verbose=1, shuffle=False)
# model = load_model(filepath=MODEL_P, custom_objects={'AttLayer': AttLayer})
# scores = model.evaluate(x_test, y_test_identity_hate)
# print("test-set %s: %.2f%%" % (model.metrics_names[1], scores[1]*100) )
# predictions = model.predict(data)
# file = open("output/hier_att_identity_hate.out", "wb")
# pickle.dump(predictions, file)
