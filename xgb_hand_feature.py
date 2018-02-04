import pickle

import en_core_web_sm
import numpy as np
import pandas as pd
import xgboost as xgb
import time

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelBinarizer

train_df = pd.read_csv("data/train.csv")
train_df['text'] = train_df['comment_text']

# spacy_nlp = en_core_web_sm.load()
#
# # change ne to tag
# def get_spacy_text(s):
#     pos, tag, dep = '', '', ''
#     for token in spacy_nlp(s):
#         pos = pos + ' ' + token.pos_
#         tag = tag + ' ' + token.tag_
#         dep = dep + ' ' + token.dep_
#
#     return pos, tag, dep
#
# poss, tags, deps = [], [], []
# for s in train_df["text"].values:
#     pos, tag, dep = get_spacy_text(s)
#     poss.append(pos)
#     tags.append(tag)
#     deps.append(dep)
# train_df['pos_txt'], train_df['tag_txt'], train_df['dep_txt'] = poss, tags, deps
#
# # cnt on tag
# c_vec3 = CountVectorizer(lowercase=False,ngram_range=(1,1))
# c_vec3.fit(train_df['tag_txt'].values.tolist() )
# train_cvec3 = c_vec3.transform(train_df['tag_txt'].values.tolist()).toarray()
# # test_cvec3 = c_vec3.transform(test_df['tag_txt'].values.tolist()).toarray()
# print(train_cvec3.shape)
#
# # cnt on ne
# c_vec4 = CountVectorizer(lowercase=False,ngram_range=(1,2))
# c_vec4.fit(train_df['pos_txt'].values.tolist() )
# train_cvec4 = c_vec4.transform(train_df['pos_txt'].values.tolist()).toarray()
# # test_cvec4 = c_vec4.transform(test_df['pos_txt'].values.tolist()).toarray()
# print(train_cvec4.shape)
#
# # cnt on dep
# c_vec7 = CountVectorizer(lowercase=False,ngram_range=(1,1))
# c_vec7.fit(train_df['dep_txt'].values.tolist() )
# train_cvec7 = c_vec7.transform(train_df['dep_txt'].values.tolist()).toarray()
# # test_cvec7 = c_vec7.transform(test_df['dep_txt'].values.tolist()).toarray()
# print(train_cvec7.shape)
#
# # tfidf on tag
# tf_vec5 = TfidfVectorizer(lowercase=False,ngram_range=(1,1))
# tf_vec5.fit(train_df['tag_txt'].values.tolist() )
# train_tf5 = tf_vec5.transform(train_df['tag_txt'].values.tolist()).toarray()
# # test_tf5 = tf_vec5.transform(test_df['tag_txt'].values.tolist()).toarray()
# print(train_tf5.shape)
#
# # tfidf on ne
# tf_vec6 = TfidfVectorizer(lowercase=False,ngram_range=(1,2))
# tf_vec6.fit(train_df['pos_txt'].values.tolist() )
# train_tf6 = tf_vec6.transform(train_df['pos_txt'].values.tolist()).toarray()
# # test_tf6 = tf_vec6.transform(test_df['pos_txt'].values.tolist()).toarray()
# print(train_tf6.shape)
#
# # tfidf on dep
# tf_vec8 = TfidfVectorizer(lowercase=False,ngram_range=(1,1))
# tf_vec8.fit(train_df['dep_txt'].values.tolist() )
# train_tf8 = tf_vec8.transform(train_df['dep_txt'].values.tolist()).toarray()
# # test_tf8 = tf_vec8.transform(test_df['dep_txt'].values.tolist()).toarray()
# print(train_tf8.shape)
#
# all_nlp_train = np.hstack([train_cvec3,train_cvec4,train_tf5,train_tf6,train_cvec7, train_tf8])
# # all_nlp_test = np.hstack([test_cvec3,test_cvec4,test_tf5,test_tf6, test_cvec7, test_tf8])
# print('nlp feat done')
#
# pkl_train_dump = open("data/temp_train_out1.txt", "wb")
# pickle.dump(all_nlp_train, pkl_train_dump)
# pkl_train_dump.close()

# add tfidf and svd
# tfidf_vec = TfidfVectorizer(ngram_range=(1,3), max_df=0.8,lowercase=False, sublinear_tf=True)
# full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() )
# train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
# # test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
# # print(train_tfidf.shape)
#
# # svd1
# n_comp = 30
# svd_obj = TruncatedSVD(n_components=n_comp)
# svd_obj.fit(full_tfidf)
# train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
# # test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
# print(train_svd.shape)
#
# ## add tfidf char
# tfidf_vec = TfidfVectorizer(ngram_range=(3,7), analyzer='char',max_df=0.8, sublinear_tf=True)
# full_tfidf2 = tfidf_vec.fit_transform(train_df['text'].values.tolist() )
# train_tfidf2 = tfidf_vec.transform(train_df['text'].values.tolist())
# # test_tfidf2 = tfidf_vec.transform(test_df['text'].values.tolist())
# print(train_tfidf2.shape)
#
# ## add svd2
# svd_obj = TruncatedSVD(n_components=n_comp)
# svd_obj.fit(full_tfidf2)
# train_svd2 = pd.DataFrame(svd_obj.transform(train_tfidf2))
# # test_svd2 = pd.DataFrame(svd_obj.transform(test_tfidf2))
# print(train_svd2.shape)
#
## add cnt vec
# c_vec = CountVectorizer(ngram_range=(1,3),max_df=0.8, lowercase=False)
# full_cvec1 = c_vec.fit_transform(train_df['text'].values.tolist() )
# train_cvec = c_vec.transform(train_df['text'].values.tolist())
# # test_cvec = c_vec.transform(test_df['text'].values.tolist())
# print(train_cvec.shape)
#
# ## add svd3
# svd_obj = TruncatedSVD(n_components=n_comp)
# svd_obj.fit(full_cvec1)
# train_svd3 = pd.DataFrame(svd_obj.transform(train_cvec))
# test_svd3 = pd.DataFrame(svd_obj.transform(test_cvec))
#
# # add cnt char
# c_vec = CountVectorizer(ngram_range=(3,7), analyzer='char',max_df=0.8)
# full_cvec2 = c_vec.fit_transform(train_df['text'].values.tolist() )
# train_cvec2 = c_vec.transform(train_df['text'].values.tolist())
# # test_cvec2 = c_vec.transform(test_df['text'].values.tolist())
# print(train_cvec2.shape)
#
# ## add svd4
# svd_obj = TruncatedSVD(n_components=n_comp)
# svd_obj.fit(full_cvec2)
# train_svd4 = pd.DataFrame(svd_obj.transform(train_cvec2))
# # test_svd4 = pd.DataFrame(svd_obj.transform(test_cvec2))
#
# # add cnt char
# c_vec = CountVectorizer(ngram_range=(1,2), analyzer='char',max_df=0.8)
# full_cvec3 = c_vec.fit_transform(train_df['text'].values.tolist() )
# train_cvec3 = c_vec.transform(train_df['text'].values.tolist())
# # test_cvec3 = c_vec.transform(test_df['text'].values.tolist())
# # print(train_cvec3.shape,test_cvec3.shape)
#
# all_svd_train = np.hstack([train_svd,train_svd3])
# all_svd_test = np.hstack([test_svd,test_svd2,test_svd3,test_svd4,test_cvec3.toarray()])
#
# pkl_train_dump = open("data/temp_train_out2.txt", "wb")
# pickle.dump(all_svd_train, pkl_train_dump)
# pkl_train_dump.close()

all_nlp_train = pickle.load(open("data/temp_train_out1.txt", "rb"))
print(all_nlp_train.shape)

all_svd_train = pickle.load(open("data/temp_train_out2.txt", "rb"))
print(all_svd_train.shape)

punctuation = ['.', '..', '...', ',', ':', ';', '-', '*', '"', '!', '?']
alphabet = 'abcdefghijklmnopqrstuvwxyz'

def clean_text(x):
    x.lower()
    for p in punctuation:
        x.replace(p, '')
    return x

def extract_features(df):
    df['text_cleaned'] = df['text'].apply(lambda x: clean_text(x))
    df['n_.'] = df['text'].str.count('\.')
    df['n_...'] = df['text'].str.count('\...')
    df['n_,'] = df['text'].str.count('\,')
    df['n_:'] = df['text'].str.count('\:')
    df['n_;'] = df['text'].str.count('\;')
    df['n_-'] = df['text'].str.count('\-')
    df['n_?'] = df['text'].str.count('\?')
    df['n_!'] = df['text'].str.count('\!')
    df['n_\''] = df['text'].str.count('\'')
    df['n_"'] = df['text'].str.count('\"')

    # First words in a sentence
    df['n_The '] = df['text'].str.count('The ')
    df['n_I '] = df['text'].str.count('I ')
    df['n_It '] = df['text'].str.count('It ')
    df['n_He '] = df['text'].str.count('He ')
    df['n_Me '] = df['text'].str.count('Me ')
    df['n_She '] = df['text'].str.count('She ')
    df['n_We '] = df['text'].str.count('We ')
    df['n_They '] = df['text'].str.count('They ')
    df['n_You '] = df['text'].str.count('You ')
    df['n_the'] = df['text_cleaned'].str.count('the ')
    df['n_ a '] = df['text_cleaned'].str.count(' a ')
    df['n_appear'] = df['text_cleaned'].str.count('appear')
    df['n_little'] = df['text_cleaned'].str.count('little')
    df['n_was '] = df['text_cleaned'].str.count('was ')
    df['n_one '] = df['text_cleaned'].str.count('one ')
    df['n_two '] = df['text_cleaned'].str.count('two ')
    df['n_three '] = df['text_cleaned'].str.count('three ')
    df['n_ten '] = df['text_cleaned'].str.count('ten ')
    df['n_is '] = df['text_cleaned'].str.count('is ')
    df['n_are '] = df['text_cleaned'].str.count('are ')
    df['n_ed'] = df['text_cleaned'].str.count('ed ')
    df['n_however'] = df['text_cleaned'].str.count('however')
    df['n_ to '] = df['text_cleaned'].str.count(' to ')
    df['n_into'] = df['text_cleaned'].str.count('into')
    df['n_about '] = df['text_cleaned'].str.count('about ')
    df['n_th'] = df['text_cleaned'].str.count('th')
    df['n_er'] = df['text_cleaned'].str.count('er')
    df['n_ex'] = df['text_cleaned'].str.count('ex')
    df['n_an '] = df['text_cleaned'].str.count('an ')
    df['n_ground'] = df['text_cleaned'].str.count('ground')
    df['n_any'] = df['text_cleaned'].str.count('any')
    df['n_silence'] = df['text_cleaned'].str.count('silence')
    df['n_wall'] = df['text_cleaned'].str.count('wall')

    df.drop(['text_cleaned', 'comment_text'], axis=1, inplace=True)
    return df.values


print('Processing train...')
# train_hand_features = \
extract_features(train_df)

eng_stopwords = [
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"]

## Number of words in the text ##
train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train_df["num_unique_words"] = train_df["text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
train_df["num_chars"] = train_df["text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train_df["num_stopwords"] = train_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the text ##
import string
train_df["num_punctuations"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
train_df["num_words_upper"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train_df["num_words_title"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
train_df["mean_word_len"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# add features
def add_feat(df):
    df['unique_r'] = df['num_unique_words'] / df['num_words']
    df['w_p'] = df['num_words'] - df['num_punctuations']
    df['w_p_r'] = df['w_p'] / df['num_words']
    df['stop_r'] = df['num_stopwords'] / df['num_words']
    df['w_p_stop'] = df['w_p'] - df['num_stopwords']
    df['w_p_stop_r'] = df['w_p_stop'] / df['num_words']
    df['num_words_upper_r'] = df['num_words_upper'] / df['num_words']
    df['num_words_title_r'] = df['num_words_title'] / df['num_words']

add_feat(train_df)
# add_feat(test_df)
print(train_df.columns)

cols_to_drop = ['id','text']
# train_Y = train_df['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
Y_train_toxic = train_df['toxic']
Y_train_severe_toxic = train_df['severe_toxic']
Y_train_obscene = train_df['obscene']
Y_train_threat = train_df['threat']
Y_train_insult = train_df['insult']
Y_train_identity_hate = train_df['identity_hate']
X_train = train_df.drop(cols_to_drop+['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1).values
# test_X = test_df.drop(cols_to_drop, axis=1).values
print(X_train.shape)
X_train = np.hstack([X_train, all_nlp_train, all_svd_train])
print(X_train.shape)

nb_validation_samples = int(0.2 * X_train.shape[0])
x_train = X_train[:-nb_validation_samples]
x_val = X_train[-nb_validation_samples:]
# y_train = train_Y[:-nb_validation_samples]
y_train_toxic = Y_train_toxic[:-nb_validation_samples]
y_val_toxic = Y_train_toxic[-nb_validation_samples:]
y_train_severe_toxic = Y_train_severe_toxic[:-nb_validation_samples]
y_val_severe_toxic = Y_train_severe_toxic[-nb_validation_samples:]
y_train_obscene = Y_train_obscene[:-nb_validation_samples]
y_val_obscene = Y_train_obscene[-nb_validation_samples:]
y_train_threat = Y_train_threat[:-nb_validation_samples]
y_val_threat = Y_train_threat[-nb_validation_samples:]
y_train_insult = Y_train_insult[:-nb_validation_samples]
y_val_insult = Y_train_insult[-nb_validation_samples:]
y_train_identity_hate = Y_train_identity_hate[:-nb_validation_samples]
y_val_identity_hate = Y_train_identity_hate[-nb_validation_samples:]

params = {
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'eta': 0.07,
    'max_depth': 3,
    'eval_metric': 'mlogloss',
    'objective': 'multi:softprob',
    'num_class': 2,
    'n_estimators': 200,
    'min_child_weight': 1,
    'max_leaf_nodes': 20,
    'lambda': 0.001
}

d_train = xgb.DMatrix(x_train, y_train_toxic)
d_valid = xgb.DMatrix(x_val, y_val_toxic)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
# train model
model = xgb.train(params, d_train, 200, watchlist, early_stopping_rounds=200, verbose_eval=200)
# get res
train_pred = model.predict(d_train)
valid_pred = model.predict(d_valid)
# cal score
train_score = log_loss(y_train_toxic, train_pred)
valid_score = log_loss(y_val_toxic, valid_pred)
print('train log loss', train_score, 'valid log loss', valid_score)
val_accuracy = accuracy_score(y_val_toxic, valid_pred.argmax(axis=1))
print('toxic validation_accuracy: ', val_accuracy)
pred_toxic = np.row_stack([train_pred, valid_pred])
print(pred_toxic.shape)
pkl_dump = open("output/xgb_toxic.out", "wb")
pickle.dump(pred_toxic, pkl_dump)
pkl_dump.close()

d_train = xgb.DMatrix(x_train, y_train_severe_toxic)
d_valid = xgb.DMatrix(x_val, y_val_severe_toxic)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
# train model
model = xgb.train(params, d_train, 200, watchlist, early_stopping_rounds=200, verbose_eval=200)
# get res
train_pred = model.predict(d_train)
valid_pred = model.predict(d_valid)
# cal score
train_score = log_loss(y_train_severe_toxic, train_pred)
valid_score = log_loss(y_val_severe_toxic, valid_pred)
print('train log loss', train_score, 'valid log loss', valid_score)
val_accuracy = accuracy_score(y_val_severe_toxic, valid_pred.argmax(axis=1))
print('severe toxic validation_accuracy: ', val_accuracy)
pred_severe_toxic = np.row_stack([train_pred, valid_pred])
print(pred_severe_toxic.shape)
pkl_dump = open("output/xgb_severe_toxic.out", "wb")
pickle.dump(pred_severe_toxic, pkl_dump)
pkl_dump.close()

d_train = xgb.DMatrix(x_train, y_train_obscene)
d_valid = xgb.DMatrix(x_val, y_val_obscene)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
# train model
model = xgb.train(params, d_train, 200, watchlist, early_stopping_rounds=200, verbose_eval=200)
# get res
train_pred = model.predict(d_train)
valid_pred = model.predict(d_valid)
# cal score
train_score = log_loss(y_train_obscene, train_pred)
valid_score = log_loss(y_val_obscene, valid_pred)
print('train log loss', train_score, 'valid log loss', valid_score)
val_accuracy = accuracy_score(y_val_obscene, valid_pred.argmax(axis=1))
print('obscene validation_accuracy: ', val_accuracy)
pred_obscene = np.row_stack([train_pred, valid_pred])
print(pred_obscene.shape)
pkl_dump = open("output/xgb_obscene.out", "wb")
pickle.dump(pred_obscene, pkl_dump)
pkl_dump.close()

d_train = xgb.DMatrix(x_train, y_train_threat)
d_valid = xgb.DMatrix(x_val, y_val_threat)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
# train model
model = xgb.train(params, d_train, 200, watchlist, early_stopping_rounds=200, verbose_eval=200)
# get res
train_pred = model.predict(d_train)
valid_pred = model.predict(d_valid)
# cal score
train_score = log_loss(y_train_threat, train_pred)
valid_score = log_loss(y_val_threat, valid_pred)
print('train log loss', train_score, 'valid log loss', valid_score)
val_accuracy = accuracy_score(y_val_threat, valid_pred.argmax(axis=1))
print('threat validation_accuracy: ', val_accuracy)
pred_threat = np.row_stack([train_pred, valid_pred])
print(pred_threat.shape)
pkl_dump = open("output/xgb_threat.out", "wb")
pickle.dump(pred_threat, pkl_dump)
pkl_dump.close()

d_train = xgb.DMatrix(x_train, y_train_insult)
d_valid = xgb.DMatrix(x_val, y_val_insult)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
# train model
model = xgb.train(params, d_train, 200, watchlist, early_stopping_rounds=200, verbose_eval=200)
# get res
train_pred = model.predict(d_train)
valid_pred = model.predict(d_valid)
# cal score
train_score = log_loss(y_train_insult, train_pred)
valid_score = log_loss(y_val_insult, valid_pred)
print('train log loss', train_score, 'valid log loss', valid_score)
val_accuracy = accuracy_score(y_val_insult, valid_pred.argmax(axis=1))
print('insult validation_accuracy: ', val_accuracy)
pred_insult = np.row_stack([train_pred, valid_pred])
print(pred_insult.shape)
pkl_dump = open("output/xgb_insult.out", "wb")
pickle.dump(pred_insult, pkl_dump)
pkl_dump.close()

d_train = xgb.DMatrix(x_train, y_train_identity_hate)
d_valid = xgb.DMatrix(x_val, y_val_identity_hate)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
# train model
model = xgb.train(params, d_train, 200, watchlist, early_stopping_rounds=200, verbose_eval=200)
# get res
train_pred = model.predict(d_train)
valid_pred = model.predict(d_valid)
# cal score
train_score = log_loss(y_train_identity_hate, train_pred)
valid_score = log_loss(y_val_identity_hate, valid_pred)
print('train log loss', train_score, 'valid log loss', valid_score)
val_accuracy = accuracy_score(y_val_identity_hate, valid_pred.argmax(axis=1))
print('identity hate validation_accuracy: ', val_accuracy)
pred_identity_hate = np.row_stack([train_pred, valid_pred])
print(pred_identity_hate.shape)
pkl_dump = open("output/xgb_identity_hate.out", "wb")
pickle.dump(pred_identity_hate, pkl_dump)
pkl_dump.close()
