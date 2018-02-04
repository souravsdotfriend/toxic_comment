
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

models = [('MultiNB', MultinomialNB(alpha=0.03)),
          ('Calibrated MultiNB', CalibratedClassifierCV(
              MultinomialNB(alpha=0.03), method='isotonic')),
          ('Calibrated BernoulliNB', CalibratedClassifierCV(
              BernoulliNB(alpha=0.03), method='isotonic')),
          ('Calibrated Huber', CalibratedClassifierCV(
              SGDClassifier(loss='modified_huber', alpha=1e-4,
                            max_iter=10000, tol=1e-4), method='sigmoid')),
          ('Logit', LogisticRegression(C=30))]

train = pd.read_csv('data/train.csv')
vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[3,3,3,1,1])
X_train = vectorizer.fit_transform(train.comment_text.values)
columnList = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
Y_train_toxic = train.toxic.values
Y_train_severe_toxic = train.severe_toxic.values
Y_train_obscene = train.obscene.values
Y_train_threat = train.threat.values
Y_train_insult = train.insult.values
Y_train_identity_hate = train.identity_hate.values

nb_validation_samples = int(0.2 * X_train.shape[0])
x_train = X_train[:-nb_validation_samples]
x_val = X_train[-nb_validation_samples:]
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

clf.fit(x_train, y_train_toxic)
valid_pred_toxic = clf.predict(x_val)
val_accuracy = accuracy_score(y_val_toxic, valid_pred_toxic)
print('toxic validation_accuracy: ', val_accuracy)
pred_toxic = clf.predict_proba(X_train)
pkl_train_dump = open("output/nb_pred_toxic.txt", "wb")
pickle.dump(pred_toxic, pkl_train_dump)
pkl_train_dump.close()

clf.fit(x_train, y_train_severe_toxic)
valid_pred_severe_toxic = clf.predict(x_val)
val_accuracy = accuracy_score(y_val_severe_toxic, valid_pred_severe_toxic)
print('severe toxic validation_accuracy: ', val_accuracy)
pred_severe_toxic = clf.predict_proba(X_train)
pkl_train_dump = open("output/nb_pred_severe_toxic.txt", "wb")
pickle.dump(pred_severe_toxic, pkl_train_dump)
pkl_train_dump.close()

clf.fit(x_train, y_train_obscene)
valid_pred_obscene = clf.predict(x_val)
val_accuracy = accuracy_score(y_val_obscene, valid_pred_obscene)
print('obscene validation_accuracy: ', val_accuracy)
pred_obscene = clf.predict_proba(X_train)
pkl_train_dump = open("output/nb_pred_obscene.txt", "wb")
pickle.dump(pred_obscene, pkl_train_dump)
pkl_train_dump.close()

clf.fit(x_train, y_train_threat)
valid_pred_threat = clf.predict(x_val)
val_accuracy = accuracy_score(y_val_threat, valid_pred_threat)
print('threat validation_accuracy: ', val_accuracy)
pred_threat = clf.predict_proba(X_train)
pkl_train_dump = open("output/nb_pred_threat.txt", "wb")
pickle.dump(pred_threat, pkl_train_dump)
pkl_train_dump.close()

clf.fit(x_train, y_train_insult)
valid_pred_insult = clf.predict(x_val)
val_accuracy = accuracy_score(y_val_insult, valid_pred_insult)
print('insult validation_accuracy: ', val_accuracy)
pred_insult = clf.predict_proba(X_train)
pkl_train_dump = open("output/nb_pred_insult.txt", "wb")
pickle.dump(pred_insult, pkl_train_dump)
pkl_train_dump.close()

clf.fit(x_train, y_train_identity_hate)
valid_pred_identity_hate = clf.predict(x_val)
val_accuracy = accuracy_score(y_val_identity_hate, valid_pred_identity_hate)
print('identity hate validation_accuracy: ', val_accuracy)
pred_identity_hate = clf.predict_proba(X_train)
pkl_train_dump = open("output/nb_pred_identity_hate.txt", "wb")
pickle.dump(pred_identity_hate, pkl_train_dump)
pkl_train_dump.close()
