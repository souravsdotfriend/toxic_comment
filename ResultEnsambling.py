import csv
import pickle
from collections import Counter

from keras.utils import to_categorical
from sklearn import metrics, linear_model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from matplotlib import gridspec
from mlxtend.classifier import StackingClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv('data/test.csv', sep=',')

comment_id_test = test_df['id']
train_Y_toxic = train_df['toxic']
train_Y_severe_toxic = train_df['severe_toxic']
train_Y_obscene = train_df['obscene']
train_Y_threat = train_df['threat']
train_Y_insult = train_df['insult']
train_Y_identity_hate = train_df['identity_hate']

cnn_predictions_toxic = pickle.load(open("output/cnn_toxic.out", "rb"))
cnn_predictions_severe_toxic = pickle.load(open("output/cnn_severe_toxic.out", "rb"))
cnn_predictions_obscene = pickle.load(open("output/cnn_obscene.out", "rb"))
cnn_predictions_threat = pickle.load(open("output/cnn_threat.out", "rb"))
cnn_predictions_insult = pickle.load(open("output/cnn_insult.out", "rb"))
cnn_predictions_identity_hate = pickle.load(open("output/cnn_identity_hate.out", "rb"))
cnn_complex_predictions_toxic = pickle.load(open("output/cnn_complex_toxic.out", "rb"))
cnn_complex_predictions_severe_toxic = pickle.load(open("output/cnn_complex_severe_toxic.out", "rb"))
cnn_complex_predictions_obscene = pickle.load(open("output/cnn_complex_obscene.out", "rb"))
cnn_complex_predictions_threat = pickle.load(open("output/cnn_complex_threat.out", "rb"))
cnn_complex_predictions_insult = pickle.load(open("output/cnn_complex_insult.out", "rb"))
cnn_complex_predictions_identity_hate = pickle.load(open("output/cnn_complex_identity_hate.out", "rb"))
hier_lstm_predictions_toxic = pickle.load(open("output/hier_lstm_toxic.out", "rb"))
hier_lstm_predictions_severe_toxic = pickle.load(open("output/hier_lstm_severe_toxic.out", "rb"))
hier_lstm_predictions_obscene = pickle.load(open("output/hier_lstm_obscene.out", "rb"))
hier_lstm_predictions_threat = pickle.load(open("output/hier_lstm_threat.out", "rb"))
hier_lstm_predictions_insult = pickle.load(open("output/hier_lstm_insult.out", "rb"))
hier_lstm_predictions_identity_hate = pickle.load(open("output/hier_lstm_identity_hate.out", "rb"))
hier_att_predictions_toxic = pickle.load(open("output/hier_att.out", "rb"))
hier_att_predictions_severe_toxic = pickle.load(open("output/hier_att_severe_toxic.out", "rb"))
hier_att_predictions_obscene = pickle.load(open("output/hier_att_obscene.out", "rb"))
hier_att_predictions_threat = pickle.load(open("output/hier_att_threat.out", "rb"))
hier_att_predictions_insult = pickle.load(open("output/hier_att_insult.out", "rb"))
hier_att_predictions_identity_hate = pickle.load(open("output/hier_att_identity_hate.out", "rb"))
bidirectional_lstm_predictions_toxic = pickle.load(open("output/bidirectional_lstm_toxic.out", "rb"))
bidirectional_lstm_predictions_severe_toxic = pickle.load(open("output/bidirectional_lstm_severe_toxic.out", "rb"))
bidirectional_lstm_predictions_obscene = pickle.load(open("output/bidirectional_lstm_obscene.out", "rb"))
bidirectional_lstm_predictions_threat = pickle.load(open("output/bidirectional_lstm_threat.out", "rb"))
bidirectional_lstm_predictions_insult = pickle.load(open("output/bidirectional_lstm_insult.out", "rb"))
bidirectional_lstm_predictions_identity_hate = pickle.load(open("output/bidirectional_lstm_identity_hate.out", "rb"))
gru_att_predictions_toxic = pickle.load(open("output/gru_att_toxic.out", "rb"))
gru_att_predictions_severe_toxic = pickle.load(open("output/gru_att_severe_toxic.out", "rb"))
gru_att_predictions_obscene = pickle.load(open("output/gru_att_obscene.out", "rb"))
gru_att_predictions_threat = pickle.load(open("output/gru_att_threat.out", "rb"))
gru_att_predictions_insult = pickle.load(open("output/gru_att_insult.out", "rb"))
gru_att_predictions_identity_hate = pickle.load(open("output/gru_att_identity_hate.out", "rb"))
# nb_predictions_toxic = pickle.load(open("output/nb_pred_toxic.txt", "rb"))
# nb_predictions_severe_toxic = pickle.load(open("output/nb_pred_severe_toxic.txt", "rb"))
# nb_predictions_obscene = pickle.load(open("output/nb_pred_obscene.txt", "rb"))
# nb_predictions_threat = pickle.load(open("output/nb_pred_threat.txt", "rb"))
# nb_predictions_insult = pickle.load(open("output/nb_pred_insult.txt", "rb"))
# nb_predictions_identity_hate = pickle.load(open("output/nb_pred_identity_hate.txt", "rb"))
# xgb_predictions_toxic = pickle.load(open("output/xgb_toxic.out", "rb"))
# xgb_predictions_severe_toxic = pickle.load(open("output/xgb_severe_toxic.out", "rb"))
# xgb_predictions_obscene = pickle.load(open("output/xgb_obscene.out", "rb"))
# xgb_predictions_threat = pickle.load(open("output/xgb_threat.out", "rb"))
# xgb_predictions_insult = pickle.load(open("output/xgb_insult.out", "rb"))
# xgb_predictions_identity_hate = pickle.load(open("output/xgb_identity_hate.out", "rb"))

train_X_toxic = np.hstack([hier_lstm_predictions_toxic, hier_att_predictions_toxic, bidirectional_lstm_predictions_toxic])
train_X_severe_toxic = np.hstack([cnn_predictions_severe_toxic, cnn_complex_predictions_severe_toxic, hier_lstm_predictions_severe_toxic, gru_att_predictions_severe_toxic])
train_X_obscene = np.hstack([hier_att_predictions_obscene, bidirectional_lstm_predictions_obscene, gru_att_predictions_obscene])
train_X_threat = np.hstack([bidirectional_lstm_predictions_threat, gru_att_predictions_threat])
train_X_insult = np.hstack([hier_att_predictions_insult, bidirectional_lstm_predictions_insult])
train_X_identity_hate = np.hstack([bidirectional_lstm_predictions_identity_hate, gru_att_predictions_identity_hate])

nb_validation_samples = 127656
x_train_toxic, x_val_toxic, x_test_toxic = np.split(train_X_toxic, [nb_validation_samples, 159571])
y_train_toxic,y_val_toxic = np.split(train_Y_toxic, [nb_validation_samples])


x_train_severe_toxic, x_val_severe_toxic, x_test_severe_toxic = np.split(train_X_severe_toxic, [nb_validation_samples, 159571])
y_train_severe_toxic, y_val_severe_toxic = np.split(train_Y_severe_toxic, [nb_validation_samples])

x_train_obscene, x_val_obscene, x_test_obscene,  = np.split(train_X_obscene, [nb_validation_samples, 159571])
y_train_obscene, y_val_obscene = np.split(train_Y_obscene, [nb_validation_samples])

x_train_threat, x_val_threat, x_test_threat = np.split(train_X_threat, [nb_validation_samples, 159571])
y_train_threat, y_val_threat = np.split(train_Y_threat, [nb_validation_samples])

x_train_insult, x_val_insult, x_test_insult = np.split(train_X_insult, [nb_validation_samples, 159571])
y_train_insult, y_val_insult = np.split(train_Y_insult, [nb_validation_samples])

x_train_identity_hate, x_val_identity_hate, x_test_identity_hate = np.split(train_X_identity_hate, [nb_validation_samples, 159571])
y_train_identity_hate, y_val_identity_hate = np.split(train_Y_identity_hate, [nb_validation_samples])

lr = linear_model.LogisticRegression()
lr.fit(x_train_toxic, y_train_toxic)
prediction_toxic = lr.predict(x_val_toxic)
print("toxic Test Accuracy :: ", metrics.accuracy_score(y_val_toxic, prediction_toxic))

lr = linear_model.LogisticRegression()
lr.fit(x_train_severe_toxic, y_train_severe_toxic)
prediction_severe_toxic = lr.predict(x_val_severe_toxic)
print("severe_toxicn Test Accuracy :: ", metrics.accuracy_score(y_val_severe_toxic, prediction_severe_toxic))

lr = linear_model.LogisticRegression()
lr.fit(x_train_insult, y_train_insult)
prediction_insult = lr.predict(x_val_insult)
print("insult Test Accuracy :: ", metrics.accuracy_score(y_val_insult, prediction_insult))

mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
mul_lr.fit(x_train_obscene, y_train_obscene)
prediction_obscene = mul_lr.predict(x_val_obscene)
print("obscene Test Accuracy :: ", metrics.accuracy_score(y_val_obscene, prediction_obscene))

mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
mul_lr.fit(x_train_threat, y_train_threat)
prediction_threat = mul_lr.predict(x_val_threat)
print("obscene Test Accuracy :: ", metrics.accuracy_score(y_val_obscene, prediction_obscene))

mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
mul_lr.fit(x_train_identity_hate, y_train_identity_hate)
prediction_identity_hate = mul_lr.predict(x_val_identity_hate)
print("identity_hate Test Accuracy :: ", metrics.accuracy_score(y_val_identity_hate, prediction_identity_hate))

# subbmission = np.column_stack([comment_id_test, prediction_toxic, prediction_severe_toxic, prediction_obscene, prediction_threat, prediction_insult, prediction_identity_hate])
# print(subbmission.shape)
#
# with open('data/subbmission_val.csv', 'w+') as outcsv:
#     writer = csv.writer(outcsv)
#     writer.writerow(['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
#     # writer.writerow(subbmission)
#     for i in range(subbmission.shape[0]):
#         writer.writerow(subbmission[i])
#
#
# print("!!!submit the submission file now!!!")























# lr = linear_model.LogisticRegression()
# lr.fit(x_train_identity_hate, y_train_identity_hate)
# Train multinomial logistic regression model
# mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
# mul_lr.fit(x_train_identity_hate, y_train_identity_hate)
# print("Logistic regression Train Accuracy :: ", metrics.accuracy_score(y_train_toxic, lr.predict(x_train_toxic)))
# print("Logistic regression Test Accuracy :: ", metrics.accuracy_score(y_val_identity_hate, lr.predict(x_val_identity_hate)))
# print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(y_train_toxic, mul_lr.predict(x_train_toxic)))
# print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(y_val_identity_hate, mul_lr.predict(x_val_identity_hate)))

# clf1 = KNeighborsClassifier(n_neighbors=5)
# clf2 = RandomForestClassifier(random_state=1)
# clf3 = GaussianNB()
# lr = LogisticRegression()
# clf4 = XGBClassifier()
# sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
#                           meta_classifier=lr)
# label = ['KNN', 'Random Forest', 'Naive Bayes', 'Stacking Classifier']
# clf_list = [clf1, clf2, clf3, sclf]
#
# fig = plt.figure(figsize=(10, 8))
# gs = gridspec.GridSpec(2, 2)
# grid = itertools.product([0, 1], repeat=2)
#
# clf_cv_mean = []
# clf_cv_std = []
# for clf, label, grd in zip(clf_list, label, grid):
#     scores = cross_val_score(clf, x_train_toxic, y_train_toxic, cv=3, scoring='accuracy')
#     print(label," Accuracy: %.2f (+/- %.2f) [%s]" % (scores.mean(), scores.std(), label))
#     clf_cv_mean.append(scores.mean())
#     clf_cv_std.append(scores.std())
#
#     clf.fit(x_train_toxic, y_train_toxic)
#     print("Train Accuracy :: ", metrics.accuracy_score(y_train_toxic, clf.predict(x_train_toxic)))
#     print("Test Accuracy :: ", metrics.accuracy_score(y_val_toxic, clf.predict(x_val_toxic)))
#     cm = confusion_matrix(y_val_toxic, clf.predict(x_val_toxic))
#     print(cm)
#     # ax = plt.subplot(gs[grd[0], grd[1]])
#     # fig = plot_decision_regions(X=x_train_toxic, y=y_train_toxic.values, clf=clf)
#     # plt.title(label)
#
# # plt.show()

# clf = KNeighborsClassifier(n_neighbors=3)
# clf.fit(x_train_toxic, y_train_toxic)
# print("toxic")
# print("Test Accuracy :: ", metrics.accuracy_score(y_val_toxic, clf.predict(x_val_toxic)))
#
# clf.fit(x_train_severe_toxic, y_train_severe_toxic)
# print("severe_toxic")
# print("Test Accuracy :: ", metrics.accuracy_score(y_val_severe_toxic, clf.predict(x_val_severe_toxic)))
#
# clf.fit(x_train_obscene, y_train_obscene)
# print("obscene")
# print("Test Accuracy :: ", metrics.accuracy_score(y_val_obscene, clf.predict(x_val_obscene)))
#
# clf.fit(x_train_threat, y_train_threat)
# print("threat")
# print("Test Accuracy :: ", metrics.accuracy_score(y_val_threat, clf.predict(x_val_threat)))
#
# clf.fit(x_train_insult, y_train_insult)
# print("insult")
# print("Test Accuracy :: ", metrics.accuracy_score(y_val_insult, clf.predict(x_val_insult)))
#
# clf.fit(x_train_identity_hate, y_train_identity_hate)
# print("identity_hate")
# print("Test Accuracy :: ", metrics.accuracy_score(y_val_identity_hate, clf.predict(x_val_identity_hate)))

# lr = linear_model.LogisticRegression()
# lr.fit(x_train_toxic, y_train_toxic)
# # Train multinomial logistic regression model
# mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(x_train_toxic, y_train_toxic)
# # print("Logistic regression Train Accuracy :: ", metrics.accuracy_score(y_train_toxic, lr.predict(x_train_toxic)))
# print("Logistic regression Test Accuracy :: ", metrics.accuracy_score(y_val_toxic, lr.predict(x_val_toxic)))
# # print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(y_train_toxic, mul_lr.predict(x_train_toxic)))
# print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(y_val_toxic, mul_lr.predict(x_val_toxic)))
# cm = confusion_matrix(y_val_toxic, mul_lr.predict(x_val_toxic))
# print(cm)
# cm = confusion_matrix(y_val_toxic, lr.predict(x_val_toxic))
# print(cm)

# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB().fit(x_train, y_train)
# gnb_predictions = gnb.predict(x_val)
# # accuracy on X_test
# accuracy = gnb.score(x_val, y_val)
# print(accuracy)
# # creating a confusion matrix
# cm = confusion_matrix(y_val, gnb_predictions)
# print(cm)

# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=7).fit(x_train, y_train)
# # accuracy on X_test
# accuracy = knn.score(x_val, y_val)
# print(accuracy)
# # creating a confusion matrix
# knn_predictions = knn.predict(x_val)
# cm = confusion_matrix(y_val, knn_predictions)
# print(cm)

# from sklearn.svm import SVC
# svm_model_linear = SVC(kernel='linear', C=1).fit(x_train, y_train)
# svm_predictions = svm_model_linear.predict(x_val)
# # model accuracy for X_test
# accuracy = svm_model_linear.score(x_val, y_val)
# print(accuracy)
# # creating a confusion matrix
# cm = confusion_matrix(y_val, svm_predictions)
# print(cm)

# training a DescisionTreeClassifier
# from sklearn.tree import DecisionTreeClassifier
# dtree_model = DecisionTreeClassifier(max_depth=2).fit(x_train, y_train)
# dtree_predictions = dtree_model.predict(x_val)
# accuracy = metrics.accuracy_score(y_val, dtree_predictions)
# print(accuracy)
# # creating a confusion matrix
# cm = confusion_matrix(y_val, dtree_predictions)
# print(cm)

# bdt_real = AdaBoostClassifier(
#     DecisionTreeClassifier(max_depth=2),
#     n_estimators=600,
#     learning_rate=1)
#
# bdt_discrete = AdaBoostClassifier(
#     DecisionTreeClassifier(max_depth=2),
#     n_estimators=600,
#     learning_rate=1.5,
#     algorithm="SAMME")
#
# bdt_real.fit(x_train, y_train)
# bdt_discrete.fit(x_train, y_train)
#
# accuracy = metrics.accuracy_score(y_val, bdt_real.predict(x_val))
# print('bdt real acc: ', accuracy)
# accuracy = metrics.accuracy_score(y_val, bdt_discrete.predict(x_val))
# print('bdt discrete acc: ', accuracy)
# creating a confusion matrix
# cm = confusion_matrix(y_val, bdt_real.predict(x_val))
# print(cm)
# cm = confusion_matrix(y_val, bdt_discrete.predict(x_val))
# print(cm)


# real_test_errors = []
# discrete_test_errors = []
#
# for real_test_predict, discrete_train_predict in zip(
#         bdt_real.staged_predict(x_val), bdt_discrete.staged_predict(x_val)):
#     real_test_errors.append(
#         1. - accuracy_score(real_test_predict, y_val))
#     discrete_test_errors.append(
#         1. - accuracy_score(discrete_train_predict, y_val))
#
# n_trees_discrete = len(bdt_discrete)
# n_trees_real = len(bdt_real)
#
#
# print(bdt_discrete.predict(x_val))
# print(bdt_real.predict(x_val))
#
# # Boosting might terminate early, but the following arrays are always
# # n_estimators long. We crop them to the actual number of trees here:
# discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
# real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
# discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]
#
# plt.figure(figsize=(15, 5))
#
# plt.subplot(131)
# plt.plot(range(1, n_trees_discrete + 1),
#          discrete_test_errors, c='black', label='SAMME')
# plt.plot(range(1, n_trees_real + 1),
#          real_test_errors, c='black',
#          linestyle='dashed', label='SAMME.R')
# plt.legend()
# plt.ylim(0.18, 0.62)
# plt.ylabel('Test Error')
# plt.xlabel('Number of Trees')
#
# plt.subplot(132)
# plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
#          "b", label='SAMME', alpha=.5)
# plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
#          "r", label='SAMME.R', alpha=.5)
# plt.legend()
# plt.ylabel('Error')
# plt.xlabel('Number of Trees')
# plt.ylim((.2,
#          max(real_estimator_errors.max(),
#              discrete_estimator_errors.max()) * 1.2))
# plt.xlim((-20, len(bdt_discrete) + 20))
#
# plt.subplot(133)
# plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
#          "b", label='SAMME')
# plt.legend()
# plt.ylabel('Weight')
# plt.xlabel('Number of Trees')
# plt.ylim((0, discrete_estimator_weights.max() * 1.2))
# plt.xlim((-20, n_trees_discrete + 20))
#
# # prevent overlapping y-axis labels
# plt.subplots_adjust(wspace=0.25)
# plt.show()