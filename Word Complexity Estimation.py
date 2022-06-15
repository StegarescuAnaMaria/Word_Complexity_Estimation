import string
import numpy as np
from collections import Counter
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.model_selection import validation_curve
import csv



def plot_figure(Y_train, Y_val, X, title=None, legend=None, labels=None):
  plt.figure()
  if title:
    plt.title(title)
  line1, = plt.plot(X, Y_train, 'bo--')
  line2, = plt.plot(X, Y_val, 'ro--')
  if legend:
    line1.set_label(legend[0])
    line2.set_label(legend[1])
    plt.legend()
  if labels:
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
  plt.show()



#returns list of elements, of separated data - column id, sentence, etc
def separate_data(text_lines):
  separated = []
  for line in text_lines:
    line_aux = line[:-1]  # eliminate "\n" from the end of the line
    separated.append(line_aux.split('\t'))
  return separated


#returns list of sentences, target phrases, and sum of total annotators and selected annotators and row ids
def sentence_phrase_annotators(text_separated, train=True):
  # list of sentences in lowercase and no punctuation
  sentences = []

  # list of target phrases for which we learn the complexity
  phrases = []

  #row IDs
  ids = []

  # total annotators who saw the sentence
  total_annotators = np.zeros((len(text_separated), 2), dtype='float')

  # labels - annotators who found the sentence difficult
  selected_annotators = np.zeros((len(text_separated), 2), dtype='float')

  # predict the sum of the native and non-native annotators
  total_annotators_sum = np.zeros(len(text_separated), dtype='float')
  selected_annotators_sum = np.zeros(len(text_separated), dtype='float')

  # phrases are the 5th element in every train_separated array
  # sentences are the 2nd element in every train_separated array
  for i in range(len(text_separated)):
    row = text_separated[i]
    ids.append(row[0])
    line = row[1].lower()
    sentences.append(line.translate(str.maketrans('', '', string.punctuation)))
    line = row[4].lower()
    phrases.append(line.translate(str.maketrans('', '', string.punctuation)))

    total_annotators[i][0] = float(row[5])
    total_annotators[i][1] = float(row[6])
    total_annotators_sum[i] = total_annotators[i][0] + total_annotators[i][1]

    if train:
      selected_annotators[i][0] = float(row[7])
      selected_annotators[i][1] = float(row[8])
      selected_annotators_sum[i] = selected_annotators[i][0] + selected_annotators[i][1]
  return sentences, phrases, total_annotators_sum, selected_annotators_sum, ids


#returns list of lists of words, a list of words is a separated sentence
def separate_sentences(sentences):
  sentences_separated = []
  for line in sentences:
    sentences_separated.append(line.split(' '))
  return sentences_separated


#returns dictionary of words
def word_dict(sentences_separated):
  training_dict = {}
  for sentence in sentences_separated:
    for word in sentence:
      if word not in training_dict.keys():
        training_dict[word] = len(training_dict.keys())
  return training_dict



#returns list of lists, of separated phrases, and extracted features in a bag of words
def bag_of_words(training_dict, phrases):
  bag_of_words = np.zeros((len(phrases), len(training_dict.keys())), dtype='float')
  phrases_separated = []
  for i in range(len(phrases)):
    words = phrases[i].split(' ')
    phrases_separated.append(words)
    d = Counter(words)  # a dictionary of nr of appearances
    for word in d:
      if word in training_dict:
        bag_of_words[i][training_dict[word]] = d[word]
  return bag_of_words, phrases_separated


#returns tf_idf extracted features
def tf_idf(training_dict, phrases_separated):
  tf_idf = np.zeros((len(phrases_separated), len(training_dict.keys())), dtype='float')

  for i in range(len(phrases_separated)):
    d = Counter(phrases_separated[i])
    for word in d:
      if word in training_dict:
        tf = d[word] / len(phrases_separated[i])
        counter = 0
        for phrase in phrases_separated:
          if word in phrase:
            counter += 1
        idf = np.log10(counter / len(phrases_separated))
        tf_idf[i][training_dict[word]] = tf * idf
  return tf_idf



folder="D:\Semestrul I\PML\pml-unibuc-2021"
train_file = folder+"/train_full.txt"

f = open(train_file, "r")
lines_train = f.readlines()

#Separated train data, into id, sentence, ..., target phrase, ...
train_separated = separate_data(lines_train)

#list of train sentences, target phrases, sum of total annotators and the annotators who found the phrase difficult
train_sentences, train_phrases, total_annotators_sum_train, selected_annotators_sum_train, _ = sentence_phrase_annotators(train_separated)

#train sentences separated into words - a list of lists
train_sentences_separated = separate_sentences(train_sentences)

#Dictionary used for feature extraction
training_dict = word_dict(train_sentences_separated)

#bag of words features, and target phrases separated in words - a list of lists
bag_of_words_train, phrases_train_separated = bag_of_words(training_dict, train_phrases)

#tf-idf feature extraction
tf_idf_train = tf_idf(training_dict, phrases_train_separated)

#Feature normalization
scaler = preprocessing.StandardScaler().fit(bag_of_words_train)
bag_of_words_train_scaled = scaler.transform(bag_of_words_train)
scaler2 = preprocessing.StandardScaler().fit(tf_idf_train)
tf_idf_train_scaled = scaler2.transform(tf_idf_train)


#scaling the target values for an easier SVM Regressor convergence
scaler_target = preprocessing.StandardScaler().fit(selected_annotators_sum_train.reshape(-1, 1))
selected_annotators_scaled = scaler_target.transform(selected_annotators_sum_train.reshape(-1, 1)).reshape(-1)

#The commented zone below is for trying to tune the hyperparameters by using the validation_curve from sklearn
"""
train_scores, valid_scores = validation_curve(estimator = MLPRegressor(solver='sgd',hidden_layer_sizes=(30, 40),alpha=0.001,learning_rate='adaptive',
                                                            early_stopping=True), X=train_features_scaled, y=selected_annotators_sum_train,
                                              scoring = 'neg_mean_squared_error', error_score="raise", param_name="max_iter",
                                              param_range=[200, 250, 300, 400, 500], cv=5)
"""
"""
train_scores, valid_scores = validation_curve(estimator=MLPRegressor(solver='sgd', learning_rate='adaptive',hidden_layer_sizes=(10, 10),
                                                          early_stopping=True), X=tf_idf_train_scaled, y=selected_annotators_sum_train,
                                              scoring = 'neg_mean_squared_error', error_score="raise", param_name="max_iter",
                                              param_range=[300, 500, 800, 1000], cv=5)
"""
"""
train_scores, valid_scores = validation_curve(estimator=svm.SVR(kernel='poly', degree=2, C=1, gamma='scale', epsilon=0,
                                                                cache_size=50, verbose=True),
                                              X=tf_idf_scaled, y=selected_annotators_scaled,
                                              scoring = 'neg_mean_squared_error', error_score="raise", param_name="max_iter",
                                              param_range=[250, -1], cv=5) #does 5-fold validation

"""

"""
print(train_scores)
print(valid_scores)
train_scores_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
print(train_scores_mean)
print(valid_scores_mean)
print("aici")

#Value of X for plotting, usually the range of the observed hyperparameter
X = [1, 2, 3, 4]
plot_figure(train_scores_mean, valid_scores_mean, X, title=None, legend=['Train error', 'Validation error'],
            labels=['Nr of iterations', 'Neg MSE'])
"""
#SVR model with best result, it takes a lot of time to converge, as max_iter is infinite
svr = svm.SVR(kernel='poly', degree=2, C=1, gamma='scale', epsilon=0, cache_size=50, verbose=True, max_iter=-1)
svr.fit(tf_idf_train_scaled, selected_annotators_scaled)

#MLPRegressor model with best result
regr = MLPRegressor(solver='sgd')
regr.fit(bag_of_words_train_scaled, selected_annotators_sum_train)



test_file = folder+"/test.txt"
f_test = open(test_file, "r")
test_lines = f_test.readlines()

#separated data into id, sentence, ..., target phrase, ...
test_separated = separate_data(test_lines)

#list of test phrases, sum of total annotators, sum of annotators who found the phrase to be difficult
_, test_phrases, total_annotators_sum_test, selected_annotators_sum_test, ids = sentence_phrase_annotators(test_separated, False)

#bag of words features on test data, list of target phrases separated into words
bag_of_words_test, test_phrases_separated = bag_of_words(training_dict, test_phrases)

#tf-idf features
tf_idf_test = tf_idf(training_dict, test_phrases_separated)

#scaling the features
test_features_scaled = scaler.transform(bag_of_words_test)
test_tfidf_scaled = scaler2.transform(tf_idf_test)

#predicting with our 2 models
labels_test = regr.predict(test_features_scaled)
labels_test2 = svr.predict(test_tfidf_scaled)

#predicting the complexity with the sum of annotators we got
complexity_output_test = np.around(labels_test)/total_annotators_sum_test

#because we scaled the train labels, the predicted values need to be re-scaled.
complexity_output_test2 = np.around(scaler_target.inverse_transform(labels_test2.reshape(-1, 1)).reshape(-1))/total_annotators_sum_test

#The models apply ReLU only in between layers
for i in range(len(test_lines)):
  if complexity_output_test[i] <= -0.0:
    complexity_output_test[i] = 0.0
  if complexity_output_test2[i] <= -0.0:
    complexity_output_test2[i] = 0.0

submission = open(folder+"/submission14.txt", "w")
submission.write("id,label\n")

#writing the submission
for i in range(len(test_lines)):
  #we can also write complexity_output_test[i] depending on which results we want to submit
  submission.write(ids[i]+','+str(complexity_output_test2[i])+'\n')
submission.close()


submission = open(folder+"/submission11.csv", "w")

writer = csv.writer(submission, delimiter=',')
writer.writerow(["id","label"])

#writing the submission as a csv text
for i in range(len(test_lines)):
  # we can also write complexity_output_test[i] depending on which results we want to submit
  writer.writerow([ids[i], str(complexity_output_test2[i])])
submission.close()