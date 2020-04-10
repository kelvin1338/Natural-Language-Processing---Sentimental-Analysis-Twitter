#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import re
import nltk
import testsets
import evaluation
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
from collections import defaultdict
import numpy as np
np.random.seed(1)

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Embedding

from keras.preprocessing.sequence import pad_sequences

training_processed_tweets = []
testing_processed_tweets = []


string_training = []
training_count = []
string_training_labels = []

string_testing = []
testing_count = []
string_testing_labels = []

stop_set = set(stopwords.words("english"))
prediction_label_dict = {0:"negative", 1:"neutral", 2:"positive"}


#Training data - Opening the data and preprocessing
with open('training_output.txt', 'w', encoding='utf8') as writer:
    with open('twitter-training-data.txt', 'r', encoding='utf8') as read_file:
        for line in csv.reader(read_file, delimiter='\t'):
            line[2] = line[2].lower()

            # Removing the \n new line and \r carriage return
            line[2] = re.sub('\\r', ' ', line[2])
            line[2] = re.sub('\\n', ' ', line[2])

            # Removes all non ASCII characters i.e. \u00a0
            line[2] = re.sub(r'[^\x00-\x7F]+', ' ', line[2])
            #Attempting to remove URLs
            line[2] = re.sub(
                '((http|https):\/\/)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-]){2,}', '',
                line[2])
            # Replacing acronyms first, i.e. replaces U.K with UK
            line[2] = re.sub(r'(?<!\w)([a-z])\.', r'\1', line[2])
            #Remove @user mentions
            line[2] = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z0-9_]+)', ' ', line[2])
            #Removing non alphanumeric characters
            line[2] = re.sub(r'[^a-zA-Z0-9\s]', ' ', line[2])
            #Removing single letter words
            line[2] = re.sub(r'\b[a-zA-Z]\b', '', line[2])
            #Removing pure digits
            line[2] = re.sub(r'\b\d+\b', '', line[2])

            lmzr = WordNetLemmatizer()
            temp = line[2].split()
            # lemmatise nouns and verbs and remove stopwords
            line[2] = [lmzr.lemmatize(lmzr.lemmatize(x, pos="v"), pos="n") for x in temp if x not in stop_set]


            #append current split processed tweet to string_testing list
            string_training.append(line[2])

            writer.write(str(line[2]))
            writer.write('\n')


            #For naive bayes classifier: Have 3 lists, each for containing the tweets for each sentiment
            if line[1] == "positive":
                string_training_labels.append(2) #Append a label '2' to the positive tweet. String training labels stores the labels of all tweets
            elif line[1] == "neutral":
                string_training_labels.append(1)
            elif line[1] == "negative":
                string_training_labels.append(0)

test_ids = [] #Have all the ID's of the test set in order

#Test data - open it and process it in the same way as above
with open('test_output.txt', 'w', encoding='utf8') as writer:
    with open(testsets.testsets[0], 'r', encoding='utf8') as read_file:
        for line in csv.reader(read_file, delimiter='\t'):
            test_ids.append(line[0])
            line[2] = line[2].lower()

            # Removing the \n new line and \r carriage return
            line[2] = re.sub('\\r', ' ', line[2])
            line[2] = re.sub('\\n', ' ', line[2])

            # Removes all non ASCII characters i.e. \u00a0
            line[2] = re.sub(r'[^\x00-\x7F]+', ' ', line[2])
            #Attempting to remove URLs
            line[2] = re.sub(
                '((http|https):\/\/)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-]){2,}', '',
                line[2])
            # Replacing acronyms first, i.e. replaces U.K with UK
            line[2] = re.sub(r'(?<!\w)([a-z])\.', r'\1', line[2])
            #Remove @user mentions
            line[2] = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z0-9_]+)', ' ', line[2])
            #Removing non alphanumeric characters
            line[2] = re.sub(r'[^a-zA-Z0-9\s]', ' ', line[2])
            #Removing single letter words
            line[2] = re.sub(r'\b[a-zA-Z]\b', '', line[2])
            #Removing pure digits
            line[2] = re.sub(r'\b\d+\b', '', line[2])

            lmzr = WordNetLemmatizer()
            temp = line[2].split()
            # lemmatise nouns and verbs and remove stopwords
            line[2] = [lmzr.lemmatize(lmzr.lemmatize(x, pos="v"), pos="n") for x in temp if
                       x not in stop_set]

            #append current split processed tweet to string_testing list
            string_testing.append(line[2])

            writer.write(str(line[2]))
            writer.write('\n')

            #For naive bayes classifier: A list of 3 dictionaries, each for containing the tweets for each sentiment
            if line[1] == "positive":
                string_testing_labels.append(2)
            elif line[1] == "neutral":
                string_testing_labels.append(1)
            elif line[1] == "negative":
                string_testing_labels.append(0)



print("Training naivebayes")
# Create 3 dictionaries like the default thing
num_classes = 3
# Create a list of 3 dictionaries for each of the classes
class_dictionaries = [defaultdict(int) for _ in range(num_classes)]
word_totals = defaultdict(int)
for training_line, label in zip(string_training, string_training_labels):
    # label is either 0, 1 or 2 so the index of the class dictionary
    for word in training_line:
        class_dictionaries[label][word] += 1 #finds word in corresponding dictionary and adds 1 to count i.e. 'apple' has count 43 in positive, 40 in neutral, 20 in negative
        word_totals[word] += 1 #counts the total number of times word has come up regardless of class


#BAYESCLASSIFIER
#uses a trained dictionary to work out probability of string of words being in a class
def naivebayes(class_dictionaries, word_list) -> int:
    probabilities = [1 for _ in range(num_classes)]
    for word in word_list:
        if word in word_totals: #makes sure that the test word is in training data to avoid 0/0
            probabilities = [probabilities[i]*class_dictionaries[i][word]/word_totals[word] for i in range(num_classes)] #freq of word in sentiment / freq of word in total, multiplies each sentence of tweet together
    return np.argmax(probabilities) #picks largest value returned for probability generated by each sentiment

# Testing the test set accuracy
score = 0
predicted_label_naivebayes = []
for testing_line, label in zip(string_testing, string_testing_labels): #for each tweet in test data, with its label (sentiment)
    predicted_label = naivebayes(class_dictionaries, testing_line) #This gives the test tweet a predicted sentiment label
    predicted_label_naivebayes.append(predicted_label)
    if predicted_label == label: #If the predicted label matches its real label, add 1 to the score
        score += 1
accuracy = score/len(string_testing_labels) #Accuracy is the score divided by the total number of test tweets
predicted_label_naivebayes = np.array(predicted_label_naivebayes)
naivebayes_id_preds = {test_id: prediction_label_dict[prediction] for test_id, prediction in zip(test_ids, predicted_label_naivebayes)}
evaluation.confusion(naivebayes_id_preds, testsets.testsets[0], "naivebayes")
evaluation.evaluate(naivebayes_id_preds, testsets.testsets[0], "naivebayes")
print("The accuracy of naivebayes is {0:.3f}".format(accuracy))


#SVM CLASSIFIER
print("")
print("Training SVM")
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
processed_training_tweets = [' '.join(i) for i in string_training]
processed_testing_tweets = [' '.join(i) for i in string_testing]
train_vectors = vectorizer.fit_transform(processed_training_tweets)
test_vectors = vectorizer.transform(processed_testing_tweets)

SVM_classifier_SGD = SGDClassifier()
SVM_classifier_SGD.fit(train_vectors, string_training_labels)
SVM_prediction_SGD = SVM_classifier_SGD.predict(test_vectors)

#puts prediction into prediction_label_dict to extract sentiment in word form to suit evaluation.py
svm_id_preds = {test_id: prediction_label_dict[prediction] for test_id, prediction in zip(test_ids, SVM_prediction_SGD)} #test ID followed by its predicted sentiment (for evaluation later)
evaluation.confusion(svm_id_preds, testsets.testsets[0], "SVM")
evaluation.evaluate(svm_id_preds, testsets.testsets[0], "SVM")
print("The accuracy of SVM is {0:.3f}".format(accuracy_score(string_testing_labels, SVM_prediction_SGD)))


#LSTM CLASSIFIER
print("")
print("Training LSTM")
#We want an embedding matrix, i.e. an array where each row represents each word from the training data, with the embedding information (from glove)
max_words = 5000
embedding_dim = 100
max_tweet_length = 100


#make a dictionary embeddings which has the glove embedding for every token in word_totals
with open('glove.6B.100d.txt', 'r', encoding='utf8') as read_file:
    embeddings = {}
    for line in csv.reader(read_file, delimiter=' ', quotechar = None):
        if line[0] in word_totals:
            embeddings[line[0]] = line[1:]
#sort and take the embeddings for the (max_words) most frequent tokens
tokens = sorted(embeddings.keys(), key = lambda x:word_totals[x], reverse = True)[:max_words]
token_dict = {"":0} #0th entry is a dummy string, we want 1-5000th entry, not 0-4999th
#Create numpy array where each row represents each word in the training data with the embedding information (from glove)
embedding_matrix = np.zeros((max_words + 1, embedding_dim))  #numpy array, takes a tuple as information, creates an array of zeros, max_words = row number, embedding_dim = column number
for i in range(len(tokens)):
    token = tokens[i]
    embedding_matrix[i + 1, :] = embeddings[token] #We are attempting to update each row of the array. Colon selects all columns of the row
    token_dict[token] = i + 1 #keeps track of what order we added tokens, so we can get the corresponding row for each word in the embedding matrix later


x_train = pad_sequences([[token_dict.get(x, 0) for x in tweet] for tweet in string_training] , maxlen=max_tweet_length) #words that are not in token dictionary get 0

model = Sequential()
model.add(Embedding(max_words + 1, embedding_dim, weights=[embedding_matrix], input_length=max_tweet_length, trainable=False))
model.add(LSTM(100))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])
model.fit(x_train, string_training_labels, epochs=4, batch_size=64, verbose=0)

x_test = pad_sequences([[token_dict.get(x, 0) for x in tweet] for tweet in string_testing] , maxlen=max_tweet_length)
scores = model.evaluate(x_test, string_testing_labels, verbose=0) #scores will be used for calculating accuracy

predicted_label_LSTM = []
for prediction in model.predict(x_test):
    if prediction < 2/3:
        predicted_label_LSTM.append(0)
    elif prediction > 4/3:
        predicted_label_LSTM.append(2)
    else: predicted_label_LSTM.append(1)

lstm_id_preds = {test_id: prediction_label_dict[prediction] for test_id, prediction in zip(test_ids, predicted_label_LSTM)}
evaluation.confusion(lstm_id_preds, testsets.testsets[0], "LSTM")
evaluation.evaluate(lstm_id_preds, testsets.testsets[0], "LSTM")
print("The accuracy of LSTM is {0:.3f}".format(scores[1]))