import csv
import re
import string
import random
import math
import itertools
import pdb
import argparse
from collections import Counter
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score

generate_numeric_data = True

nltk_like_sw = [
			"me", "i", 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
			'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
			'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
			"she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
			'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
			'who', 'whom', 'this', 'that', "that'll", "these", 'those', 'am',
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
			'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
			'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
			'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
			'through', 'during', 'before', 'after', 'above', 'below', 'to',
			'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
			'again', 'further', 'then', 'once', 'here', 'there', 'when',
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
			'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
			'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
			'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
			'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
			"couldn't", 'didn', "didn't", 'doesn', "isn't", 'ma', 'mightn',
			"mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
			'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
			"won't", 'wouldn', "wouldn't"
]


### Note: The code for each algorithm is be 20-30 lines, with additional lines
# for simple I/O and boilerplate code.

# Defining the function called, traindata_load, to load data:
def traindata_load(filename):
	file = open(filename)
	lines = iter(file.readlines()) #The iter() method creates an object which
	# can be iterated one element at a time
	columns = next(lines)
	content_result = []
	
	for line in lines:
		insult_or_not = int(line[0])
		text = line[5:-4]
		#print("Insult = 1, or Not Insult = 0: ", insult_or_not)
		#print("Comment: ", text)
		content_result.append([insult_or_not, text])
	#print(content_result)
	return content_result

def load_numeric(filename):
	file = open(filename)
	reader = csv.reader(file)
	data = []
	for line in reader:
		data.append([float(i) for i in line])
	return data

# [[label, text], [], []]
#train_filename = 'train.csv'
#train_lines = traindata_load(train_filename)

feature_words = []
word_to_prob = {}


def allowed_words(train_lines,alpha = 0):
    #
	for line in train_lines:
		#comment = line[1]
		#comment_list = tokenizer(comment)
		comment_list = line[1]
		feature_words.extend(comment_list)
    
    
	word_to_count = Counter(feature_words)
	for word, count in word_to_count.items():
		word_to_count[word]+=alpha  ## == (n_ij + alpha) == numerator
    #
	total_counts = float(sum(word_to_count.values()))
	##Since we've added alpha to all of the frequencies, total_counts
	# implicitly is sum_freq + alpha*V
	for word, count in word_to_count.items():
		word_to_prob[word]=count/total_counts
    #
	return word_to_prob

    
    

##
regex_punctuations = re.compile('[%s]' % re.escape(string.punctuation))
##

def tokenizer(text):
    global regex_punctuations
    #
    tokens = re.split(r'\W+', text)
    tokens = [tk.lower() for tk in tokens]
    
    tokens = [regex_punctuations.sub('', tk) for tk in tokens]
    
    tokens = [tk for tk in tokens if tk not in nltk_like_sw and len(tk) > 1]
    
    regex_printable = re.compile('[^%s]' % re.escape(string.printable))
    tokens = [regex_printable.sub('', tk) for tk in tokens]
    #print(tokens)
    return tokens
##


#if generate_numeric_data:
	#allowed_words()
#

#my_dictionary = {"foo":"bar"}
#value = my_dictionary["foo"] # bar
#value = my_dictionary["baz"] # KeyError


def bagofwords(sentence,words):
	sentence_words=tokenizer(sentence)
	# frequency word count
	# bag=np.zeros(len(words))
	# for sw in sentence_words:
	# 	i = 0
	# 	if sw in words:
	# 		bag[i]+= words[sw]
	# 	for word in words:
	# 		if word == sw:
	# 			bag[i] += words[word]
	# 		i += 1
	
	return [words.get(sw, 0) for sw in sentence_words]
	
	# return list(bag)

#if generate_numeric_data:
	#for input_filename, output_filename in [("train.csv", "train_clean.csv"), ("dev.csv", "dev_clean.csv")]:
		#input = traindata_load(input_filename)
		#file = open(output_filename, "w")
		#writer = csv.writer(file)
		#for line in input:
			#comment=line[1]
			#comment_list = []
			#comment_list=bagofwords(comment, word_to_prob)
			#if len(comment_list) > 0:
				#comment_list.append(line[0])
				#writer.writerow(comment_list)
		
		#file.close()

# clean_data = traindata_load("train_clean.csv")


### (2) Training ###
# Do I need a separate function, called, def vectorize(): ???


# Creating a function that separates and make the list of the lists either for insult/ not insult.
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
			# print (vector[-1])
		separated[vector[-1]].append(vector)
	return separated

## Now creating classifier:
#(1) Training portion are: def mean and def stdev
def mean(numbers):
	# print (numbers)
	return sum(numbers)/float(len(numbers))


def stdev(numbers):
	# print (numbers)
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)


def summarize(dataset):
	summaries = [(mean(attribute),stdev(attribute)) for attribute in
				 zip(*dataset)]
	del summaries[-1]
	return summaries


def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	# print (separated.keys())
	summaries = {}
	for classValue,instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries


# Prior calculation --> training dataset's mean values as the prior
def calculateProbability(x,mean,stdev):
	print (stdev)
	exponent = math.exp(-(math.pow(x - mean,2) / (2 * math.pow(stdev,2))))
	return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# Posterior calculation
# classSummaries --> comes from summarizeByClass (mean + std)
# classValue --> Target value

def calculateClassProbabilities(summaries,inputVector):
	probabilities = {}
	for classValue,classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean,stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x,mean,stdev)
	return probabilities
### The end of training portion ###

# Below are the actual classifier portion:
# Use development data to run the classifer.
def predict(summaries,inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel,bestProb = None,-1
	for classValue,probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel


# Predictions:
def getPredictions(summaries,testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries,testSet[i])
		predictions.append(result)
	return predictions


def getAccuracy(testSet,predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct / float(len(testSet))) * 100.0


# Run all the functions above
#def main():
	#filename = 'train_clean.csv'
	#trainingSet = load_numeric(filename)
	## trainingSet,testSet=splitDataset(dataset,splitRatio)
	#devSet = load_numeric("dev_clean.csv")
	#summaries = summarizeByClass(trainingSet)
	#predictions = getPredictions(summaries, devSet)
	#print (predictions)
	#accuracy = getAccuracy(devSet, predictions)
	#print('Accuracy: {0}%'.format(accuracy))
	
###



def tokenize(irItr):
    transformedRow = []
    for row in irItr:
        rtokens = tokenizer(row[1])
        transformedRow.append([int(row[0]),rtokens])
    return transformedRow


def loadDataToItr(aFile,dataTransform):
    with open(aFile) as csvfile:
        itrRead = csv.reader(csvfile)
        itrRead.next()   # skip header
        return dataTransform(itrRead)
    return None



vocab = set([])
def train(labeledRows,smoothing_alpha = 0):
    global vocab
    #
    c1_count = 0
    c2_count = 0
    c1_text = []
    c2_text = []
    c1_hist = {}
    c2_hist = {}
    c1_probs_per_word = {}
    c2_probs_per_word = {}
    for row in labeledRows:
        classId = row[0]
        if (classId == 0:
            c1_count += 1
            c1_text.extend(row[1])
        else:
            c2_count += 1
            c2_text.extend(row[1])
            
        for word in row[1]:
            vocab.add(word)  # instead of using a set
            if (classId == 0):
                if ( c1_hist[word] == None ):
                     c1_hist[word] = 1
                else:
                     c1_hist[word] += 1
            else:
                if ( c2_hist[word] == None ):
                     c2_hist[word] = 1
                else:
                     c2_hist[word] += 1
    #
    n1 = len(c1_text)
    n2 = len(c2_text)
    #
    N = c1_count + c2_count
    P1 = c1_count/N
    P2 = c2_count/N
    
    N_vocab = len(vocab)
    
    for w in vocab:
        n_k = c1_hist[w]
        c1_probs_per_word[w] = (n_k + smoothing_alpha)/(n1 + N_vocab)
        #
        n_k = c2_hist[w]
        c2_probs_per_word[w] = (n_k + smoothing_alpha)/(n2 + N_vocab)
    
    return P1, P2, c1_probs_per_word, c2_probs_per_word


#
##
###
def classisfy(docTokens,P1, P2, c1_probs_per_word, c2_probs_per_word):
    docProb1 = P1  #initialize
    docProb2 = P2
    for w in docTokens:  # go through all positions in the doc
        docProb1 *= c1_probs_per_word[w]
        docProb2 *= c2_probs_per_word[w]

    if ( docProb1 > docProb2 ):
        return 0
    else:
        return 1




###




def gradient(X_transpose,Y,Y_hat):
    Ydiff = Y - Y_hat
    G = np.dot(X_transpose,Ydiff);
    return G




def printLabeledArray(labData):
    for row in labData:
        print row

def main():
    trainData = loadDataToItr('train.csv',tokenize)
    #
    printLabeledArray(trainData)
#
#
#

######### ##########################################

# Logistic Regression on Diabetes Dataset
from random import seed
from random import randrange
from csv import reader
from math import exp

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Make a prediction with coefficients
#
def sigmoid(X):
    X_hat = -X
    S = 1.0/(1.0 + np.exp(X_hat))
    return S
#
#
def Predict(x_train_aug,coefficients):    # x_train_aug has 1's in the first column
    y_hat = np.dot(x_train_aug,coefficients)   # matrix by column
    return sigmoid(y_hat);

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(y_train, x_train, l_rate, n_epoch):
    #
    x_train_aug = np.array(x_train)
    x_train_aug = np.insert(x_train_aug, 0, 1, axis=1)
    Y_train = np.array(y_train)
    #
	coefs = np.array([0.0 for i in range(len(x_train[0]) + 1)])  #  all coeffs and y intercept
	#
	for epoch in range(n_epoch):
        #
        Y_hat = Predict(x_train_aug,coefs)
        error_bar = (Y_train - Y_hat)
        yProb = error_bar # np.multiply(error_bar,np.multiply(Y_hat,(1.0 - Y_hat)))
        #
        coefs += l_rate*np.dot(x_train.T,yProb)
    #
	return coef

# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(y_train, x_train, test, l_rate, n_epoch):
	#
	coef = coefficients_sgd(y_train, x_train, l_rate, n_epoch)
	#
	predictions = list()
	#
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
		predictions.append(yhat)
    #
	return(predictions)



# Test the logistic regression algorithm on the diabetes dataset
seed(1)
# load and prepare data
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# normalize
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.1
n_epoch = 100
scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))




    
main()


#import csv
#with open('eggs.csv', 'w', newline='') as csvfile:
    #spamwriter = csv.writer(csvfile, delimiter=' ',
                            #quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    #spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

