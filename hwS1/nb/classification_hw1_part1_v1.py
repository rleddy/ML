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
	"me","i","my",'myself','we','our','ours','ourselves',
	'he','him','his','himself','she',"she's",'her','hers','herself','it', "it's",
	'its','itself','they','them','their','theirs','themselves','what','which',
	"who",'whom','this','that',"that'll","these",'those','am','is','are','was',
	'were','be','been','being','have','has','had','having','do','does','did',
	'doing','a','an','the','and','but','if','or','because','as','until', 'while',
	'of','at','by','for','with','about','against','between','into','through',
	'during','before','after','above','below','to','from','up','down','in',
	'out','on','off','over','under','again','further','then','once','here',
	'there','when','where','why','how','all','any','both','each','few','more',
	'most','other','some','such','no','nor','not','only','own','same','so',
	'than','too','very','s','t','can','will','just','don',"don't",'should',
	"should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",
	'couldn',"couldn't",'didn',"didn't",'doesn',"isn't",'ma','mightn',
	"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',
	"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',
	"wouldn't"
]

### Note: The code for each algorithm is be 20-30 lines, with additional lines
# for simple I/O and boilerplate code.

# Defining the function called, load_data, to load data:
def load_data(filename, train = True):
	file = open(filename)
	lines = iter(file.readlines()) #The iter() method creates an object which
	# can be iterated one element at a time
	next(lines)
	content_result = []
	comment = []
	i = 0
	if train == False:
		file_io = open(filename)
		reader= csv.reader(file_io)
		for line in reader:
			if i > 0:
				comment.append(line[1])
			i += 1
#
# 	for line in lines:
# 		if train:
# 			insult_or_not = int(line[0])
#
# 			text = line[5:-4]
# 			#print("Insult = 1, or Not Insult = 0: ", insult_or_not)
# 			#print("Comment: ", text)
# 			content_result.append([insult_or_not, text])
# 		else:
# 			text=line[4:-4]
# 			#print("Insult = 1, or Not Insult = 0: ", insult_or_not)
# 			#print("Comment: ", text)
# 			content_result.append([0, text])
# 	#print(content_result)
# 	return content_result, comment
#
# def load_numeric(filename):
# 	file = open(filename)
# 	reader = csv.reader(file)
# 	data = []
# 	target = []
# 	for line in reader:
# 		target.append(int(line[-1]))
# 		data.append([float(i) for i in line])
# 	return data, target
#
# # [[label, text], [], []]
# train_filename = 'train.csv'
# train_lines, _ = load_data(train_filename)
#
# feature_words = []
# word_to_prob = {}
#
#
# def allowed_words(alpha = 0):
# 	for line in train_lines:
# 		comment = line[1]
# 		comment_list = tokenizer(comment)
# 		for each_word in comment_list:
# 			feature_words.append(each_word)
# 	word_to_count = Counter(feature_words)
# 	for word, count in word_to_count.items():
# 		word_to_count[word]+=alpha  ## == (n_ij + alpha) == numerator
# 	total_counts=float(sum(word_to_count.values()))
# 	##Since we've added alpha to all of the frequencies, total_counts
# 	# implicitly is sum_freq + alpha*V
# 	for word, count in word_to_count.items():
# 		word_to_prob[word]=count/total_counts
# 	return word_to_prob
#
#
# def tokenizer(text):
# 	tokens = re.split(r'\W+', text)
# 	tokens = [tk.lower() for tk in tokens]
# 	regex_punctuations = re.compile('[%s]' % re.escape(string.punctuation))
# 	tokens = [regex_punctuations.sub('', tk) for tk in tokens]
#
# 	tokens = [tk for tk in tokens if tk not in nltk_like_sw and len(tk)
# 				  > 1]
# 	regex_printable = re.compile('[^%s]' % re.escape(string.printable))
# 	tokens = [regex_printable.sub('', tk) for tk in tokens]
# 	tokens = [token for token in tokens if token.isalpha()]
# 	tokens = [token for token in tokens if len(token) > 3]
# 	#tokens = [token for token in tokens if token
# 	print(tokens)
# 	if len(tokens) == 0:
# 		tokens = ["nothing"]
# 	return tokens
#
# if generate_numeric_data:
# 	allowed_words()
#
#
# # spn_chars = ['á', 'é', 'í', 'ó', 'ú', 'ñ', 'ñ', 'ü']
#
#
#
#
#
# def bagofwords(sentence,words):
# 	sentence_words=tokenizer(sentence)
# 	#frequency word count
# 	# bag=np.zeros(len(words))
# 	# for sw in sentence_words:
# 	#  	i = 0
# 	#  	if sw in words:
# 	#  		bag[i]+= words[sw]
# 	#  	for word in words:
# 	#  		if word == sw:
# 	#  			bag[i] += words[word]
# 	#  		i += 1
#
# 	return [words.get(sw, 0) for sw in sentence_words]
#
# 	# return list(bag)
#
# test_comments = []
# if generate_numeric_data:
# 	for input_filename, output_filename, train in [("train.csv", "train_clean.csv",True), ("dev.csv", "dev_clean.csv", True),
# 											("test.csv", "test_clean.csv", False)]:
# 		input, comments = load_data(input_filename, train)
# 		test_comments = comments
# 		file = open(output_filename, "w")
# 		writer = csv.writer(file)
# 		for line in input:
# 			comment=line[1]
# 			comment_list = []
# 			comment_list=bagofwords(comment, word_to_prob)
# 			if len(comment_list) > 0:
# 				comment_list.append(line[0])
# 				writer.writerow(comment_list)
#
# 		file.close()
#
# # clean_data = load_data("train_clean.csv")
#
#
# ### (2) Training ###
# # Do I need a separate function, called, def vectorize(): ???
#
#
# # Creating a function that separates and make the list of the lists either for insult/ not insult.
# def separateByClass(dataset):
# 	separated = {}
# 	for i in range(len(dataset)):
# 		vector = dataset[i]
# 		if (vector[-1] not in separated):
# 			separated[vector[-1]] = []
# 			# print (vector[-1])
# 		separated[vector[-1]].append(vector)
# 	return separated
#
# ## Now creating classifier:
# #(1) Training portion are: def mean and def stdev
# def mean(numbers):
# 	# print (numbers)
# 	return sum(numbers)/float(len(numbers))
#
#
# def stdev(numbers):
# 	# print (numbers)
# 	avg = mean(numbers)
# 	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
# 	return math.sqrt(variance)
#
#
# def summarize(dataset):
# 	summaries = [(mean(attribute),stdev(attribute)) for attribute in
# 				 zip(*dataset)]
# 	del summaries[-1]
# 	return summaries
#
#
# def summarizeByClass(dataset):
# 	separated = separateByClass(dataset)
# 	# print (separated.keys())
# 	summaries = {}
# 	for classValue,instances in separated.items():
# 		summaries[classValue] = summarize(instances)
# 	return summaries
#
#
# # Prior calculation --> training dataset's mean values as the prior
# def calculateProbability(x,mean,stdev):
# 	# print (stdev)
# 	exponent = math.exp(-(math.pow(x - mean,2) / (2 * math.pow(stdev,2))))
# 	return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
#
#
# # Posterior calculation
# # classSummaries --> comes from summarizeByClass (mean + std)
# # classValue --> Target value
#
# def calculateClassProbabilities(summaries,inputVector):
# 	probabilities = {}
# 	for classValue,classSummaries in summaries.items():
# 		probabilities[classValue] = 1
# 		for i in range(len(classSummaries)):
# 			mean,stdev = classSummaries[i]
# 			x = inputVector[i]
# 			probabilities[classValue] *= calculateProbability(x,mean,stdev)
# 	return probabilities
# ### The end of training portion ###
#
# # Below are the actual classifier portion:
# # Use development data to run the classifer.
# def predict(summaries,inputVector):
# 	probabilities = calculateClassProbabilities(summaries, inputVector)
# 	bestLabel,bestProb = None,-1
# 	for classValue,probability in probabilities.items():
# 		if bestLabel is None or probability > bestProb:
# 			bestProb = probability
# 			bestLabel = classValue
# 	return bestLabel
#
#
# # Predictions:
# def getPredictions(summaries,testSet):
# 	predictions = []
# 	for i in range(len(testSet)):
# 		result = predict(summaries,testSet[i])
# 		predictions.append(result)
# 	return predictions
#
#
# def getAccuracy(testSet,predictions):
# 	correct = 0
# 	for i in range(len(testSet)):
# 		if testSet[i][-1] == predictions[i]:
# 			correct += 1
# 	return (correct / float(len(testSet))) * 100.0
#
#
# # Run all the functions above
# def main():
# 	filename = 'train_clean.csv'
# 	print("Training (train_clean.csv) generated...")
# 	trainingSet, _ = load_numeric(filename)
# 	# trainingSet,testSet=splitDataset(dataset,splitRatio)
# 	devSet, target_dev = load_numeric("dev_clean.csv")
# 	summaries = summarizeByClass(trainingSet)
# 	print("Development (dev_clean.csv) generated....")
# 	predictions = getPredictions(summaries, devSet)
# 	accuracy = getAccuracy(devSet, predictions)
# 	print('Accuracy Score Is: {0}%'.format(accuracy))
# 	testSet, _ = load_numeric("test_clean.csv")
# 	file_out = open("prediction.csv", "w")
# 	writer = csv.writer(file_out)
# 	print("Testing/Predicting (prediction.csv) generated....")
# 	predictions_test = getPredictions(summaries, testSet)
# 	writer.writerow(["Insult", "Comment"])
# 	for pred, comment in zip(list(predictions_test), test_comments):
# 		writer.writerow([int(pred), comment])
# 	file_out.close()
# 	print(f1_score(target_dev,predictions))
# 	print(len(set(word_to_prob.keys())))
#
# main()
#
#
#
#
