# https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac 
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

generate_numeric_data=True

nltk_like_sw=['it',"it's",'its','itself','they','them','their','theirs',
	'themselves','what','which','who','whom','this','that',"that'll","these",
	'those','am','is','are','was','were','be','been','being','have','has',
			  'had',
	'having','do','does','did','doing','a','the','and','but','if','or',
	'because','as','until','while','of','at','by','for','with','about',
	'between','into','through','during','before','after','above','below','to',
	'from','up','down','in','out','on','off','over','under','again','further',
	'then','once','here','there','when','where','why','how','all','any','both',
	'each','few','more','most','other','some','such','only','own','same','so',
	'than','s','t','can','will','just','don',"don't",'should',"should've",
			  'now',
	'd','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',"couldn't",
	'didn',"didn't",'doesn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",
	'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",
	'weren',"weren't",'won',"won't",'wouldn',"wouldn't",'nadie','se','salva',
	'de','la','regla','34','xd']

def traindata_load(filename,train=True):
	file=open(filename)
	lines=iter(file.readlines())  #The iter() method creates an object which
	# can be iterated one element at a time
	columns=next(lines)
	content_result=[]
	comment=[]
	i=0
	if train==False:
		file_io=open(filename)
		reader=csv.reader(file_io)
		for line in reader:
			if i>0:
				comment.append(line[1])
			i+=1
	
	for line in lines:
		if train:
			insult_or_not=int(line[0])
			
			text=line[5:-4]
			#print("Insult = 1, or Not Insult = 0: ", insult_or_not)
			#print("Comment: ", text)
			content_result.append([insult_or_not,text])
		else:
			text=line[4:-4]
			#print("Insult = 1, or Not Insult = 0: ", insult_or_not)
			#print("Comment: ", text)
			content_result.append([0,text])
	#print(content_result)
	return content_result,comment


def load_numeric(filename):
	file=open(filename)
	reader=csv.reader(file)
	data=[]
	target=[]
	for line in reader:
		target.append(int(line[-1]))
		data.append([float(i) for i in line])
	return data,target


# [[label, text], [], []]
train_filename='train.csv'
#train_lines,_=traindata_load(train_filename)

feature_words=[]
word_to_prob={}


def allowed_words(alpha=0):
	for line in train_lines:
		comment=line[1]
		comment_list=tokenizer(comment)
		for each_word in comment_list:
			feature_words.append(each_word)
	word_to_count=Counter(feature_words)
	for word,count in word_to_count.items():
		word_to_count[word]+=alpha  ## == (n_ij + alpha) == numerator
	total_counts=float(sum(word_to_count.values()))
	##Since we've added alpha to all of the frequencies, total_counts
	# implicitly is sum_freq + alpha*V
	for word,count in word_to_count.items():
		word_to_prob[word]=count/total_counts
	return word_to_prob


def tokenizer(text):
	tokens=re.split(r'\W+',text)
	tokens=[tk.lower() for tk in tokens]
	regex_punctuations=re.compile('[%s]'%re.escape(string.punctuation))
	tokens=[regex_punctuations.sub('',tk) for tk in tokens]
	
	tokens=[tk for tk in tokens if tk not in nltk_like_sw and len(tk)>1]
	regex_printable=re.compile('[^%s]'%re.escape(string.printable))
	tokens=[regex_printable.sub('',tk) for tk in tokens]
	tokens=[token for token in tokens if token.isalpha()]
	tokens=[token for token in tokens if len(token)>3]
	print(tokens)
	if len(tokens)==0:
		tokens=["nothing"]
	return tokens

#if generate_numeric_data:
#	allowed_words()


def bagofwords(sentence,words):
	sentence_words=tokenizer(sentence)
	# frequency word count
	bag=[0]*len(word_to_prob)
	for sw in sentence_words:
		if sw in word_to_prob:
	 		bag[list(word_to_prob.keys()).index(sw)] += word_to_prob[sw]
	
	return bag


# return list(bag)

# test_comments=[]
# if generate_numeric_data:
# 	for input_filename,output_filename,train in [
# 		("dev.csv","dev_clean.csv",True),
# 		("test.csv","test_clean.csv",False)]:
# 		input,comments=traindata_load(input_filename,train)
# 		test_comments=comments
# 		file = open(output_filename,"w")
# 		import csv
# 		writer = csv.writer(file)
# 		for line in input:
# 			comment=line[1]
# 			comment_list=[]
# 			comment_list=bagofwords(comment,word_to_prob)
# 			if len(comment_list)>0:
# 				comment_list.append(line[0])
# 				writer.writerow(comment_list)
#
# 		file.close()


class LogisticRegression:
	def __init__(self,learning_rate=0.00005,num_steps=300000,fit_intercept=True,
				 verbose=True):
		self.learning_rate=learning_rate
		self.num_steps=num_steps
		self.fit_intercept=fit_intercept
		self.verbose = verbose
	
	def __add_intercept(self,X):
		intercept=np.ones((X.shape[0],1))
		return np.concatenate((intercept,X),axis=1)
	
	def __sigmoid(self,z):
		return 1/(1+np.exp(-z))
	
	def __loss(self,h,y):
		return (-y*np.log(h)-(1-y)*np.log(1-h)).mean()
	
	def fit(self,X,y):
		if self.fit_intercept:
			X=self.__add_intercept(X)
		
		# weights initialization
		self.theta=np.zeros(X.shape[1])
		
		for i in range(self.num_steps):
			z=np.dot(X,self.theta)
			h=self.__sigmoid(z)
			gradient=np.dot(X.T,(h-y))/y.size
			self.theta-=self.learning_rate*gradient
			
			if (self.verbose==True and i%10000==0):
				z=np.dot(X,self.theta)
				h=self.__sigmoid(z)
				print(f'loss: {self.__loss(h,y)} \t')
	
# ax + by + cz ....
# 3x + 2y = 6
	def predict_prob(self,X):
		if self.fit_intercept:
			X=self.__add_intercept(X)
		
		return self.__sigmoid(np.dot(X,self.theta))
	
	def predict(self,X,threshold):
		return self.predict_prob(X)>=threshold
	
import csv
model = LogisticRegression()
file_io = open("train_clean.csv")
read = csv.reader(file_io)
X = []
y = []
for line in read:
	X.append([float(i) for i in line[:-1]])
	y.append(int(line[-1]))
X= np.array(X)
y = np.array(y)
model.fit(X, y)
X = []
y = []
file_io = open("dev_clean.csv")
read = csv.reader(file_io)
for line in read:
	X.append([float(i) for i in line[:-1]])
	y.append(int(line[-1]))
X= np.array(X)
y = np.array(y)
pred1 = model.predict_prob(X)
thres = pred1.mean()
print (pred1)
pred = model.predict(X, thres)

from sklearn.metrics import accuracy_score
print (accuracy_score(y, pred))


X = []
y = []
file_io = open("test_clean.csv")
read = csv.reader(file_io)
for line in read:
	X.append([float(i) for i in line[:-1]])
X= np.array(X)
predictions_test = model.predict(X, thres)

file_out=open("prediction.csv","w")
writer=csv.writer(file_out)
writer.writerow(["Insult","Comment"])

test_comments = []
file_io=open("test.csv")
reader=csv.reader(file_io)
i = 0
for line in reader:
	if i>0:
		test_comments.append(line[1])
	i+=1
		
for pred,comment in zip(list(predictions_test),test_comments):
	print(comment)
	writer.writerow([int(pred),comment])
file_out.close()