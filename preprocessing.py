#!/usr/bin/python
# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
from nltk.tokenize import sent_tokenize, word_tokenize

def removepunc(s):
        #URLless_string = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', s)
        punctuations = '''-[]{}'"\,<>./?$%^&*_~'''
        no_punct = ""
        for char in s:
                if char not in punctuations:
                        no_punct = no_punct + char
        return no_punct
def preprocessing(tweet):
	text1=str(''.join([i if ord(i) < 128 else ' ' for i in tweet]))#remove characters having ACII values more than 127
	result = ''.join([i for i in text1 if not i.isdigit()])#remove digits
	#result= removepunc(result)
        result = re.sub("<.*?#@>","",result)
        result = re.sub(r"http\S+", "", result)
        #print result
        result=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",result).split())
        result= result.lstrip("RT")
        return result

def remove_stopwords(example_sent):
	stop_words = set(stopwords.words('english')) 
  	word_tokens = word_tokenize(example_sent) 
  	filtered_sentence = []
	filtered_sentence = [w for w in word_tokens if not w in stop_words] 
  	filtered_sentence=' '.join(filtered_sentence)
	removepuncreturn filtered_sentence

def xpreprocessing(df):
	
	l_text=[]
	
	for i,row in df.iterrows():
		text=row["full_text"]
		text=removepunc(text)
		text=preprocessing(text)
		text=remove_stopwords(text)
		
		l_text.append(text)
		
	df["preprocessed_text"]=l_text
	return df
	


