import nltk
import os
import time 
import codecs
from sklearn import feature_extraction
import mpld3
import json_csv as js
import preprocessor as p
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
#import preprocessing as pre
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import json
from collections import Counter
from datetime import datetime
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfTransformer

'''
def post_processing(df):
	#<--remove null--->
	df = df[pd.notnull(df['text'])]

	#<--stemming--->
	df[["preprocessed_text"]]=df[["preprocessed_text"]].fillna('')
	l_stem=[]
	for i,row in df.iterrows():
		sentence=row["preprocessed_text"]
		#stemmer=Porter2Stemmer()
		stemmer = PorterStemmer()
		tweet_stem=' '.join([stemmer.stem(word) for word in sentence.split(" ")])
		tweet_stem=tweet_stem.lower()#<--make it lower---->	
		l_stem.append(tweet_stem)
		#print i

	df["tweet_stem"]=l_stem
	#print "*************After stemming the dataframe*****"
	#print df.head()
	
	#<----remove less than 3--->

	df[["tweet_stem"]]=df[["tweet_stem"]].fillna('')
	tweet_stem=list(df["tweet_stem"])
	tweet_rm_3=[]
	for i in range(0,len(tweet_stem)):
		#print i
		tweet_rm_3.append(' '.join([w for w in tweet_stem[i].split() if len(w)>3]))
	df["tweet_rm_3"]=tweet_rm_3
	df = df[pd.notnull(df['text'])]
	
	return df




def removepunc(s):
	for c in string.punctuation:
		s= s.replace(c,"")
	return s
def remove_stopwords(example_sent):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(example_sent)
        filtered_sentence = []
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        filtered_sentence=' '.join(filtered_sentence)
        return filtered_sentence

def remove_numbers(text):
	output = re.sub(r'\d+', '', text)
	return output	
'''

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
	return filtered_sentence

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)#Remove all special characters, punctuation and spaces from string
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text	
	


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]

def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


#dictionary_event_details={}
t=0
def make_dictionary(hashtag,df_clus):
	dictionary_temporary={}
	list_events_loc=[]
	event_id=time.time()
	
	#print hashtag
	if len(hashtag)>=1:
		
		x=Counter(hashtag)
		y=x.most_common(3)
		#print y
		if len(y)==3:
			event_name=y[0][0].encode('utf-8')+" "+y[1][0].encode('utf-8')+" "+y[2][0].encode('utf-8')
		elif len(y)==2:
			event_name=y[0][0].encode('utf-8')+" "+y[1][0].encode('utf-8')
		else:
			event_name=y[0][0].encode('utf-8')
		
			
		#print event_name
		#print "hashtag%%%%%%%%%%",hashtag;input()




		
		global t;t=t+1
	
		with open("location_dictionary.json", 'r') as file:
			di = json.load(file)
		#print type(di);print di;input()
		
		for i,row in df_clus.iterrows():
				flag=0
				geo=row["geo"]#;print row["user"]["location"].encode('utf-8');input()
				location=row["user"]["location"]
		                
				if location:
					for key in di:
		                		if key in location:
		                        		lat_long=di[key]
							dictionary_temporary[row["id_str"]]={"id_str":row["id_str"],"latitude":lat_long[0],"longitude":lat_long[1],"ename":event_name,"created_at":row["created_at"],"location":location,"text":row["text"],"main_event":0,"user_name":row["user"]["name"],"follower_count":row["user"]["followers_count"],"retweet_count":row["retweet_count"]}
							break
					
								
	else:	
		'''
		docs=df_clus['preprocessed_text']
		cv=CountVectorizer(stop_words='english',ngram_range=(1,3))
		word_count_vector=cv.fit_transform(docs)
		print list(cv.vocabulary_.keys())[:5]
		tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
		tfidf_transformer.fit(word_count_vector)
		'''
		#Convert most freq words to dataframe for plotting bar plot
		corpus=list(df_clus['preprocessed_text'])		
		#top_words = get_top_n_words(corpus, n=5)
		top3_words = get_top_n3_words(corpus, n=5)
		#print top3_words;input()
		event_name=top3_words[0][0].encode('utf-8')
		with open("location_dictionary.json", 'r') as file:
			di = json.load(file)
		for i,row in df_clus.iterrows():
			location=row["user"]["location"]
			if location:
				for key in di:
					if key in location:
		                        		lat_long=di[key]
							dictionary_temporary[row["id_str"]]={"id_str":row["id_str"],"latitude":lat_long[0],"longitude":lat_long[1],"ename":event_name,"created_at":row["created_at"],"location":location,"text":row["text"],"main_event":0,"user_name":row["user"]["name"],"follower_count":row["user"]["followers_count"],"retweet_count":row["retweet_count"]}
							break

	return dictionary_temporary,event_name


	



#********************Program start****************
#convert csv to DataFrame
df=js.make_csv("mongo_dataRealTime.json")
#df=df[:1000]
#print df.shape

#print df.columns

#do preprocessing
#df=pre.xpreprocessing(df)
#df=post_processing(df)
'''
list_pre=[]
for i,row in df.iterrows():
	text=removepunc(row["text"])
	text= text.lstrip("RT")
	text=remove_stopwords(text)
	text=remove_numbers(text)
	list_pre.append(p.clean(text))

df["preprocessed_text"]=list_pre
'''
df['preprocessed_text'] = df['text'].map(lambda com : preprocessing(com))
df['preprocessed_text'] = df['preprocessed_text'].map(lambda com : clean_text(com))
df['preprocessed_text'] = df['preprocessed_text'].map(lambda com : remove_stopwords(com))
tweets=list(df["preprocessed_text"])

#print tweets[:5];input()
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tweets)

#vectorizer.get_feature_names()

#print(X.toarray())     
#print "Before Clustering##########"
#print time.time()



transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)
print(tfidf.shape )                        

from sklearn.cluster import KMeans

num_clusters = 5 #Change it according to your data.
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf)
clusters = km.labels_.tolist()


df["Cluster"]=clusters
#print df['Cluster']
#print "\n"
#print frame #Print the doc with the labeled cluster number.
#print "\n"
#print frame['Cluster'].value_counts() #Print the counts of doc belonging to each cluster.
#print df.head()



count = df["Cluster"].value_counts()
tuples = [tuple((x, y)) for x, y in count.items()]

print tuples
#x=sorted(tuples,reverse=True,key=itemgetter(1))
tuples_sorted=sorted(tuples,key=lambda x: x[1], reverse=True)

print tuples_sorted

list_clus_index=[]
for i in range(0,5):
	list_clus_index.append(tuples_sorted[i][0])

	
print list_clus_index


for i in range(0,len(list_clus_index)):
	clus_no=[list_clus_index[i]]
	df_clus=df[df["Cluster"].isin(clus_no)]
	#print df_clus.head();input()
	

	tweets=list(df_clus["text"])
	
	hashtag=[]
	for j,row in df_clus.iterrows():
		lis=row["entities"]["hashtags"]
		for e in lis:	
			hashtag.append(e["text"])
	
	
	dictionary_temporary,event_name=make_dictionary(hashtag,df_clus)	
	
	with open("event_data/{0}.json".format(event_name),"a") as f:
		json.dump(dictionary_temporary,f)

