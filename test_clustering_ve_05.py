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
from wordcloud import WordCloud

#i=0
def makeImage(text,event_name):
    global i    
    wc = WordCloud(background_color="black")
    print "*********************", text, i
    # generate word cloud
    cloud=wc.generate_from_frequencies(text)
    cloud.to_file('output/VE_5/{0}.png'.format(event_name))
    #i=i+1

def post_processing(df):
	#<--remove null--->
	df = df[pd.notnull(df['full_text'])]

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
	df = df[pd.notnull(df['full_text'])]
	
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
	
	
dictionary_event_details={}
t=0
def make_dictionary(hashtag,df_clus):
	dictionary_temporary={}
	list_events_loc=[]
	event_id=time.time()
	
	#event_name_list_tuple=Counter(hashtag).most_common(3)
	#event_name=event_name_list_tuple[0][0]+" "+event_name_list_tuple[1][0]+" "+event_name_list_tuple[2][0]	
	x=Counter(hashtag)
	z=dict(x)
	y=x.most_common(3)
	#print y
	if len(y)==3:
		event_name=y[0][0].encode('utf-8')+" "+y[1][0].encode('utf-8')+" "+y[2][0].encode('utf-8')
	elif len(y)==2:
		event_name=y[0][0].encode('utf-8')+" "+y[1][0].encode('utf-8')
	else:
		event_name=y[0][0].encode('utf-8')

	#print event_name;input()

	makeImage(z,event_name)
	'''

	#print "This time df_clus is", df_clus.shape
	global t;t=t+1
	
	with open("location_dictionary.json", 'r') as file:
		di = json.load(file)
	
	for k in hashtag:
		for key in di:
			if key == k.lower():
				list_events_loc.append(key)
	
	if len(list_events_loc)!=0:
		
		
		location=Counter(list_events_loc).most_common(1)
		
		real_location=location[0][0].encode('utf-8')
		
		
		
		time1=[]
		for j,row in df_clus.iterrows():
			if real_location in row["full_text"]:
				datetime_object = datetime.strptime(row["created_at"], '%a %b %d %H:%M:%S +0000 %Y')
				x = time.mktime(datetime_object.timetuple())
				time1.append(x)
					
		
		if len(time1)!=0:
			
		
			sorted_time=sorted(time1)	
		

			y=datetime.fromtimestamp(sorted_time[0]).strftime("%A, %B %d, %Y %I:%M:%S")
		
			dictionary_temporary[event_id]={"latitude":di[real_location][0],"longitude":di[real_location][1],"ename":event_name,"created_at":y,"location":real_location,"text":"event{0}".format(t),"main_event":1}
			
			
			
	
		

	for i,row in df_clus.iterrows():
			flag=0
			geo=row["geo"];location=row["user"]["location"].encode('utf-8');
                        
			if location:
				for key in di:
                        		if key in location:
                                		lat_long=di[key]
						flag=1
				if flag==1:		
					dictionary_temporary[row["id_str"]]={"id_str":row["id_str"],"latitude":lat_long[0],"longitude":lat_long[1],"ename":event_name,"created_at":row["created_at"],"location":location,"text":row["full_text"],"main_event":0,"user_name":row["user"]["name"],"follower_count":row["user"]["followers_count"],"retweet_count":row["retweet_count"]}
				

	print  dictionary_temporary
	return dictionary_temporary,event_name

	
	'''



#convert csv to DataFrame
df=js.make_csv("classified_data/violentExtremism.json")
#df=df[:1000]
print df.shape

print df.columns
#do preprocessing
#df=pre.xpreprocessing(df)
#df=post_processing(df)
list_pre=[]
for i,row in df.iterrows():
	text=removepunc(row["full_text"])
	text= text.lstrip("RT")
	text=remove_stopwords(text)
	text=remove_numbers(text)
	list_pre.append(p.clean(text))

df["preprocessed_text"]=list_pre


tweets=list(df["preprocessed_text"])
'''
ids=list(df["id_str"])


#print df["preprocessed_text"]

tweets1=list(df["full_text"])



tweets=[str(x).lower() for x in tweets]

tweets=[p.clean(x) for x in tweets]

ids=[str(x) for x in ids]
'''

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tweets)

#vectorizer.get_feature_names()

#print(X.toarray())     
#print "Before Clustering##########"
#print time.time()

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)
print(tfidf.shape )                        

from sklearn.cluster import KMeans

num_clusters = 05 #Change it according to your data.
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf)
clusters = km.labels_.tolist()

#idea={'Id':ids,'Text':tweets1,'Preprocessed':tweets, 'Cluster':clusters} #Creating dict having doc with the corresponding cluster number.
#frame=pd.DataFrame(idea,index=[clusters], columns=['Idea','Cluster']) # Converting it into a dataframe.

#frame=pd.DataFrame(idea, columns=['Id','Text','Preprocessed','Cluster'])
df["Cluster"]=clusters

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
	

	tweets=list(df_clus["full_text"])
		
	hashtag=[]
	for j,row in df_clus.iterrows():
		lis=row["entities"]["hashtags"]
		for e in lis:	
			hashtag.append(e["text"])
	make_dictionary(hashtag,df_clus)	
	#dictionary_temporary,event_name=make_dictionary(hashtag,df_clus)
	#print type(dictionary_temporary)
	#input()
	
	#if len(dictionary_temporary)!=0:
	#	for key,val in dictionary_temporary.iteritems():
	#		dictionary_event_details[key]=val
			
	#else:		
	#	continue
	#with open("event_data/ve/{0}.json".format(event_name),"a") as f:
	#	json.dump(dictionary_temporary,f)

