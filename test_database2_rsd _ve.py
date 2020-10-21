import nltk
# nltk.download("stopwords")
# nltk.download("words")
import pymongo
import json
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import pandas as pd
import preprocessing as pre
import time
from mongoDBUtils import mongoDBUtils 
#from porter2stemmer import Porter2Stemmer
from nltk.stem import PorterStemmer
import re
from collections import Counter
import calculate_feature as cf
import clustering as clus
import real_time_wordcloud as rt
import re
import temTFIDF as tem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import os
import glob
import json
#import datetime
import json_csv as json_csv

import string
from datetime import datetime

def make_dictionary(hashtag,df_clus):
	
	list_events_loc=[]
	event_id=time.time()
	
	with open("location_dictionary.json", 'r') as file:
		di = json.load(file)
	
	for k in hashtag:
		for key in di:
			if key == k.lower():
				list_events_loc.append(key)
	
	#print list_events_loc		
	#z=dict(Counter(list_events_loc))
	#print z	;input()
	if len(list_events_loc)!=0:
		location=Counter(list_events_loc).most_common(1)
		
		real_location=location[0][0].encode('utf-8')
		print real_location
		print type(real_location)
	
	
		print list_events_loc

		time1,list_ids,list_text,list_lat,list_long=[],[],[],[],[]
		for j,row in df_clus.iterrows():
			if real_location in row["full_text"]:
				datetime_object = datetime.strptime(row["created_at"], '%a %b %d %H:%M:%S +0000 %Y')
				x = time.mktime(datetime_object.timetuple())
				time1.append(x)
			flag=0	
			geo=row["geo"];location=row["user"]["location"].encode('utf-8');
                
			if geo:
                    		list_text.append(row["full_text"])
				list_ids.append(row["id"])
				list_lat.append(row["geo"]["coordinates"][0])
				list_long.append(row["geo"]["coordinates"][1])
			elif location:
				for key in di:
                        		if key in location:
                                		flag=1;lat_long=di[key]
				
				if flag==1:
					list_text.append(row["full_text"])
                      			list_ids.append(row["id"])
					list_lat.append(lat_long[0])
                                	list_long.append(lat_long[1])

		#print time1
		if len(time1)==0:
			return event_id,[0.0,0.0],time.time(),"Null","Null"
		else:
			sorted_time=sorted(time1)	
		#print sorted_time

			y=datetime.fromtimestamp(sorted_time[0]).strftime("%A, %B %d, %Y %I:%M:%S")
		#print y
		
		
	

			event_name_list_tuple=Counter(hashtag).most_common(3)
			event_name=event_name_list_tuple[0][0]+" "+event_name_list_tuple[1][0]+" "+event_name_list_tuple[2][0]
			return event_id,di[real_location],y,real_location,event_name,list_ids,list_text,list_lat,list_long 
			
	else: 
		p,q=[],[]
		return event_id,[0.0,0.0],time.time(),"Null","Null",p,q 

dictionary_event_details={}	
dictionary_event_tweets={}



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


def compute_features(df):
	
	df[['geo']] = df[['geo']].fillna('')
	#df_sub=df[df["id"].isin(list_of_id)]
	list_of_features=[]
	allPre=[]
	[allPre.extend(x.split()) for x in list(df["tweet_rm_3"])]
	for i,row in df.iterrows():
		x=[]
		x.append((row["id"]))
		df_sub=df[df["id"].isin(x)]
		#features=cf.compute_feature(df,df_sub)
		features=cf.compute_feature(df,row,allPre)
		#print features
		#print type(features)
		list_of_features.append(features)
	df["tweet_FeatureVector"]=list_of_features
	#print df.head()	
	#df.to_csv("feature.csv",index=False)
	return df

def Convert(tup, di): 
    for a, b in tup: 
        di.setdefault(b, []).append(a) 
    return di 
      


def make_lsa(tweets):
	print "i am under lsa"
	vectorizer = TfidfVectorizer()
	X =vectorizer.fit_transform(tweets)
	#print X
	lsa = TruncatedSVD(n_components=2,n_iter=100)

	lsa.fit(X)
	terms = vectorizer.get_feature_names()
	#print terms
	dict_of_clusters={}
	for i,comp in enumerate(lsa.components_):
		termsInComp = zip(terms,comp)
		sortedterms = sorted(termsInComp, key=lambda x: x[1],reverse=True)[:10]
		#print("Concept %d:" % i)
		#for term in sortedterms:
		#	print term
	 	#print " "
	    	dictionary = {} 
		di=Convert(sortedterms, dictionary)
		dict_of_clusters[i]=di
	rt.start_cloud(dict_of_clusters)
'''
def make_hashtag(tweets):
	hashtag=[]
	for l in tweets:
		hashtag.extend(re.findall(r"#(\w+)", l))
	
	#hashtag=Counter(hashtag); hashtag=dict(hashtag)
	#print type(hashtag), hashtag
	return hashtag

'''
def make_dictionary(hashtag,df_clus):
	
	list_events_loc=[]
	event_id=time.time()
	
	with open("location_dictionary.json", 'r') as file:
		di = json.load(file)
	
	for k in hashtag:
		for key in di:
			if key == k.lower():
				list_events_loc.append(key)
	
	#print list_events_loc		
	#z=dict(Counter(list_events_loc))
	#print z	;input()
	if len(list_events_loc)!=0:
		location=Counter(list_events_loc).most_common(1)
		
		real_location=location[0][0].encode('utf-8')
		print real_location
		print type(real_location)
	
	
		print list_events_loc

		time1,list_ids,list_text,list_lat,list_long=[],[],[],[],[]
		for j,row in df_clus.iterrows():
			if real_location in row["full_text"]:
				datetime_object = datetime.strptime(row["created_at"], '%a %b %d %H:%M:%S +0000 %Y')
				x = time.mktime(datetime_object.timetuple())
				time1.append(x)
			flag=0	
			geo=row["geo"];location=row["user"]["location"].encode('utf-8');
                
			if geo:
                    		list_text.append(row["full_text"])
				list_ids.append(row["id"])
				list_lat.append(row["geo"]["coordinates"][0])
				list_long.append(row["geo"]["coordinates"][1])
			elif location:
				for key in di:
                        		if key in location:
                                		flag=1;lat_long=di[key]
				
				if flag==1:
					list_text.append(row["full_text"])
                      			list_ids.append(row["id"])
					list_lat.append(lat_long[0])
                                	list_long.append(lat_long[1])

		#print time1
		if len(time1)==0:
			return event_id,[0.0,0.0],time.time(),"Null","Null"
		else:
			sorted_time=sorted(time1)	
		#print sorted_time

			y=datetime.fromtimestamp(sorted_time[0]).strftime("%A, %B %d, %Y %I:%M:%S")
		#print y
		
		
	

			event_name_list_tuple=Counter(hashtag).most_common(3)
			event_name=event_name_list_tuple[0][0]+" "+event_name_list_tuple[1][0]+" "+event_name_list_tuple[2][0]
			return event_id,di[real_location],y,real_location,event_name,list_ids,list_text,list_lat,list_long 
			
	else: 
		p,q=[],[]
		return event_id,[0.0,0.0],time.time(),"Null","Null",p,q 

dictionary_event_details={}	
dictionary_event_tweets={}

#i=0
def make_dataframe(file,class_name):
	'''	
	tids1, tids2, nonRelevantTweetIds = getRelevantTweetIds("radical")
	global i
	tids = []
	for id in tids1:
		tids.append(id)
	for id in tids2:
		tids.append(id)
	tweetData = getRelevantTweetData(tids)


	di={}
	lid = []
	lgeo = []
	lcreated_at = []
	lentities = []
	ltext = []
	lextendedEntities = []
	for x in tweetData:
		lid.append(x["tweet_id"])
		lgeo.append(x["geo"])
		lcreated_at.append(x["created_at"])
		lentities.append(x["entities"])
		lextendedEntities.append(x["extended_entities"])
		ltext.append(x["tweet_text"])


	di["id"]=lid
	di["created_at"]=lcreated_at
	di["text"]=ltext
	di["geo"]=lgeo
	di["entities"]=lentities
	di['extended_entities']=lextendedEntities

	df=pd.DataFrame(di)
	'''
	df=json_csv.make_json_csv(file)
	
	#global i
	#i=i+1
	#global dictionary_event

	#df=pd.read_csv("experiment.csv",delimiter=",")
	df=df[:500]
	#print "hello..shape dekh ..."
	#print df.shape;
	#df.to_csv("Violence.csv",index=False)

	#print "{0}th time****".format(i)
	
	#print df.head()
	#print df.shape
	
	
	#<---preprocessing---->
	df=pre.xpreprocessing(df)
	df=post_processing(df)
	df=compute_features(df)	
	#print list(df["tweet_FeatureVector"])[:10];input()
	print "WAIT######################################";	
	#input()
	print "start clus"
	clustersD,length_of_clusters=clus.make_clusters(df)
	#print "WAIT######################################";input("enter a no")
	#print clustersD
	print "length of clusters==   ",length_of_clusters
	dict_of_clusters={}
	
	print "***********Report of Clustering**********"	
	
	for i in range(0,length_of_clusters):
		df_clus=df[df["id"].isin(clustersD[i]["twId"])]
		tweets=list(df_clus["full_text"])
		#make_lsa(tweets)	
		hashtag=[]
		for j,row in df_clus.iterrows():
			lis=row["entities"]["hashtags"]
			for e in lis:	
				hashtag.append(e["text"])
		
		event_id,lat_long,y,location_name,event_name,list_ids,list_text,list_lat,list_long=make_dictionary(hashtag,df_clus)
		print len(list_ids),len(list_text),len(list_lat),len(list_long)
		
		input()
		
		dictionary_event[event_id]=[lat_long[0],lat_long[1],class_name,event_name,[event_id,y],y,location_name]	
		#dictionary_event_details[]
		print "*******Next cluster*****"
		#dict_of_clusters.update({i:hashtag})

	
	
	#<--Temporal TFIDF---->
	#dict_of_clusters=tem.make_temporal_TFIDF(clustersD,df)
			
	
	#<--geneerate word cloud>	
	#rt.start_cloud(dict_of_clusters,i)	
	
	
	'''	
	updateTrainedIds1(tids1)
	updateTrainedIds2(tids2)
	updateTrainedIds2(nonRelevantTweetIds)
	'''

	
print("Starting")
#while True:


'''
make_dataframe("classified_data/nonRadicalViolence.json","nrv")
print "Done with nrv"
print dictionary_event


make_dataframe("classified_data/radicalViolence.json","rv")
print "Done with rv"
print dictionary_event
'''

make_dataframe("classified_data/violentExtremism.json","ve")
print "Done with ve"
print dictionary_event

#make_dataframe("classified_data/nonViolentExtremism.json","nve")
#print "Done with nve"
#print dictionary_event

	


print "*************We got Final dictionary***************","\n\n\n\n"
print dictionary_event
f=open("event_data/event_tweets.json","a")
json.dump(dictionary_event,f)
f.close()

