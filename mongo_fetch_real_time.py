#from __future__ import print_function
#import tweepy
import json
#from pymongo import MongoClient
import datetime


import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["Event"]

mycol = mydb["data"]

#cursor = mycol.find().sort([("created_at",-1)])
cursor = mycol.find()
#print cursor

#f=open()
c=0

with open("mongo_dataRealTime.json","a") as f:
	for ele in cursor:
		#print(ele)		
		del ele['_id']
				
		#print ele["text"]
		json.dump(ele,f)
		f.write("\n")
		c=c+1

print(c)


