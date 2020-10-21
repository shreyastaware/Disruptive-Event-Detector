

import tweepy

from tweepy.api import API

import json

import datetime

import time

 

 

API_KEY = ''

API_SECRET = ''

ACCESS_TOKEN = ''

ACCESS_TOKEN_SECRET = ''

 

key = tweepy.OAuthHandler(API_KEY, API_SECRET)

key.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

 

 

 

status_count=0

file_count=0

class Stream2Screen(tweepy.StreamListener):


    def on_status(self, status):


        global status_count,file_count


        status_count = status_count + 1


        ts=time.time()


        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')


        if status_count%160000==0:


            file_count=file_count+1

 


        with open('Terrorism/Terrorism_%s_File_%s.json'%(st,file_count), 'a') as file:


            json.dump(status._json, file)


            file.write('\n')

 

paths = 'terrorism.txt'

 

while 1:

    try:

        stream = tweepy.streaming.Stream(key, Stream2Screen())

        stream.filter(track=open(paths, 'r'), languages=['en'])
    
    except:
        continue

 

 

