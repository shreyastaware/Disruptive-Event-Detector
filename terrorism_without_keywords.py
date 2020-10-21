import tweepy
from tweepy.api import API
import json
import datetime
import time


API_KEY = 'ACAPnMYope5mrgs75bCGNMb8f'
API_SECRET = 'WdgltYOrMPJmpslouXmhm2LnI2mD8BAmSuH8AAMCp2P7mRSaeU'
ACCESS_TOKEN = '3219124868-XQF5tkmCUGa2oxRf8DtvCDZTMC2rhTZQnU5oSIg'
ACCESS_TOKEN_SECRET = 'GDxyys0X9rMzXGUjC2hFhHu6BHpptrrSo42u4BaA3SJMe'

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



while 1:
	stream = tweepy.streaming.Stream(key, Stream2Screen())
	stream.sample()
        


