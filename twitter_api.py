"""
Created on Fri May  4 01:43:42 2018

@author: Bogdan
"""
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_analysis as s

# API keys
consumer_key = 'Iy9vo9Sge6nehGL76WTb6FyNT'
consumer_secret = 'V9BzUfHzuKhAox6hrbs9tFGTu0Fgt14UsXLHVH3Q9ZfciLdn6D'
access_token = '1857791228-NGjHHpsi7gn3VRWyPG4IQPWSjDV4mticN3yFfGy'
access_secret = 'oehED95vMnTG9WGJTRCeQCyVZdvAWywtSNtpB3y6Y6Fe6'

class listener(StreamListener):

    def on_data(self, data):
        
        all_data = json.loads(data)
        
        tweet = all_data["text"]
        sentiment, confidence = s.analyse(tweet)
        print(tweet, sentiment, confidence)
        
        if confidence*100 >= 60:
            f = open('logs/twitters.txt', 'a')
            f.write(sentiment + '\n')
            f.close()
        
        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["good"])
