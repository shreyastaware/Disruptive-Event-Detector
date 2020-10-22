# Disruptive-Event-Detector
Steps involved were -
1. Collected tweets using Twitter API(tweepy) and stored them in a mongoDB database.
2. Fetched tweets of last two hour from mongoDB database and applied K-means clustering algorithm over them continuously.
3. Fetched only those clusters having large number of words and run TF-IDF technique over these selected clusters 
4. Applied word cloud technique on the result of TF-IDF to get the most discussed events.
