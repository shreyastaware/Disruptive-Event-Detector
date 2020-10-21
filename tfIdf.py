#this file is set of methods 1) tfidf 2) cosine similarity

#call get_result_cosine(a, b) to calculate cosine similarity, where a and b are set of tokens..

#cal tfIdf(list_of_tokens) to get tfIdf of each words in descending order

import gensim
from gensim import corpora, models, similarities
import pandas as pd
from ast import literal_eval
import re
import math
from collections import Counter



def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words)


def get_result_cosine(content_a, content_b):
    text1 = content_a
    text2 = content_b

    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)

    cosine_result = get_cosine(vector1, vector2)
    return cosine_result	
'''
tweet1="Hi i am saswata live in patna residing in boys hostel at IIT Patna where do u live in"
tweet2="tweet retweet Hi i am brijendra live in patna residing in boys hostel at IIT Madras @manish #riot afaf sffsfs afafaf afgafaff"
x=get_result_cosine(tweet1,tweet2)
print x
'''
#cosine_result=get_result_cosine("Hi i am am sandy", "Hi i am am saswata")
#print cosine_result
#d1={"hi":3.66,"saswata":4.55,"manish":4.55}
#d2={"hi":2.66,"shalini":0.55,"manish":3.55}
#l=get_cosine(d1,d2)
#print l
'''
dataset=[['kneeling', 'Dear', 'God', 'If', 'you', 'are', 'there', 'and', 'I', 'hope', 'you', 'are', 'Now', 'would', 'be', 'good', 'time', 'for', 'a', 'mutiny', 'of', 'leading', 'figures', 'i', 'u'], ['No', 'his', 'disdain', 'for', 'the', 'lies', 'the', 'fake', 'justification', 'for', 'the', 'lies', 'bul', 'u'], ['AND', 'FINALLY', 'BBC', 'And', 'Brexit', 'A', 'Loss', 'Of', 'Nerve', 'broadcaster', 'caves', 'to', 'Brexiteer', 'bullying', 'BUT', 'IS', 'UNABLE', 'TO', 'ADMIT', 'IT'], ['Note', 'to', 'Giuliani', 'Look', 'it', 'up', 'uColluding', 'to', 'violate', 'the', 'law', 'is', 'conspiracy', 'USC', 'Obstructing', 'Justice', 'also', 'is', 'crim', 'u'], ['I', 'added', 'a', 'video', 'to', 'a', 'playlist', 'Bullying', 'Senior', 'Students', 'are', 'in', 'for', 'a', 'Rude', 'Awakening'], ['if', 'youre', 'new', 'to', 'this', 'community', 'and', 'you', 'didnt', 'go', 'through', 'or', 'know', 'about', 'those', 'fucking', 'years', 'of', 'anxiety', 'panic', 'and', 'depres', 'u'], ['if', 'youre', 'new', 'to', 'this', 'community', 'and', 'you', 'didnt', 'go', 'through', 'or', 'know', 'about', 'those', 'fucking', 'years', 'of', 'anxiety', 'panic', 'and', 'depres', 'u'], ['Whattay', 'interview', 'Bullying', 'the', 'bully', 'The', 'long', 'lost', 'art', 'This', 'was', 'not', 'at', 'a', 'u'], ['tobi', 'is', 'bullying', 'me'], ['STOP', 'BULLYING', 'ME', 'YALL', 'DIDNT', 'SAY', 'SHIT', 'WHEN', 'PETE', 'POSTED', 'IT', 'BUT', 'NOW', 'ITS', 'MY', 'PFP', 'AND', 'EVERYONE', 'IS', 'BULLYING', 'ME', 'THIS', 'IS', 'HOM', 'u'], ['if', 'youre', 'new', 'to', 'this', 'community', 'and', 'you', 'didnt', 'go', 'through', 'or', 'know', 'about', 'those', 'fucking', 'years', 'of', 'anxiety', 'panic', 'and', 'depres', 'u'], ['Note', 'to', 'Giuliani', 'Look', 'it', 'up', 'uColluding', 'to', 'violate', 'the', 'law', 'is', 'conspiracy', 'USC', 'Obstructing', 'Justice', 'also', 'is', 'crim', 'u'], ['nikkis', 'bullying', 'me', 'whatd', 'i', 'ever', 'do', 'to', 'her'], ['And', 'who', 'is', 'bullying', 'you', 'Cuz', 'imma', 'go', 'full', 'momma', 'bear', 'on', 'them'], ['Since', 'the', 'New', 'Zealand', 'Football', 'press', 'release', 'just', 'talks', 'vaguely', 'about', 'ucculture', 'ud', 'reminder', 'ucallegations', 'around', 'bullying', 'intimi', 'u'], ['Silly', 'boys', 'Jack', 'amp', 'Mark', 'truly', 'fear', 'us', 'so', 'with', 'bullying', 'tactics', 'they', 'ure', 'trying', 'to', 'shut', 'us', 'up', 'Their', 'bans', 'are', 'akin', 'to', 'fascism', 'u'], ['Lionel', 'Messi', 'bullying', 'his', 'dog', 'udd', 'ude'], ['Note', 'to', 'Giuliani', 'Look', 'it', 'up', 'uColluding', 'to', 'violate', 'the', 'law', 'is', 'conspiracy', 'USC', 'Obstructing', 'Justice', 'also', 'is', 'crim', 'u'], ['For', 'the', 'Fandom', 'Bullying', 'each', 'other', 'for', 'the', 'way', 'they', 'write', 'Remember', 'that', 'they', 'arent', 'American', 'same', 'as', 'the', 'twins', 'u'], ['That', 'us', 'nice', 'amp', 'all', 'Mrs', 'Trump', 'but', 'we', 'the', 'people', 'would', 'like', 'to', 'see', 'major', 'progress', 'in', 'your', 'antibullying', 'c', 'u'], ['Unhinged', 'Top', 'Planned', 'Parenthood', 'Exec', 'BUSTED', 'for', 'Bullying', 'ProA', 'Parkland', 'Suvivor'], ['PHYSICAL', 'BULLYING'], ['Note', 'to', 'Giuliani', 'Look', 'it', 'up', 'uColluding', 'to', 'violate', 'the', 'law', 'is', 'conspiracy', 'USC', 'Obstructing', 'Justice', 'also', 'is', 'crim', 'u'], ['Note', 'to', 'Giuliani', 'Look', 'it', 'up', 'uColluding', 'to', 'violate', 'the', 'law', 'is', 'conspiracy', 'USC', 'Obstructing', 'Justice', 'also', 'is', 'crim', 'u'], ['Lionel', 'Messi', 'bullying', 'his', 'dog'], ['Jennie', 'Uglies', 'She', 'us', 'a', 'total', 'bitch', 'my', 'cousin', 'us', 'best', 'friend', 'who', 'has', 'this', 'friend', 'in', 'high', 'school', 'met', 'this', 'one', 'girl', 'and', 'she', 'u'], ['if', 'youre', 'new', 'to', 'this', 'community', 'and', 'you', 'didnt', 'go', 'through', 'or', 'know', 'about', 'those', 'fucking', 'years', 'of', 'anxiety', 'panic', 'and', 'depres', 'u'], ['This', 'is', 'Perez', 'Hilton', 'writing', 'gossip', 'columns', 'and', 'bullying', 'celebrity', 'women', 'for', 'almost', 'a', 'decade', 'with', 'his', 'unprofessional', 'bootleg', 'l', 'u'], ['Hahahahahahahaha', 'How', 'The', 'Fuck', 'Is', 'Cyber', 'Bullying', 'Real', 'Hahahaha', 'Nigga', 'Just', 'Walk', 'Away', 'From', 'The', 'Screen', 'Like', 'Nigga', 'Close', 'Yo', 'u'], ['ucthis', 'is', 'bullying', 'udd', 'ude', 'ud', 'ucyou', 'ure', 'so', 'mean', 'udd', 'ude', 'ud', 'me'], ['i', 'just', 'miss', 'u', 'bullying', 'me', 'udd', 'udea'], ['Lionel', 'Messi', 'bullying', 'his', 'dog', 'udd', 'ude'], ['We', 'will', 'not', 'stand', 'for', 'evil', 'in', 'this', 'world', 'Our', 'babies', 'children', 'teens', 'and', 'young', 'adults', 'deserve', 'to', 'live', 'with', 'out', 'fea', 'u'], ['July', 'at', 'about', 'o', 'uclock', 'in', 'the', 'morning', 'SPO', 'Gerry', 'H', 'Ramos', 'Traffic', 'PNCO', 'of', 'Pagbilao', 'MPS', 'under', 'the', 'supe', 'u'], ['if', 'youre', 'new', 'to', 'this', 'community', 'and', 'you', 'didnt', 'go', 'through', 'or', 'know', 'about', 'those', 'fucking', 'years', 'of', 'anxiety', 'panic', 'and', 'depres', 'u'], ['Lionel', 'Messi', 'bullying', 'his', 'dog', 'udd', 'ude'], ['Note', 'to', 'Giuliani', 'Look', 'it', 'up', 'uColluding', 'to', 'violate', 'the', 'law', 'is', 'conspiracy', 'USC', 'Obstructing', 'Justice', 'also', 'is', 'crim', 'u'], ['Jennie', 'Uglies', 'She', 'us', 'a', 'total', 'bitch', 'my', 'cousin', 'us', 'best', 'friend', 'who', 'has', 'this', 'friend', 'in', 'high', 'school', 'met', 'this', 'one', 'girl', 'and', 'she', 'u'], ['twitter', 'should', 'verify', 'me', 'for', 'my', 'brave', 'efforts', 'in', 'bullying', 'anime', 'nazis'], ['if', 'youre', 'new', 'to', 'this', 'community', 'and', 'you', 'didnt', 'go', 'through', 'or', 'know', 'about', 'those', 'fucking', 'years', 'of', 'anxiety', 'panic', 'and', 'depres', 'u'], ['It', 'us', 'BULLYING', 'It', 'us', 'targeting', 'people', 'who', 'have', 'SEIZURES', 'and', 'will', 'live', 'another', 'YEARS', 'just', 'leave', 'them', 'the', 'fuck', 'alon', 'u'], ['Why', 'y', 'uall', 'be', 'bullying', 'queen', 'it', 'us', 'not', 'her', 'fault', 'she', 'look', 'like', 'a', 'ninja', 'turtle'], ['hat', 'us', 'bullying'], ['Hindus', 'were', 'often', 'violently', 'suppressed', 'during', 'periods', 'of', 'foreign', 'rule', 'Today', 'intellectual', 'and', 'media', 'bullying', 'and', 'defam', 'u'], ['That', 'us', 'nice', 'amp', 'all', 'Mrs', 'Trump', 'but', 'we', 'the', 'people', 'would', 'like', 'to', 'see', 'major', 'progress', 'in', 'your', 'antibullying', 'c', 'u']]


def tfIdf(dataset):##list of tokens
	print dataset[:5]
	print len(dataset)
	dictionary = corpora.Dictionary(line for line in dataset)
	print dictionary;input("enter a number")
	corpus=[dictionary.doc2bow(doc) for doc in dataset]
	print corpus;print len(corpus);input("enter a number")
	tfidf=models.TfidfModel(corpus)
	print tfidf;input("enter a number")
	corpus=tfidf[corpus]
	print corpus;input("enter a number")
	corpus=list(corpus)
	print corpus
	print len(corpus); input("enter a number")
	topWordsL=[]
	topWordsL=[ tup for lis in corpus for tup in lis  ]
	print topWordsL;input("enter a no")
	topWordsL=sorted(topWordsL, key=lambda x:float(x[1]), reverse=True)
	print topWordsL;input("enter a no")
	topWordsL=[dictionary[tup[0]] for tup in topWordsL]
	print topWordsL;input("enter a no")
	print len(topWordsL);print type(topWordsL)
	t=topWordsL[:10]	
	print t;print type(t)
	with open("topWordsL.txt","a") as f:
		for x in t:
			f.write('%s\n'%x)
	return topWordsL
topWordsL=tfIdf(dataset)
print topWordsL

def tfIdfList(dataset):##list of tokens
	dictionary = corpora.Dictionary(line for line in dataset)
	corpus=[dictionary.doc2bow(doc) for doc in dataset]
	tfidf=models.TfidfModel(corpus)
	corpus=tfidf[corpus]
	corpus=list(corpus)
	for i,tokenL in enumerate(corpus):
		corpus[i]=[ x[1] for x in corpus[i]]
	#print corpus[:5]
	#print len(corpus); input("enter a number")
	return corpus


'''
