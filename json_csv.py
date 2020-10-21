import pandas as pd
#from pandas import DataFrame 
import json
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def make_csv(file_name):
	uniqueKeys=[]
	f=open(file_name, "r")
	for l in f:
        	d=json.loads(l)
        	uniqueKeys.extend(d.keys())
        	uniqueKeys=list(set(uniqueKeys))
	f.close()


	#print "uniqueKeys", len(uniqueKeys)
	#input("enter a number")
	#print "before sorted: ",uniqueKeys
	uniqueKeys=sorted(uniqueKeys);
	#print "\n\nafter sorted: ",uniqueKeys;


	f=open(file_name, "r")
	mainL=[];jj=0
	for l in f:
        	d=json.loads(l)
        	values=[]
        	for i, k in enumerate(uniqueKeys):
                	try:
                        	values.append(d[k])
                	except:
                        	values.append(None)
       		mainL.append(values)
        #if jj>100:
        #       break
        #jj+=1
	#f.close()
	print "list formed"
	df=pd.DataFrame(mainL, columns=uniqueKeys)
	return df

