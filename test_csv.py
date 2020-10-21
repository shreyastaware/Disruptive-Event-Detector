import pandas as pd
df=pd.read_csv("test.csv",delimiter=",")
print(df.shape)
#print(df)
#print(df.head())
#print(df.tail())
df["color"]=['g','b','v','r','y','w','o']
df.to_csv("new.csv",index=False)
