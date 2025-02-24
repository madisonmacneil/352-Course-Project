import pandas as pd 

df = pd.read_csv('db.csv')
print(df.head(5))

#df has 24920 observations 
print(f"Size before removing duplicates: {df.size}")
df = df.drop_duplicates()
print(f"Size after removing duplicates: {df.size}")

#df after removing duplicates has 22920 observations 


