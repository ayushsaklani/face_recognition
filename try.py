import pandas as pd 
import numpy as np 

database = pd.read_csv('database.csv',index_col = 'Name')

# print(database.index.name)
for i in database.index:
	print(np.array(database.loc[str(i)]))