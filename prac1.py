import pandas as pd

# read the pva97nk dataset
df = pd.read_csv('pva97nk.csv')

# show all columns information
print(df.info())