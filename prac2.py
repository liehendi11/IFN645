import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# read the pva97nk dataset
df = pd.read_csv('pva97nk.csv')

# show all columns information
print(df.info())

###################################################
# Exploration stage using 'DemAge' column for example, comment/disable for the end product

# getting overall knowledge of the column
print(df['DemAge'].describe())

# checking unique values of the column. Notice the NaN (missing value) in there.
print(df['DemAge'].unique())

# how many records for each value
print(df['DemAge'].value_counts())

# checking the average age of lapsing donor vs non-lapsing donor (TargetB)
print(df.groupby(['TargetB'])['DemAge'].mean())

# plot distribution of age using seaborn distplot
# (dropna is used because 'DemAge' has missing values)
dg = sns.distplot(df['DemAge'].dropna())
plt.show()

# plot distribution of dem gender. Because dem gender is a categorical variable, we can't use distplot. We will use countplot instead
dg = sns.countplot(data=df, x='DemGender')
plt.show()

###################################################

###################################################
# Setting correct levels and reject unnecessary columns

# drop ID and the unused target variable
df.drop(['ID', 'TargetD'], axis=1, inplace=True)

# impute missing values in DemAge with its mean
df['DemAge'].fillna(df['DemAge'].mean(), inplace=True)

# change DemCluster from interval/integer to nominal/str
df['DemCluster'] = df['DemCluster'].astype(str)

# change DemHomeOwner into binary 0/1 variable
dem_home_owner_map = {'U':0, 'H': 1}
df['DemHomeOwner'] = df['DemHomeOwner'].map(dem_home_owner_map)

# denote miss values in DemMidIncome
temp = df['DemMedIncome']
temp[temp < 1] = 0
df['DemMedIncome'] = temp

df['DemMedIncome'].replace(0, np.nan, inplace=True)

# impute med income using average strategy
df['DemMedIncome'].fillna(df['DemMedIncome'].mean(), inplace=True)

# impute gift avg card 36 using average strategy
df['GiftAvgCard36'].fillna(df['GiftAvgCard36'].mean(), inplace=True)

###################################################