import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

djia_Df= pd.read_csv("C:/Users/HP/Desktop/EDAs/DJIA_table.csv")
print(djia_Df.info())
print(djia_Df.head()) # No NaNs

djia_Df['Date'] = pd.to_datetime(djia_Df['Date'])

## Build Data Profiles and Table
# 1. Histogram
hist = djia_Df.hist(bins =10, figsize= (20, 10) )
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/djia histogram.png")
plt.show()
# Adj_Close, mirrors Close, Open, High, Low
# This will reflect in Pairplots and we should be seeing high correlations in the heatmap

# 2. Boxplots
plt.figure(figsize=(20, 10))

num_cols = djia_Df.select_dtypes(include = 'float')
num_cols = num_cols.append(djia_Df.select_dtypes(include = 'int'))
for i, col in enumerate(num_cols):
    plt.subplot(5, 2, i+1)
    sns.boxplot(djia_Df[col], color = "xkcd:lime")
    plt.title(f"{col}", fontsize = 10)
    plt.xlabel(' ')
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/djia Boxplot.png")
plt.show()
# Every col seem to be skewed to the left towards 0.

# 3. Pairplots
djia_pp = sns.pairplot(djia_Df)
plt.savefig("C:/Users/HP/PycharmPRojects/Complete_EDA/djia heatmap.png")
plt.show()

# From this pairplot, I've a feeling of positive relationship between playing, pausing and seeking video.

## Variable Measurement

# 1. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in djia_Df.columns:
    if djia_Df[col].dtypes != object and djia_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{djia_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Cont and Num Variables')
n_cont = []
for col in djia_Df.columns:
    if djia_Df[col].dtypes != object and djia_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{djia_Df[col].min()} and Max_{djia_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(djia_Df.columns) - (len(n_num) + len(n_cont)))} columns")

## Explore Data Relationships
djia_corr = djia_Df.corr()
print(djia_corr.abs())
plt.figure(figsize = (20, 10))

mask = np.zeros_like(djia_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(djia_corr, mask = mask, vmin=-1.2, vmax=1.2, center = 0,
         linewidth = .1, cbar_kws = {"shrink":.5}, cmap="coolwarm",
         square = False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/djia Heatmap.png")
plt.show()

## Conclusion
# There is definitely wrong with this data set if were to be social phenomenon,
# however this could an early analysis of a mechanic object, because every other variable but
# 'volume' correlates +ve perfectly, while volume has -ve .69 with every variable, 'Date' was excluded.

# Moving Forward
# Accentuate other kernels, is the data perfect or is need for more cleaning.
# If data is clean I don't see the need to further analysis a data with such perfect relationship.