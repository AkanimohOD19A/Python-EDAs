import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

hmeq_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/hmeq.csv")
print(hmeq_Df.info())
print(hmeq_Df.head().T)
# There a quite a number of NaNs here, but the data looks clean, i.e NaNs mights be valid.
## Initiating Data Checks and Cleaning
hmeq_Df['BAD'].replace({0:"No", 1:"Yes"}, inplace = True)

## Building Data Tables and Profiles
# 1. Histogram
hist = hmeq_Df.hist(bins =10, figsize= (20, 10) )
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/hmeq histogram.png")
plt.show()
# A lot of loans are been given at around the 2000 mark,
# the distribution resembles what we get at 'mortagedue' and 'clno'

# 2. Boxplots
plt.figure(figsize=(20, 10))

num_cols = hmeq_Df.select_dtypes(include = 'int64')
num_cols = num_cols.append(hmeq_Df.select_dtypes(include = 'float'))
for i, col in enumerate(num_cols):
    plt.subplot(5, 2, i+1)
    sns.boxplot(hmeq_Df[col], color = "xkcd:lime")
    plt.title(f"{col}", fontsize = 10)
    plt.xlabel(' ')
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/hmeq Boxplot.png")
plt.show()

# 3. Pairplots
# hmeq_pp = sns.pairplot(hmeq_Df)
# plt.savefig("C:/Users/HP/PycharmPRojects/Complete_EDA/hmeq heatmap.png")
# plt.show()

# MortageDue and Loan

## Variable Measurement
# 1. Categorical Variables
print()
print('CAT Variables')
print()
n_cat = []
non_col = []
for col in hmeq_Df.columns:
    if hmeq_Df[col].dtypes == object and hmeq_Df[col].nunique() < 15:
        n_cat.append(col)
        print('=='*7)
        print(f"{hmeq_Df[col].value_counts()}")
    elif hmeq_Df[col].dtypes == object: 
        non_col.append(col)
print(non_col)
print(f'no. of CAT Variables: {len(n_cat)}')

# 2. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in hmeq_Df.columns:
    if hmeq_Df[col].dtypes != object and hmeq_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{hmeq_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Cont and Num Variables')
n_cont = []
for col in hmeq_Df.columns:
    if hmeq_Df[col].dtypes != object and hmeq_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{hmeq_Df[col].min()} and Max_{hmeq_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(hmeq_Df.columns) - (len(n_num) + len(n_cont)))} columns")

# the least amt of loan = 8000 and <= 900,000
 
## Explore Data Relationship
# Correlation and HeatMap
hmeq_corr = hmeq_Df.corr()
print(hmeq_corr.abs())
plt.figure(figsize = (20, 10))

mask = np.zeros_like(hmeq_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(hmeq_corr, mask = mask, vmin=-1.2, vmax=1.2, center = 0,
         linewidth = .1, cbar_kws = {"shrink":.5}, cmap="coolwarm",
         square = False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/hmeq Heatmap.png")
plt.show()

# MORTDUE on Value is very positively correlated, and on CLNO, DEBTINC, CLAGE, otherwise it comprises
# of a large pool on intervening. Loan kind of shares the same properties.

## Moving Forward
# We need to create considerable model for Loan,
# then start even with deeper analysis and then predictive metrics

