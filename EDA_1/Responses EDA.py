import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

resp_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/responses.csv")
print(resp_Df.info())
print(resp_Df.head().T)

resp_Df.loc[:, resp_Df.notnull().all()].columns # Only 6 cols have no null

## Build Data Tables and Profiles
# 1. Histogram-Numerical Variables
resp_hist = resp_Df.hist(bins = 10, figsize = (20, 10))
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/respiratory Histogram.png")
plt.show()

# 2. Boxplot
plt.figure(figsize = (80, 10))

num_cols = resp_Df.select_dtypes(include = "int64").columns # adjust with value_counts*
num_cols = num_cols.append(resp_Df.select_dtypes(include = "float").columns)
for i,col in enumerate(num_cols):
    plt.subplot(15, 10, i+1)
    sns.boxplot(resp_Df[col], color = 'xkcd:lime')
    plt.title(f"{col}", fontsize = 10)
    plt.xlabel(' ')
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/respiratory Boxplot.png")
plt.show()

# 3. PairPlots
# resp_pp = sns.pairplot(resp_Df)
# plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/respiratory Pairplot.png")
# plt.show()

## Variable Measurement
# 1. Categorical Variables
print()
print('CAT Variables')
print()
n_cat = []
non_col = []
for col in resp_Df.columns:
    if resp_Df[col].dtypes == object and resp_Df[col].nunique() < 15:
        n_cat.append(col)
        print('=='*7)
        print(f"{resp_Df[col].value_counts()}")
    elif resp_Df[col].dtypes == object:
        non_col.append(col)
print(non_col)
print(f'no. of CAT Variables: {len(n_cat)}')

# 2. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in resp_Df.columns:
    if resp_Df[col].dtypes != object and resp_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{resp_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Continuous Variables')
n_cont = []
for col in resp_Df.columns:
    if resp_Df[col].dtypes != object and resp_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{resp_Df[col].min()} and Max_{resp_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(resp_Df.columns) - (len(n_cat) + len(n_num) + len(n_cont)))} columns")

## Exploring Data Relationships
resp_corr = resp_Df.corr()
print(resp_corr.abs())

mask = np.zeros_like(resp_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (20, 10))
sns.heatmap(resp_corr, mask = mask, center =0, vmin =-1.2, vmax = 1.2,
            linewidths= .1, cbar_kws = {'shrink': .5}, cmap = 'coolwarm',
            square =False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/responses Heatmap.png")
plt.show()

# This heatmap looks like swarm of bees, a better means has to be taken to clean it better.

## Conclusion
# Obvoiusly this is a very large dataset, and making a comprehensive report or conclusion is a long shot,
# I avioded this totally.

## Moving Forward
# This kind of data is tailor made for AI engineering, it is perculiarly a social science affair,
# which often yield unpredictable results but will suffice for some intense engineering.