import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

nba_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/nba_2013.csv")


## Intiating Data Cleaning and Checks
nba_Df = nba_Df.dropna() # 403 rows

nba_Df.drop(columns = ['season','season_end'], inplace = True) # Monotonous
print(nba_Df.info())
print(nba_Df.head().T)

## Build Data Tables and Profiles
# 1. Histogram-Numerical Variables
nba_hist = nba_Df.hist(bins = 10, figsize = (20, 10))
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/nba Histogram.png")
plt.show()
#

# 2. Boxplot
plt.figure(figsize = (40, 10))

num_cols = nba_Df.select_dtypes(include = "int64").columns
num_cols = num_cols.append(nba_Df.select_dtypes(include = 'float').columns)
for i,col in enumerate(num_cols):
    plt.subplot(10, 4, i+1)
    sns.boxplot(nba_Df[col], color = ('xkcd:lime'))
    plt.title(f"{col}", fontsize = 10)
    plt.xlabel(' ')
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/nba Boxplot.png")
plt.show()

# 3. PairPlots
nba_pp = sns.pairplot(nba_Df)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/nba Pairplot.png")
plt.show()

## Variable Measurement
# 1. Categorical Variables
print()
print('CAT Variables')
print()
n_cat = []
non_col = []
for col in nba_Df.columns:
    if nba_Df[col].dtypes == object and nba_Df[col].nunique() < 15:
        n_cat.append(col)
        print('=='*7)
        print(f"{nba_Df[col].value_counts()}")
    elif nba_Df[col].dtypes == object: # Usually objects skip set parameters
        non_col.append(col)
print(non_col)
print(f'no. of CAT Variables: {len(n_cat)}')

# 2. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in nba_Df.columns:
    if nba_Df[col].dtypes != object and nba_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{nba_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Continuous Variables')
n_cont = []
for col in nba_Df.columns:
    if nba_Df[col].dtypes != object and nba_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{nba_Df[col].min()} and Max_{nba_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(nba_Df.columns) - (len(n_cat) + len(n_num) + len(n_cont)))} columns")

# Exploring Data Relationships
nba_corr = nba_Df.corr()
print(nba_corr.abs())

mask = np.zeros_like(nba_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (20, 10))
sns.heatmap(nba_corr, mask = mask, center =0, vmin =-1.2, vmax = 1.2,
            linewidths= .1, cbar_kws = {'shrink': .5}, cmap = 'coolwarm',
            square =False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/nba Heatmap.png")
plt.show()

## Conclusion
# Because of my lack of understanding of Basketball, I'll refrain from interpreting this result
# However the heatmap shows a lot of positive correlation btw variables,
# which indicates this a data set on a subset of basket ball players and it seems age does not matter.

## Moving Forward
# Definately get more understanding of the data set, clean accordingly
# Apply Indepth and Appropriate methods