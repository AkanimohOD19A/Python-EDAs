import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

wine_Df = pd.read_csv("C:/Users/HP/Documents/EDA/Python/red-white-wine-dataset/wine_dataset.csv")
print(wine_Df.info())
print(wine_Df.head())

## Building Data Profiles and Tables
# 1. Histogram
hist = wine_Df.hist(bins = 10, figsize= (20, 10))
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/wine_EDA Histogram.png")
plt.show()
# Alot of middle fingers?

# 2. Boxplots
num_col = wine_Df.select_dtypes(include = 'float')
plt.figure(figsize = (20, 10))
for i, col in enumerate(num_col):
    plt.subplot(5, 4, i+1)
    sns.boxplot(wine_Df[col], color=('xkcd:lime'))
    plt.title(f'{col}')
    plt.xlabel(' ')
plt.tight_layout()
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/wine_Boxplots.png")
plt.show()

# 3. Pairplot
# flight_pp = sns.pairplot(wine_Df)
# plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/wine_EDA Pairplot.png")
# plt.show()
# density is at a low level regardless of variable, although is seem to go well with r_sugar,
# TSD seem to pair well with sulphates.

#Variable Measurement
# 1. Categorical Variables
print()
print('CAT Variables')
print()
n_cat = []
non_col = []
for col in wine_Df.columns:
    if wine_Df[col].dtypes == object and wine_Df[col].nunique() < 15:
        n_cat.append(col)
        print('=='*7)
        print(f"{wine_Df[col].value_counts()}")
    elif wine_Df[col].dtypes == object: # Usually objects skip set parameters
        non_col.append(col)
print(non_col)
print(f'no. of CAT Variables: {len(n_cat)}')

# 2. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in wine_Df.columns:
    if wine_Df[col].dtypes != object and wine_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{wine_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Cont Variables')
n_cont = []
for col in wine_Df.columns:
    if wine_Df[col].dtypes != object and wine_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{wine_Df[col].min()} and Max_{wine_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(wine_Df.columns) - (len(n_cat) + len(n_num) + len(n_cont)))} columns")

## Exploring Data Relationship
# 1. GroupBys
col_df = []
for col in wine_Df.columns:
    col_df.append(col)
col_df.remove('style')
style_df = wine_Df.groupby(['style'])[col_df]
print(style_df.describe(percentiles=[]))

# 1b. Pivot Table
wine_style = wine_Df.pivot_table(col_df, ['style'])
print(wine_style.T)
# White wines have slightly more quality, fsd~tsd, r_sugar
# Red wines have more f_acidity, v_acidity, pH...

# 2. Correlation and HeatMap
wine_corr = wine_Df.corr()
print(wine_corr.abs())

mask = np.zeros_like(wine_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize= (20, 10))
sns.heatmap(wine_corr, mask = mask, vmin = -1.2, vmax = 1.2, center = 0, linewidths=.1,
            cbar_kws = {"shrink":.5}, cmap = 'coolwarm', square = False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/wine_EDA Heatmap.png")
plt.show()

# We can find a varying level of relationships, for instance between alcohol X density -.69
# TSD and FSD(.72), 'quality' only correlates positively and highly with alcohol.

## Moving Forward
# I see 'quality' has y, so it would be cool to use KNN and Feature selection metrics, to measure those
# distant relatives, compare for red and white wines.