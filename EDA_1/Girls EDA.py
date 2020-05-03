import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

girls_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/girls.csv")
print(girls_Df.info())
print(girls_Df.head()) # No NaNs

## Building Data Tables and Profiles
# 1. Histogram
hist = girls_Df.hist(bins =10, figsize= (20, 10) )
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/girls histogram.png")
plt.show()

# Hips, Waist, Weight seem to follows a normal distribution, as regards we don't have a lot of short girls.

# 2. Boxplots
plt.figure(figsize=(20, 10))

num_cols = girls_Df.select_dtypes(include = 'int64')
for i, col in enumerate(num_cols):
    plt.subplot(5, 2, i+1)
    sns.boxplot(girls_Df[col], color = "xkcd:lime")
    plt.title(f"{col}", fontsize = 10)
    plt.xlabel(' ')
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/girls Boxplot.png")
plt.show()
# This shows that a alot of ladies do have hips, and don't weigh too high

# 3. Pairplots
girls_pp = sns.pairplot(girls_Df)
plt.savefig("C:/Users/HP/PycharmPRojects/Complete_EDA/girls heatmap.png")
plt.show()
# From here we can manage to say that weight would correlate well with height, then quite mildly
# hips and waist, i.e when one of such cols started to change these other proponents changed accordingly
# which honestly is not far fetched, our girls don't end up looking like amoebas.

## Variable Measurement
# 1. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in girls_Df.columns:
    if girls_Df[col].dtypes != object and girls_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{girls_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Cont and Num Variables')
n_cont = []
for col in girls_Df.columns:
    if girls_Df[col].dtypes != object and girls_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{girls_Df[col].min()} and Max_{girls_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(girls_Df.columns) - (len(n_num) + len(n_cont)))} columns")

## Explore Data Relationships
# Groupbys-
# 1. Month & Year
columns_to_show = ['Bust', 'Waist', 'Weight']
MY_df = girls_Df.groupby(['Month','Year'])[columns_to_show]
print(MY_df.describe(percentiles=[]))
# 2. Year
columns_to_show = ['Bust', 'Waist', 'Weight']
year_df = girls_Df.groupby(['Year'])[columns_to_show]
print(year_df.describe(percentiles=[]))
# Average Bust actually reduced with the passing year.

# 3. Month
columns_to_show = ['Bust', 'Waist', 'Weight']
month_df = girls_Df.groupby(['Month'])[columns_to_show]
print(month_df.describe(percentiles=[]))
# Girls get more avrage bust in January, it either a data error of the celebration turkey..
# Lets check for the measurement of relationships.

# Correlation and HeatMap
girls_corr = girls_Df.drop(columns = ['Year']).corr()
print(girls_corr.abs())
plt.figure(figsize = (20, 10))

mask = np.zeros_like(girls_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(girls_corr, mask = mask, vmin=-1.2, vmax=1.2, center = 0,
         linewidth = .1, cbar_kws = {"shrink":.5}, cmap="coolwarm",
         square = False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/girls Heatmap.png")
plt.show()

## Conclusion
# There are mild to high correlations in this dataset, weight correlates well every col,
# especially height +.71,
# Bust explains Hips quite highly at +.46

## Moving Forward
# Better specialized and appropriate methods.