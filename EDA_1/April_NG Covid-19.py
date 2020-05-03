import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

NGApril_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/ngcovid19april.csv")


## Initiating Data Cleaning and Checks
drp = ['Year', 'Population*', 'Country', 'Import', 'Import %', 'Contact', 'Cummulative Cases', 'Source']
NGApril_Df.drop(columns = drp, inplace = True)

NGApril_Df['Month'] = pd.to_datetime(NGApril_Df['Month'])

print(NGApril_Df.info())
print(NGApril_Df.head())

## Building Data Profiles and Tables
# 1. Histogram
hist = NGApril_Df.hist(bins = 10, figsize= (20, 10))
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/NG Covid-19 April Histogram.png")
plt.show()
# Alot of middle fingers?

# 2. Boxplots
num_col = NGApril_Df.select_dtypes(include = 'float')
plt.figure(figsize = (20, 10))
for i, col in enumerate(num_col):
    plt.subplot(5, 4, i+1)
    sns.boxplot(NGApril_Df[col], color=('xkcd:lime'))
    plt.title(f'{col}')
    plt.xlabel(' ')
plt.tight_layout()
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/NGApril_Boxplots.png")
plt.show()

# 3. Pairplot
sns.pairplot(NGApril_Df)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/NG Covid-19 April Pairplot.png")
plt.show()


#Variable Measurement
# 1. Categorical Variables
print()
print('CAT Variables')
print()
n_cat = []
non_col = []
for col in NGApril_Df.columns:
    if NGApril_Df[col].dtypes == object and NGApril_Df[col].nunique() < 15:
        n_cat.append(col)
        print('=='*7)
        print(f"{NGApril_Df[col].value_counts()}")
    elif NGApril_Df[col].dtypes == object: # Usually objects skip set parameters
        non_col.append(col)
print(non_col)
print(f'no. of CAT Variables: {len(n_cat)}')

# 2. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in NGApril_Df.columns:
    if NGApril_Df[col].dtypes != object and NGApril_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{NGApril_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Cont Variables')
n_cont = []
for col in NGApril_Df.columns:
    if NGApril_Df[col].dtypes != object and NGApril_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{NGApril_Df[col].min()} and Max_{NGApril_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(NGApril_Df.columns) - (len(n_cat) + len(n_num) + len(n_cont)))} columns")

## Exploring Data Relationship
# 1. GroupBys
col_df = []
for col in NGApril_Df.columns:
    col_df.append(col)
col_df.remove('State')
col_df.remove('Month')
style_df = NGApril_Df.groupby(['State'])[col_df]
print(style_df.describe(percentiles=[]))

# 1b. Pivot Table
NGApril_style = NGApril_Df.pivot_table(col_df, ['State'])
print(NGApril_style.T)

# 2. Correlation and HeatMap
NGApril_corr = NGApril_Df.corr()
print(NGApril_corr.abs())

mask = np.zeros_like(NGApril_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize= (20, 10))
sns.heatmap(NGApril_corr, mask = mask, vmin = -1.2, vmax = 1.2, center = 0, linewidths=.1,
            cbar_kws = {"shrink":.5}, cmap = 'coolwarm', square = False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/NG Covid-19 April Heatmap.png")
plt.show()

## Moving Foward & Conclusion
# This is an ongoing data analysis, a time-series, and the methods applied here are bu=asic but lacking
# an appropriate time series modelling would be very appropriate.
