import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", "{:.2f}".format)

le_Df = pd.read_csv("C:/Users/HP/Documents/EDA/Python/Life_Expectancy/Life Expectancy Data.csv")
print(le_Df.info(), '\n')

print(le_Df.head())

#Initiating Checks
# Changing column names
le_Df.rename(columns = lambda x:x.strip().replace(' ', '_').lower(), inplace = True)
le_Df.rename(columns = {'thinness__1-19_years':'thinness_10-19_years'}, inplace = True)

## Managing Columns
# Interpolation for NaNs
na_cols = le_Df.loc[:, le_Df.isnull().any()].columns
for col in na_cols:
    le_Df.loc[:, col] = le_Df.loc[:, col].interpolate(limit_direction = 'both')
# This clears the null values, however in prior analysis some whole countries like, Sudan and S-Sudan
# have entire cols without values. For now we create a clean csv

# Changing dtype
le_Df['population'] = le_Df['population'].astype(int)
print(le_Df.info())

le_Df.to_csv("C:/Users/HP/Documents/EDA/Python/Life_Expectancy/clean_Life Expectancy Data.csv",
             encoding='utf-8', index=False)
print(le_Df.head().T)

## Build Data Profile Tables and Plots
# 1. Distribution with Histogram
hist = le_Df.hist(bins = 10, figsize = (20, 10))
plt.show()


# 2. Detecting outliers with Boxplots
plt.figure(figsize=(16, 200))

num_cols = le_Df.select_dtypes(include = 'float').columns
for i, col in enumerate(num_cols):
    plt.subplot(5, 4, i+1)
    sns.boxplot(le_Df[col], color=('xkcd:lime'))
    plt.title(f'{col}', fontsize=10)
    plt.xlabel('')
plt.tight_layout()
plt.show()

## Pairplot
#g = sns.pairplot(le_Df)
#plt.show()

## Measuring our variables
# 1. Categorical Variables
print()
print('CAT Variables')
print()
for col in le_Df.columns:
    if le_Df[col].dtypes == object and le_Df[col].nunique() < 10:
        print('=='*7)
        print(f"{le_Df[col].value_counts()}")

# 2. Numerical Variables
print()
print('Num Variables')
for col in le_Df.columns:
    if le_Df[col].dtypes != object and le_Df[col].nunique() < 25:
        print()
        print(f"{col}: Min_{le_Df[col].min()} and Max_{le_Df[col].max()}")

# 2. Continuous Variables
print()
print('Cont Variables')
for col in le_Df.columns:
    if le_Df[col].dtypes != object and le_Df[col].nunique() >= 25:
        print()
        print(f"{col}: Min_{le_Df[col].min()} and Max_{le_Df[col].max()}")


# Explore Data Relationships
## Pivot Tables and CrossTabs, c_Tabs is best between two categorical variables
# 1. CT: Distribution of status by year, result is quite redundant
le_status = pd.crosstab(le_Df['year'], le_Df['status'])
print(le_status) #Except for 2016, developing countries remained at 151.

# 2. PT:
le_pu5 = le_Df.pivot_table(['population', 'adult_mortality', 'under-five_deaths'], ['life_expectancy'])
print(le_pu5)
# Here we can see the distribution of le by the variables, by the result
# we should expect a strong and positive corr between mortality rates and the dependent var.

# Correlation
le_corr = le_Df.corr()
print(le_corr.abs())

mask = np.zeros_like(le_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (50, 10))
sns.heatmap(le_corr, mask = mask, center = 0, square=False, vmin= -1.2, vmax = 1.2,
            cbar_kws={"shrink": .5}, cmap="coolwarm", linewidths=.1, annot = True)
plt.show()
#The ralationship to LE is highest btw schooling & ICR (~+.70%) and adult_mortality (-.70%)
# From the map we can observe a intuitive features of intervening variables on LE
# Other interesting relationships are btw schooling and ICR, <5 & IMR - a solid 1, thinness(es)
# and negatively btw adult_mort and LE

# Moving forward Features Selection would be appropriate, PCA.