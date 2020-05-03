import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
pd.set_option("display.float_format", "{:.2f}".format)

banks_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/bank.csv")
## Initiating checks and cleaning
print(banks_Df.info())
print(banks_Df.head().T) #Clean Dataset
# I'll to change 'unknown' jobs to others
#banks_Df['job'] = banks_Df.job.replace({'unknown':'others'}, inplace = True)

## Build Data Profiles and Tables
banks_Df['pdays'] = banks_Df.pdays.astype(float)
# 1. Histograms - returns only integers
bank_hist = banks_Df.hist(bins =10, figsize = (20, 10))
plt.show()
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/bank_num_hist.png")
plt.close()

# 2. CAT Plot
#for col in banks_Df.select_dtypes(include = 'object').columns:
#    banks_Df[col].hist(bins = 10, figsize = (20, 10))
#    plt.title(f"{col}", fontsize = 20)
#    plt.show()

# 3. Box Plot
plt.figure(figsize=(40, 10)) # width x height

num_cols = banks_Df.select_dtypes(include = 'int64').columns
num_cols = num_cols.append(banks_Df.select_dtypes(include = 'float').columns)
for i, col in enumerate(num_cols):
    plt.subplot(4, 2, i+1)
    sns.boxplot(banks_Df[col], color=('xkcd:lime'))
    plt.title(f'{col}', fontsize=10)
    plt.xlabel('')
plt.tight_layout()
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/bank_boxplots.png")
plt.show()
# There are variables here, that considering their outliers would be difficult to predict,
# Perhaps a better understanding of the dataset would require a log10 transformation.

## 3. Pairplots: plot pairwise bi-variate distribution
#banks_pp = sns.pairplot(banks_Df, dropna = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/bank_pairplots.png")
plt.show()
# Age variable is interesting to observe here, the bulk are btw 20 and 50,
# and are quite reactive given its response to other variables
plt.close()
## Measuring Variables
# 1. Categorical Variables
print()
print('CAT Variables')
print()
n_cat = []
for col in banks_Df.columns:
    if banks_Df[col].dtypes == object and banks_Df[col].nunique() < 12:
        n_cat.append(col)
        print('=='*7)
        print(f"{banks_Df[col].value_counts()}")
print(f'no. of CAT Variables: {len(n_cat)}')

# 2. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in banks_Df.columns:
    if banks_Df[col].dtypes != object and banks_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{banks_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Cont Variables')
n_cont = []
for col in banks_Df.columns:
    if banks_Df[col].dtypes != object and banks_Df[col].nunique() >= 20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{banks_Df[col].min()} and Max_{banks_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {abs((len(n_cat) + len(n_num) + len(n_cont)) - len(banks_Df.columns))} columns")

# Exploring Relationships
banks_corr = banks_Df.corr()
print(banks_corr.abs())

mask = np.zeros_like(banks_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)]  = True

plt.figure(figsize = (20, 10))
sns.heatmap(banks_corr, mask = mask, vmin = -1.2, vmax = 1.2, center = 0, linewidths = .1,
            cbar_kws={'shrink':.5}, cmap = 'coolwarm', annot = True, square = False)
# I changed the arguement for 'cmap' to 'cbar', and the result was COOL.
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/bank_heatmap.png")
plt.show()
# y is  most affected positively by 'duration' of loan
# pdays has the highest correlation, with 'previous'.
# As far as negative relationships, it is weak, the strongest is -.09 btw pdays X days & campaign,
# the latter which share a considerable positive relationship

# Correlation here is limited to certain variables, to transform our object* to numbers would exhaust our goals for now,
# it would be insightful to include more variables

## Moving Forward
# This a very interesting and  data set, however this is just an EDA project,
# but would lovely to explore and even predict relationships - KNN