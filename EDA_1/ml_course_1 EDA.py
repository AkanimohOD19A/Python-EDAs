import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ml1_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/mlbootcamp5_train.csv", sep=';')
print(ml1_Df.info())
print(ml1_Df.head().T)

## Initiating Data Checks and Cleaning
for col in ml1_Df.columns:
    if ml1_Df[col].nunique() == 2:
        ml1_Df[col] = ml1_Df[col].replace({0:'No', 1:'Yes'}) #Assumption*

## Building Data Profiles and Tables
# 1. Histogram
hist = ml1_Df.hist(bins = 10, figsize= (20, 10))
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/ml1 Histogram.png")
plt.show()
# Height and weight mirror each other, and seem to follow a normal distribution.
# The same with api_lo and api_hi

# 2. Pairplot
# flight_pp = sns.pairplot(ml1_Df)
# plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/ml1 Pairplot.png")
# plt.show()
# The variables aforementioned to mirror, have characteristical distribution with other cols.

#Variable Measurement
# 1. Categorical Variables
print()
print('CAT Variables')
print()
n_cat = []
non_col = []
for col in ml1_Df.columns:
    if ml1_Df[col].dtypes == object and ml1_Df[col].nunique() < 15:
        n_cat.append(col)
        print('=='*7)
        print(f"{ml1_Df[col].value_counts()}")
    elif ml1_Df[col].dtypes == object: # Usually objects skip set parameters
        non_col.append(col)
print(non_col)
print(f'no. of CAT Variables: {len(n_cat)}')

# 2. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in ml1_Df.columns:
    if ml1_Df[col].dtypes != object and ml1_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{ml1_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Cont Variables')
n_cont = []
for col in ml1_Df.columns:
    if ml1_Df[col].dtypes != object and ml1_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{ml1_Df[col].min()} and Max_{ml1_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(ml1_Df.columns) - (len(n_cat) + len(n_num) + len(n_cont)))} columns")

## Exploring Data Relationship
ml1_corr = ml1_Df.corr()
print(ml1_corr.abs())

mask = np.zeros_like(ml1_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize= (20, 10))
sns.heatmap(ml1_corr, mask = mask, vmin = -1.2, vmax = 1.2, center = 0, linewidths=.1,
            cbar_kws = {"shrink":.5}, cmap = 'coolwarm', square = False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/ml1 Heatmap.png")
plt.show()

# Given knowledge of BioSci, the correlations were quite expected, gluc X cholesterol +.45, weight X height
# while every other variable seem to just intervene.

## Moving Forward
# Boxplots gave some challenges, hope to resolve that.
# Indepth and Appropriate method as regards subject matter, this was cleared gathered for ML purposes.