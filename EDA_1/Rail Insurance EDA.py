import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

claims_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/Rail_Insurance_Claims.csv")
print(claims_Df.info())
print(claims_Df.head()) #No NaNs

claims_Df.rename(columns = {'# OF STOPS':'STOPS'}, inplace = True)
claims_Df[claims_Df['WEIGHT'] == 0] #500 rows seem to be 0., for lack of more info lets them to have the mean weight
claims_Df['WEIGHT'].replace({0:131.28}, inplace =True)
# Above, is to preserve our :.2f, claims_Df['WEIGHT'].replace({0:claims_Df['WEIGHT'].mean()}, inplace =True)

## Build Data Tables and Profiles
# 1. Histogram-Numerical Variables
claims_hist = claims_Df.hist(bins = 10, figsize = (20, 10))
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/railway_claims Histogram.png")
plt.show()
# Rails usually use btw 25 to 55[] of fuel and stops are <5
# Mails seem to be evenly distributed

# 2. Boxplot
plt.figure(figsize = (40, 10))

num_cols = claims_Df.select_dtypes(include = "int64").columns # adjust with value_counts*
for i,col in enumerate(num_cols):
    plt.subplot(5, 4, i+1)
    sns.boxplot(claims_Df[col], color = 'xkcd:lime')
    plt.title(f"{col}", fontsize = 10)
    plt.xlabel(' ')
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/railway_claims Boxplot.png")
plt.show()

# 3. PairPlots
# claims_pp = sns.pairplot(claims_Df)
# plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/railway_claims Pairplot.png")
# plt.show()

# Miles, Fuel Used, Stops and weight fade with more damages, which is rational however
# the more damage there there seem to be more car value, perhaps car value is propensity for scrap

# Weight maintains a positive trend with Fuel Used,
# however I must have filled weight inappropriately since the mean distribution is so prominent.*

## Variable Measurement
# 1. Categorical Variables
print()
print('CAT Variables')
print()
n_cat = []
non_col = []
for col in claims_Df.columns:
    if claims_Df[col].dtypes == object and claims_Df[col].nunique() < 15:
        n_cat.append(col)
        print('=='*7)
        print(f"{claims_Df[col].value_counts()}")
    elif claims_Df[col].dtypes == object: # Usually objects skip set parameters
        non_col.append(col)
print(non_col)
print(f'no. of CAT Variables: {len(n_cat)}')

# 2. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in claims_Df.columns:
    if claims_Df[col].dtypes != object and claims_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{claims_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Continuous Variables')
n_cont = []
for col in claims_Df.columns:
    if claims_Df[col].dtypes != object and claims_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{claims_Df[col].min()} and Max_{claims_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(claims_Df.columns) - (len(n_cat) + len(n_num) + len(n_cont)))} columns")

# Exploring Data Relationships
claims_corr = claims_Df.corr()
print(claims_corr.abs())

mask = np.zeros_like(claims_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (20, 10))
sns.heatmap(claims_corr, mask = mask, center =0, vmin =-1.2, vmax = 1.2,
            linewidths= .1, cbar_kws = {'shrink': .5}, cmap = 'coolwarm',
            square =False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/railway_claims Heatmap.png")
plt.show()
# The most tangible relationship is between FUEL USED and WEIGHT, .091
# All other relationships are mild, and would act as intervening variables.

## Moving Forward
# Explore more methods.. exploration and cleaning