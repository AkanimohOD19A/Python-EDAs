import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", "{:.2f}".format)

ckidney_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/chronic_kidney_disease.csv")

print(ckidney_Df.info())
print(ckidney_Df.head().T) #No NaNs


##Initiating Data Checks and Cleaning

## dropping '?'
o_index = ckidney_Df.shape[0]
q_col = []
for col in ckidney_Df.columns:
    if len(ckidney_Df[ckidney_Df[col] == '?'] ) >= 1:
        q_col.append(col)
        ckidney_Df.drop(ckidney_Df.index[ckidney_Df[ckidney_Df[col] == '?'].index], inplace = True)
        ckidney_Df = ckidney_Df.reset_index(drop = True)
print(len(q_col),'columns')
print(f"dropped {o_index - ckidney_Df.shape[0]} rows of data") # Thia alot

# adjusting dtype
int_df = ["age", "bp", "al", "su", "bgr", "bu", "sod", "pcv", "wc"]
flt_df = ["rc", "hemo", "pot", "sc", "sg"]
for col in int_df:
    ckidney_Df[col] = ckidney_Df[col].astype("int64")
for i in flt_df:
    ckidney_Df[col] = ckidney_Df[col].astype(float)

print(ckidney_Df.info())
print(ckidney_Df.head().T) #No NaNs
ckidney_Df.to_csv("C:/Users/HP/Desktop/EDAs/clean chronic_kidney_disease.csv", encoding="utf-8", index = False)

## Building Data Tables and Profiles
# 1. Histogram
hist = ckidney_Df.hist(bins = 10, figsize = (20, 10))
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/chronic_kidney_disease histogram.png")
plt.show()

# 2. Boxplots
plt.figure( figsize = (10, 10))

num_cols = ckidney_Df.select_dtypes(include = 'float64')
num_cols = num_cols.append(ckidney_Df.select_dtypes(include = 'int64'))

for i,col in enumerate(num_cols):
    plt.subplot(5, 2, i+1)
    sns.boxplot(ckidney_Df[col], color = "xkcd:lime")
    plt.title(f"{col}", fontsize = 10)
    plt.xlabel(' ')
plt.tight_layout()
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/chronic_kidney_disease boxplot.png")
plt.show()

# 3. PairPlots
# ckidney_pp = sns.pairplot(ckidney_Df)
# plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/chronic_kidney_disease pairplot.png")
# plt.show()

## Variable Measurements
# 1. Categorical Variables
print()
print('CAT Variables')
print()
n_cat = []
non_col = []
for col in ckidney_Df.columns:
    if ckidney_Df[col].dtypes == object and ckidney_Df[col].nunique() < 15:
        n_cat.append(col)
        print('=='*7)
        print(f"{ckidney_Df[col].value_counts()}")
    elif ckidney_Df[col].dtypes == object: # Usual skip
        non_col.append(col)
print(non_col)
print(f'no. of CAT Variables: {len(n_cat)}')

# 2. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in ckidney_Df.columns:
    if ckidney_Df[col].dtypes != object and ckidney_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{ckidney_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Continuous Variables')
n_cont = []
for col in ckidney_Df.columns:
    if ckidney_Df[col].dtypes != object and ckidney_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{ckidney_Df[col].min()} and Max_{ckidney_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(ckidney_Df.columns) - (len(n_cat) + len(n_num) + len(n_cont)))} columns")

## Exploring Data Relationships
ckidney_corr = ckidney_Df.corr()
print(ckidney_corr.abs())

mask = np.zeros_like(ckidney_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (20, 10))
sns.heatmap(ckidney_corr, mask = mask, vmin = -1.2, vmax = 1.2, center = 0,
            linewidths = .1, cbar_kws = {"shrink": .5}, cmap = "coolwarm",
            square = False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/chronic_kidney_disease heatmap.png")
plt.tight_layout()
plt.show()

# Conclusion
# There are lot of unfamiliar cols here, for any satisfactory conclusions, however the EDA still shows
# relationships, so tangible they have a 1 corr.

## Moving Forward
# Should involve some understanding of subject/matter and approriate method for Nephrology