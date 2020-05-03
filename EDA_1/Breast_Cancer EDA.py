import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", "{:.2f}".format)

cbrst_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/breast_cancer.csv")
print(cbrst_Df.info())
print(cbrst_Df.head())

# Initiating Checks and Cleaning
cbrst_Df.drop(columns = ['Unnamed: 32'], inplace = True)

cbrst_Df.to_csv("C:/Users/HP/Desktop/EDAs/clean breast_cancer.csv")

## Build Data Profiles and Table
# Histogram
cb_hist = cbrst_Df.hist(bins = 10, figsize = (50, 20))
plt.savefig('C:/Users/HP/PycharmProjects/breast_Cancer Histogram.png')
plt.show()

# Boxplot
plt.figure(figsize=(20, 10)) # width x height

# Numerical vars
num_cols = cbrst_Df.select_dtypes(include = 'int64')
num_cols = num_cols.append(cbrst_Df.select_dtypes(include = 'float64'))
for i, col in enumerate(num_cols):
    plt.subplot(9, 7, i+1)
    sns.boxplot(cbrst_Df[col], color='xkcd:lime')
    plt.title(f'{col}', fontsize=10)
    plt.xlabel('')
plt.tight_layout()
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/breast_cancer Boxplot.png")
plt.show()
# All area appended variables have quite incredible outliers

# Pairplot
#cbrst_pp = sns.pairplot(cbrst_Df)
#plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/breast_cancer Pairplot.png")
#plt.show()

## Variable measurement
print(cbrst_Df.diagnosis.value_counts()) # The only 'object' variable.

print()
print('Cont and Num Variables')
n_cont = []
for col in cbrst_Df.columns:
    if cbrst_Df[col].dtypes != object:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{cbrst_Df[col].min()} and Max_{cbrst_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(cbrst_Df.columns) - len(n_cont) )} columns")

## Explore Data Relationships
cbrst_corr = cbrst_Df.corr()
print(cbrst_corr.abs())

mask = np.zeros_like(cbrst_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (50, 20))
sns.heatmap(cbrst_corr, mask = mask, vmin = -1.2, vmax = 1.2, center = 0,
            linewidths = .1, cbar_kws={'shrink': .5}, cmap = 'coolwarm',
            square = False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/breast_cancer Heatmap.png")
plt.show()

## Conclusion
# It would take a specialist, or an analyst with a more robust knowledge of ,
# However, we can see the great influence of radius mean on the variables

##Moving Forward
# This would depend on some specialized knowledge of Oncology.