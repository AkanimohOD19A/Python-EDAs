import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format","{:.2f}".format)

fdelay_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/flight_delays_train.csv")
print(fdelay_Df.info())
print(fdelay_Df.head().T)#No NaNs

## Build Data Profiles and Tables
# 1. Histogram
hist = fdelay_Df.hist(bins = 10, figsize= (20, 10))
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/flights_Delay Histogram.png")
plt.show()
#Most flights delays are over shorter distances, flight delays decreases with distances.

# 2. Boxplot
num_cols = fdelay_Df.select_dtypes(include = int)

for i,col in enumerate(num_cols):
    plt.subplots(5, 4, i+1)
    sns.boxplot(fdelay_Df[col], color = 'xkcd:lime')
    plt.title(f"{col}", fontsize = 10)
    plt.xlabel(' ')
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/flights_Delay Boxplot.png")
plt.show()

# 3. Pairplot
flight_pp = sns.pairplot(fdelay_Df)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/flights_Delay Pairplot.png")
plt.show()
# distances seem to increase at a certain departure time, 1000

#Variable Measurement
# 1. Categorical Variables
print()
print('CAT Variables')
print()
n_cat = []
non_col = []
for col in fdelay_Df.columns:
    if fdelay_Df[col].dtypes == object and fdelay_Df[col].nunique() < 15:
        n_cat.append(col)
        print('=='*7)
        print(f"{fdelay_Df[col].value_counts()}")
    elif fdelay_Df[col].dtypes == object: # Usually objects skip set parameters
        non_col.append(col)
print(non_col)
print(f'no. of CAT Variables: {len(n_cat)}')

# 2. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in fdelay_Df.columns:
    if fdelay_Df[col].dtypes != object and fdelay_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{fdelay_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Cont and Num Variables')
n_cont = []
for col in fdelay_Df.columns:
    if fdelay_Df[col].dtypes != object and fdelay_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{fdelay_Df[col].min()} and Max_{fdelay_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(fdelay_Df.columns) - (len(n_cat) + len(n_num) + len(n_cont)))} columns")

# Since distance seem to be a interesting variable, lets compare origin and dest
# Pivot Table
pT = fdelay_Df.pivot_table(['Distance'], ['Origin','Dest'])
print(pT)

# This might be all the analysis we will need, however let's explore correlations.

## Exploring Data Relationship
fdelay_corr = fdelay_Df.corr()
print(fdelay_corr.abs())

mask = np.zeros_like(fdelay_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize= (20, 10))
sns.heatmap(fdelay_corr, mask = mask, vmin = -1.2, vmax = 1.2, center = 0, linewidths=.1,
            cbar_kws = {"shrink":.5}, cmap = 'coolwarm', square = False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/flights_Delay Heatmap.png")
plt.show()
# Returns only a correlation between distance and DepTime, which a very l0w -.021

## Conclusion
# Flights delays seem to be area specific, which explains the high value counts for places like ATL
# even over actual longer distances, this may be due to limits at Airports, however this would require more investigation

## Moving Forward
# We'll need to identify more methods to explore the relationship btw Dest and Origin flights