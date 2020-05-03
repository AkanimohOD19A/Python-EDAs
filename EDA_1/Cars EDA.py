import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", "{:.2f}".format)

car_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/cars.csv", sep = ';')
print(car_Df.info())
print(car_Df.head().T)

## Initiating Checks and Cleaning
car_Df.drop(columns = ['Obs'], inplace = True)
curr = ['MSRP', 'Invoice']
for i in curr:
    car_Df[i].apply(lambda cell: cell.strip('$'))
    car_Df[i].apply(lambda cell: cell.strip(','))

## Build Data Profiles and Tables
# 1. Histogram
car_hist = car_Df.hist(bins = 10, figsize=(20, 10))
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/car histogram.png")
plt.show()
# Engine size is generally around 2-4; HP 150-200; Length 170-200;
# MPG-City 20-25; MPG-HW 25-30; Weight 3000-4000; Wheelbox 100-115.

# 2. Boxplot
plt.figure(figsize=(20, 10)) # width x height

num_cols = car_Df.select_dtypes(include = 'int64')
num_cols = num_cols.append(car_Df.select_dtypes(include = 'float64'))
for i, col in enumerate(num_cols):
    plt.subplot(5, 2, i+1)
    sns.boxplot(car_Df[col], color='xkcd:lime')
    plt.title(f'{col}', fontsize=10)
    plt.xlabel('')
plt.tight_layout()
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/car Boxplot.png")
plt.show()

# 3. Pairplot
#car_pp = sns.pairplot(car_Df)
#plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/car Pairplot.png")
#plt.show()

# For Bi-variate Relationship: We can already the glean the nature relationship
# like between +ve 'engine size' and [HP, Weight, Wheelbase, Length]
# -ve [MP_city/height], much of what we hope to see in the heatmap.

## Variable Measurement
# 1. Categorical Variables
print()
print('CAT Variables')
print()
n_cat = []
non_col = []
for col in car_Df.columns:
    if car_Df[col].dtypes == object and car_Df[col].nunique() < 15:
        n_cat.append(col)
        print('=='*7)
        print(f"{car_Df[col].value_counts()}")
    elif car_Df[col].dtypes == object: # Usually objects skip set parameters
        non_col.append(col)
print(non_col)
print(f'no. of CAT Variables: {len(n_cat)}')

# 2. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in car_Df.columns:
    if car_Df[col].dtypes != object and car_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{car_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Cont and Num Variables')
n_cont = []
for col in car_Df.columns:
    if car_Df[col].dtypes != object and car_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{car_Df[col].min()} and Max_{car_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(car_Df.columns) - (len(n_cat) + len(n_num) + len(n_cont)))} columns")

## Exploring Data Relationships
car_corr = car_Df.corr()
print(car_corr.abs())

mask = np.zeros_like(car_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (20, 10))
sns.heatmap(car_corr, mask = mask, center = 0, vmin = -1.2, vmax = 1.2,
            linewidths = .1, cbar_kws = {'shrink': .5}, cmap = 'coolwarm',
            square = False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/car Pairplot.png")
plt.show()
# Measures the exact extent of relationship between numerical variables as seen in the previous Pairplot.

# Conclusion
# Taking Engine Size as y, we have certain variables that can predict this properly ,
# MPG_city is very closely related to highways, thus one can be dropped or create an avr col

## Moving Forward
# Since we stopped at correlation and assume a y, the best immediate thing to do is create various models,
# then test them. However other methods as per subject matter would be cool.