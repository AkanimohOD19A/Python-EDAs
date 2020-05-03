import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", "{:.2f}".format)

bkrent_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/bikes_rent.csv")
bk_Df = bkrent_Df.copy() # For later use in corr...

print(bkrent_Df.info())
print(bkrent_Df.head(10).T) #No NaNs

#Initiating Data Checks and Cleaning
# Adjusting Column dtype
bool_col = ['holiday', 'workingday', 'yr']
for col in bool_col:
    bkrent_Df[col].replace({0:"No", 1:"Yes"}, inplace = True)
    #print(f"{bkrent_Df[col].value_counts()}") doesn't work??

bkrent_Df['weekday'].replace({0:"Sunday", 1:"Monday",
                              2:"Tuesday", 3:"Wednesday",
                              4:"Thursday", 5:"Friday",
                              6:"Saturday"}, inplace = True)

bkrent_Df['mnth'].replace({1:"January", 2:"Febuary", 3:"March", 11:"November",
                         4:"April", 5:"MAy", 6:"June", 7:"July",
                         8:"August", 9:"September", 10:"October", 12:"December"
                         }, inplace = True)
# Note: both week & year coarse are not necessary, but important for method emphasis
bkrent_Df['season'] = bkrent_Df['season'].astype(dtype = 'int')

print(bkrent_Df.head().T)
#Save
bkrent_Df.to_csv("C:/Users/HP/Desktop/EDAs/clean_bikes_rent.csv")

## Build DataProfiles and Tables
# 1. Histogram
hist = bkrent_Df.hist(bins = 10, figsize=(20, 10))
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/bike_rent Histogram.png")
plt.show()

## Assuming 0 to be No and 1 Yes,
# Average Temperature for rents is usually between 12 and 35 deg. Humidity 50 - 75
# A great drop in rents, during holidays
# During the months, there a higher rents at the beginning and end of the year,
# which is kinda counter intuitive to our knowledge of response to Holidays
# the four seasons have identical rates
# the 2 windspeed parameters mirror each other
# on a working day there a more rents

# 2. Boxplots
plt.figure(figsize=(20, 10)) # width x height

# Non boolean Numerical vars
num_cols = bkrent_Df.select_dtypes(include = 'int64')
num_cols = num_cols.append(bkrent_Df.select_dtypes(include = 'float64'))
for i, col in enumerate(num_cols):
    plt.subplot(5, 2, i+1)
    sns.boxplot(bkrent_Df[col], color=('xkcd:lime'))
    plt.title(f'{col}', fontsize=10)
    plt.xlabel('')
plt.tight_layout()
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/bike_rent Boxplot.png")
plt.show()
# There are no high outliers, however it should be clear whenever we see plots like yr & workingday,
# that they are dichotomous.

# 3. Pairplots
#bike_pp = sns.pairplot(bk_Df)
#plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/bike_rent PAirplot.png")
#plt.show()

# Continuing with our plot analysis, this bivariate plot already shows how
# highly correlated some vars will be. And how temperature/weather might influence rent.

## Variable Measurement
# 1. Categorical Variables
print()
print('CAT Variables')
print()
n_cat = []
for col in bkrent_Df.columns:
    if bkrent_Df[col].dtypes == object and bkrent_Df[col].nunique() < 12:
        n_cat.append(col)
        print('=='*7)
        print(f"{bkrent_Df[col].value_counts()}")
print(f'no. of CAT Variables: {len(n_cat)}')

# 2. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in bkrent_Df.columns:
    if bkrent_Df[col].dtypes != object and bkrent_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{bkrent_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Cont and Num Variables')
n_cont = []
for col in bkrent_Df.columns:
    if bkrent_Df[col].dtypes != object and bkrent_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{bkrent_Df[col].min()} and Max_{bkrent_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(bkrent_Df.columns) - (len(n_cat) + len(n_num) + len(n_cont)))} columns")

## Exploring Data Relationships
# 1. Times X cnt
time_pt = bkrent_Df.pivot_table(['cnt'], ['mnth', 'weekday'])
print(time_pt)

# 2. Correlation and heatmap
bkrent_corr = bk_Df.corr()
print()
print(bkrent_corr.abs())

mask = np.zeros_like(bkrent_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize= (20, 10))
sns.heatmap(bkrent_corr, mask = mask, center = 0, vmin = -1.2, vmax = 1.2,
            linewidths = .1, cbar_kws = {'shrink': .5}, cmap = 'coolwarm',
            square = False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/bike_rents Heatmap.png")
plt.show()
# From the corr-analysis, the biggest influence on rent('cnt'), is temp, year, season and mnth +ve
# -ve, weathersit, speed are strong influence, the other variables are just intervening.

# Conclusion
# this data was divided to preserve its numerical features in bk_Df, which I think would be
# relevant for deeper exploration of relationships, however bike rents seem to be a seasonal game
# but decision makers will have to consider how to manipulate the range of intervening variables to scale.

## Moving Forward
# My first intuition is time what times of the year contribute more, of course we can do
# that with pivot Table, but would be interesting to apply more sophisticated methods.