import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format',"{:.2f}".format)

# Initiating & Glimpsing Data
titanic_Df = pd.read_csv("C:/Users/HP/Documents/EDA/R/S_R/titanic/train.csv")

print(titanic_Df.info())

print(titanic_Df.head())
print(titanic_Df.shape)

# Data Cleaning
#Age: There are certain indexes with < 1
#print(titanic_Df[titanic_Df.Age == 0.42].T)
print(titanic_Df[titanic_Df['Age'] < 1])

titanic_Df.drop(['Cabin'], axis = 1, inplace=True)
titanic_Df.drop(titanic_Df.index[[78, 305, 469, 644, 755, 803, 831]], inplace = True)
#titanic_Df.drop(titanic_Df.index[titanic_Df.Age > 80], inplace = True)
titanic_Df = titanic_Df.reset_index(drop = True)

print(titanic_Df.shape)

titanic_Df = titanic_Df[titanic_Df.Age.notna()]
# titanic_Df = titanic_Df.reset_index(); this adds another column called 'Index'

#print(titanic_Df[titanic_Df.Embarked.isnull()], '\n', titanic_Df.loc[ titanic_Df.Embarked.isnull()])
titanic_Df.Embarked.fillna(method = 'ffill', inplace=True)
print(titanic_Df.info())

# Let's save the clean the data set
titanic_Df.to_csv("C:/Users/HP/Documents/EDA/R/S_R/titanic/clean_train.csv", encoding = "utf-8", index = False)

# Exploration
## Visualize Distributions
# 1. Histograms
hist = titanic_Df.hist(bins = 10, figsize = (10, 10))
plt.show()
# From the graph we can easily id the cat from numerical variables,
# and for where we have var __ we see a skewed distribution favouring lower sets

# 2. Box plots
boxplot = titanic_Df.boxplot(vert = True, fontsize=15)
plt.show()
# Ugly Plot,
# PassengerId is very elastic, due to the ordered and exempting nature of its numerical dtype,
# Age has a few expected variables, for instance we only have one 80YO.
# Fare, from the previous plots, we can obs it has a bulk of its shape towards the lower fares,
# and the outliers- at higher fares- fall outside this density.

# 3. Pair plots
pp = sns.pairplot(titanic_Df)
plt.show()
# The slope in this plot is a mirror of the prior histogram,
# otherwise the plot show interesting relationships, we will explore in the next analysis on Correlation.

## Identifying and Measuring our vars
# 1. Categorical Variable
for col in titanic_Df.columns:
    if titanic_Df[col].dtypes == object and titanic_Df[col].nunique() < 10:
        print(f"{col} :\n {titanic_Df[col].value_counts()}")
        print('--'*7)

# 2. Numerical/ Integers
for col in titanic_Df.columns:
    if titanic_Df[col].dtypes != object and titanic_Df[col].nunique() <= 25:
        print(f"{col} :\n {titanic_Df[col].value_counts()}")
        print('=='*7)

# 3. Continuous Variable
for col in titanic_Df.columns:
    if titanic_Df[col].dtypes != object and titanic_Df[col].nunique() > 25:
        print(f"{col} : MIN_ {titanic_Df[col].min()} and MAX_ {titanic_Df[col].max()}")
        print('++'*7)

# Correlation on Survival
p_sur = titanic_Df.corr()

print(p_sur.Survived, "\n", p_sur.Survived.abs() )
# Heatmap
mask = np.zeros_like(p_sur, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(p_sur, mask = mask, center = 0, square = True,
            linewidth = .5, cbar_kws = {"shrink": .5}, vmin = -1.2, vmax =1.2,
            cmap = "YlGnBu", annot = True)
plt.show()

# From our results we can observe that Fare{.27} and Pclass{-.36}
# represent the two spectrum of survival, and both vars in themselves share a very negative relationship

### Exception
# Another method for classifying vars

#df_cat = titanic_Df.select_dtypes(include="object")
#for col in df_cat.columns:
#    print()
#    pr
#    int(df_cat[col].value_counts())

