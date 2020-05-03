import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mthdg_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/drugs-and-math.csv")
print(mthdg_Df.info())
print(mthdg_Df)

# Clean, Save and Plots
mthdg_Df.drop(columns = ['Unnamed: 0'], inplace=True)

mthdg_Df.to_csv("C:/Users/HP/Desktop/EDAs/clean_drugs-and-math.csv")

hist = mthdg_Df.hist(bins = 10, figsize = (20, 10))
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/drug_and_math histogram")
plt.show()

pp = sns.pairplot(mthdg_Df)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/drug_and_math pairplot")
plt.show()
# From the pairplot we can see a clear -ve relationship between Drugs and score,
# we plot a heatmap to measure this relationship

mthdg_corr = mthdg_Df.corr()
print(mthdg_corr)

mask = np.zeros_like(mthdg_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(20, 10))
sns.heatmap(mthdg_corr, mask = mask, vmin =-1.2, vmax=1.2, center = 0,
            cbar_kws={'shrink': .5}, cmap = 'coolwarm', linewidths=.5,
            square=False, annot = True)
plt.tight_layout()
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/drug_and_math Heatmap")
plt.show()
# A very high -ve relative, at -.94.
# DDD!
