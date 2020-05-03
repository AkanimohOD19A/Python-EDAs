import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

hostels_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/hostel_factors.csv")
print(hostels_Df.info())
print(hostels_Df.head().T)

## Initiating Data Checks and Cleaning
# assuming all 10 f/actors were measured on the same metrics, lets make a single factor

hostels_Df['avr_factor'] = (hostels_Df['f1']+hostels_Df['f2']+hostels_Df['f6']+hostels_Df['f4']+\
                       hostels_Df['f5']+hostels_Df['f3']+hostels_Df['f7']+hostels_Df['f8']+\
                       hostels_Df['f9']+hostels_Df['f10'])/10

## Plot, Pivot Table and Correlation
# 1. Histogram
hist = hostels_Df.hist(bins =10, figsize = (20, 10))
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/hostels_Factors histogram.png")
plt.show()

# ~7 & 8, are the most common ratings,
# as regards the avr_Factor, I think we have fewer factors influencing hotels, than otherwise.

# 2. Boxplots
plt.figure(figsize=(20, 10))

num_cols = hostels_Df.select_dtypes(include = 'float')
for i, col in enumerate(num_cols):
    plt.subplot(5, 4, i+1)
    sns.boxplot(hostels_Df[col], color = "xkcd:lime")
    plt.title(f"{col}", fontsize = 10)
    plt.xlabel(' ')
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/hostels_Factors Boxplot.png")
plt.show()

# All our variables, although small, seem to be well distributed with few outliers

# 3. Pairplots
hostels_pp = sns.pairplot(hostels_Df)
plt.savefig("C:/Users/HP/PycharmPRojects/Complete_EDA/hostels_Factors Pairplots.png")
plt.show()
# The pairplot looks like we are in a bag of fleas, I don't envisage any tangible correlation btw factors,
# Thus, let's tabulate the its relationship per hostel.

## Pivot Table and Group-bys Analysis
# 1. Pivot Table
hostels_PT = hostels_Df.pivot_table(['avr_factor','rating'], ['hostel'])
print(hostels_PT)

# 2. Group-by Analysis
hostels_FR = hostels_Df.groupby(['hostel'])['avr_factor','rating']
print(hostels_FR.describe(percentiles=[]).T)

# Due to the smallness od the data,
# this descriptive analysis is not reliable, min, max and mean are quite the same.