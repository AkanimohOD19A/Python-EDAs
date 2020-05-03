import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", "{:.2f}".format)

ibm_hr_Df = pd.read_csv("C:/Users/HP/Documents/EDA/Python"
                        "/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# NOTE
## Strip the data into a smaller shape, using '= ibm_hr_Df.loc[:,999]', it should make analysis faster


# Initiating Checks
print(ibm_hr_Df.info()) #1470 X 35, No NA's
print(ibm_hr_Df.head().T)

# Lets sample a couple workers
print(ibm_hr_Df.loc[[408,1336]].T)
# 0, is a single lady in her 40's that left:
# 1207, is a divorced male in her 50's that left:
# Interesting variables that are suspected to influence Attrition so far,
# seem to be JobSatisfication, OverTime, MaritalStatus and the strangely the relevance of EducationField and Department


# Building Data Profiles, Tables and Plots
## Histogram _integers/num
hist = ibm_hr_Df.hist(bins = 10, figsize=(20, 10))
plt.show()
# Among others, a number of vars mirrored JLevel, income wise like: StockOptionLevel, MonthlyRates %SalaryHike, etc
# We also will find a number of redundant variables, SHour, ECount and O18, thus their drops.

drp = ['StandardHours', "EmployeeCount", "Over18"]
ibm_hr_Df.drop(columns = drp, inplace=True)

## Pair plots
#pp = sns.pairplot(ibm_hr_Df) #Muted forits large values
#plt.show()

## BoxPlots
drp = ["MonthlyIncome","DailyRate", "EmployeeNumber", "MonthlyRate"]
bp = ibm_hr_Df.drop(columns = drp).boxplot(vert = False)
plt.show()
# The chart depicts what we already seen in the previous charts, outliers in Years spent, age of employers...,
# it looks quite normal, however certain vars, the ones exempted above,
# expected depicted very high values in itself and outliers.

# Exploring Relationships:
# 1. Measurements
## CAT
print()
print("Categorical Variables")
for col in ibm_hr_Df.columns:
    if ibm_hr_Df[col].dtypes == object and ibm_hr_Df[col].nunique() < 5:
        print(f"{ibm_hr_Df[col].value_counts()}")
        print("=="*7)

## NUM
print()
print("Numerical Variables")
for col in ibm_hr_Df.columns:
    if ibm_hr_Df[col].dtypes != object and ibm_hr_Df[col].nunique() < 10:
        print(f"{ibm_hr_Df[col].value_counts()}")
        print("=-="*5)

## CONT
print()
print("Continuous Variables")
for col in ibm_hr_Df.columns:
    if ibm_hr_Df[col].dtypes != object and ibm_hr_Df[col].nunique() > 10:
        print(f"{col}: Min_{ibm_hr_Df[col].min()}; Max_{ibm_hr_Df[col].max()}")
        print("-=-"*5)


# 2. Correlations
hr_corr = ibm_hr_Df.corr()
print(hr_corr, '\n', hr_corr.abs())

## Visualization
mask = np.zeros_like(hr_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize= (40, 20))
sns.heatmap(hr_corr, mask=mask, linewidths= .5, center = 0, square = False, cbar_kws = {"shrink": .5},
            vmin = -1.2, vmax= 1.2, cmap = "coolwarm", annot = True)
plt.show()

# Conclusion
# There are more redundant variables, that are low predictive features,
# Age seems to determine a lot of relations, Edu, JLevel, MIncome, YWC...
# PerformanceRatings greatly determines whether you get a %SalaryHike
# With higher JobLevel the higher the benefits...

##Moving Forward
# Convert Attrition to Float for further correlation,
# Or Use other methods,
# Feature selection would be very suitable for Prediction Analysis