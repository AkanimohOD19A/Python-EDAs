import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", "{:.2f}".format)

stdt_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/student-mat.csv")
student = stdt_Df.copy()

print(stdt_Df.info())
print(stdt_Df.head().T)

#Initiating Data Checks and Cleaning
print(stdt_Df.isnull().sum()) # No NaNs

stdt_Df["aGrades"] = (stdt_Df["G1"] + stdt_Df["G2"] + stdt_Df["G3"])/3
stdt_Df["ParentEdu"] = (stdt_Df['Medu'] + stdt_Df['Fedu'])/2
stdt_Df['ParentEdu'] = stdt_Df["ParentEdu"].astype(int)
drp = ["G1",'G2', "G3", "Medu", "Fedu"]
stdt_Df.drop(columns = drp, inplace = True)
print(stdt_Df["ParentEdu"])
#stdt_Df.to_csv("C:/Users/HP/Desktop/EDAs/student-mat.csv", encoding='utf-8', index=True)

## Building Data Tables and Profiles
# 1. Histogram-Numerical Variables
stdt_hist = stdt_Df.hist(bins = 10, figsize = (20, 10))
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/student_mat Histogram.png")
plt.show()

# Parents usually have an average of stage 2&3 level of Edu.
# Most average Grades are between 10 and 15
# Age displays dubious chart, save the bins every other thing is cool... alot don't fail.
# student have reported a lot of free time but few travel time*

# 2. Boxplot
plt.figure(figsize = (40, 10))

num_cols = stdt_Df.select_dtypes(include = "int64").columns # adjust with value_counts*
for i,col in enumerate(num_cols):
    plt.subplot(5, 4, i+1)
    sns.boxplot(stdt_Df[col], color = 'xkcd:lime')
    plt.title(f"{col}", fontsize = 10)
    plt.xlabel(' ')
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/student_mat Boxplot.png")
plt.show()

# 3. PairPlots
stdt_pp = sns.pairplot(stdt_Df)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/student_mat Pairplot.png")
plt.show()

## Variable Measurement
# 1. Categorical Variables
print()
print('CAT Variables')
print()
n_cat = []
non_col = []
for col in stdt_Df.columns:
    if stdt_Df[col].dtypes == object and stdt_Df[col].nunique() < 15:
        n_cat.append(col)
        print('=='*7)
        print(f"{stdt_Df[col].value_counts()}")
    elif stdt_Df[col].dtypes == object: # Usually objects skip set parameters
        non_col.append(col)
print(non_col)
print(f'no. of CAT Variables: {len(n_cat)}')

# 2. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in stdt_Df.columns:
    if stdt_Df[col].dtypes != object and stdt_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{stdt_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Continuous Variables')
n_cont = []
for col in stdt_Df.columns:
    if stdt_Df[col].dtypes != object and stdt_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{stdt_Df[col].min()} and Max_{stdt_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(stdt_Df.columns) - (len(n_cat) + len(n_num) + len(n_cont)))} columns")

# Exploring Data Relationships
stdt_corr = stdt_Df.corr()
print(stdt_corr.abs())

mask = np.zeros_like(stdt_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize = (20, 10))
sns.heatmap(stdt_corr, mask = mask, center =0, vmin =-1.2, vmax = 1.2,
            linewidths= .1, cbar_kws = {'shrink': .5}, cmap = 'coolwarm',
            square =False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/student_mat Heatmap.png")
plt.show()
# A number of obs here, studytime counteract (traveltime [-0.13]), Age correlates negatively
# with Average Grades and correlates even better +ve with failures[.24].
# I find the relationship btw failures and Grades acceptable but quite intriguing,
# because can we really acknowledge this given it is mere consequential variable.

# Conclusion
# This analysis was geared to exploring relationship, especially between on Grades, there are lot of
# plausible relationship +ve and -ve, none proved to be a very strong influence, however this is expected
# in most social science

## Moving Forward
# It would be cool, to mix variables into new vars, to further explore influences, for instance
# if we can create ParentJob from the the two parent Job...notice how Pjob relates to 'travel time'