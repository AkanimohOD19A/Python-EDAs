import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

click_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/clikstream_data.csv")
print(click_Df.info())
print(click_Df.head()) # No NaNs

## Initiating Data Cleaning and Checks
click_Df.drop(columns = ['user_id'], inplace = True)

## Build Data Tables and Profiles
# 1. Histogram
hist = click_Df.hist(bins =10, figsize= (20, 10) )
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/clicks histogram.png")
plt.show()

# If what we are dealing with is secs, then few have the patience for it to load, play or even seek the video

# 2. Boxplots
plt.figure(figsize=(20, 10))

num_cols = click_Df.select_dtypes(include = 'float')
for i, col in enumerate(num_cols):
    plt.subplot(5, 2, i+1)
    sns.boxplot(click_Df[col], color = "xkcd:lime")
    plt.title(f"{col}", fontsize = 10)
    plt.xlabel(' ')
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/clicks Boxplot.png")
plt.show()
# Every col seem to be skewed to the left towards 0.

# 3. Pairplots
clicks_pp = sns.pairplot(click_Df)
plt.savefig("C:/Users/HP/PycharmPRojects/Complete_EDA/clicks heatmap.png")
plt.show()

# From this pairplot, I've a feeling of positive relationship between playing, pausing and seeking video.

## Variable Measurement

# 1. Numerical Variables
print()
print('Num Variables')
n_num = []
for col in click_Df.columns:
    if click_Df[col].dtypes != object and click_Df[col].nunique() < 20:
        n_num.append(col)
        print('=+='*7)
        print(f"{click_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

# 3. Continuous Variables
print()
print('Cont and Num Variables')
n_cont = []
for col in click_Df.columns:
    if click_Df[col].dtypes != object and click_Df[col].nunique() >=20:
        n_cont.append(col)
        print("+=+"*7)
        print(f"{col}: Min_{click_Df[col].min()} and Max_{click_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {(len(click_Df.columns) - (len(n_num) + len(n_cont)))} columns")

## Explore Data Relationships
click_corr = click_Df.corr()
print(click_corr.abs())
plt.figure(figsize = (20, 10))

mask = np.zeros_like(click_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(click_corr, mask = mask, vmin=-1.2, vmax=1.2, center = 0,
         linewidth = .1, cbar_kws = {"shrink":.5}, cmap="coolwarm",
         square = False, annot = True)
plt.savefig("C:/Users/HP/PycharmProjects/Complete_EDA/clicks Heatmap.png")
plt.show()

# load_video only correlates negatively with speed_change_video, my intuition was quite correct
# however seeking correlates better with playing video

## Conclusion
# This data is quite staright-forward, as it confirms first intuition of what relationships should
# look like, however the data description should be visited to understand its true nature.

## Moving Forward
# So, far we've done analysis leading to understanding basic relationships, it would be better to produce
# models, start with linear regression model, then any other deeper model that is appropriate.