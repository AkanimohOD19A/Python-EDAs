import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#pd.set_option("display.float_format","{:.2f}".format)

#pen_Df = pd.read_csv("C:/Users/HP/Documents/EDA/Python/fifa-world-cup/penalties.csv")
matches_Df = pd.read_csv("C:/Users/HP/Documents/EDA/Python/fifa-world-cup/WorldCupMatches.csv")
players_Df = pd.read_csv("C:/Users/HP/Documents/EDA/Python/fifa-world-cup/WorldCupPlayers.csv")
wc_Df      = pd.read_csv("C:/Users/HP/Documents/EDA/Python/fifa-world-cup/WorldCups.csv")

print(matches_Df.info())

drp = ["RoundID", "MatchID", "Position", "Event"]
players_Df.drop(columns = drp, inplace = True)
print(players_Df.T)
print(players_Df["Line-up"].value_counts())

print(wc_Df.tail().T, '\n', wc_Df.info())
# wc_Df is much of a pivot_table than a data set.
# players_Df is a small data set, and is not really relevant.
# matches_Df seems to be the only large and valid data to use, now we need to clean it.

# Initiating Checks

matches_Df = matches_Df.loc[:851]
print(matches_Df[matches_Df["Attendance"].isnull()].index.to_list()) #Prints the index where there are null
matches_Df["Attendance"].fillna(matches_Df["Attendance"].mean(), inplace = True) #Fill them with the mean
# or 'matches_Df.fillna(matches_Df.mean(), inplace =True)' since there is only one null var

# Some country names have errors
matches_Df['Home Team Name'] = matches_Df['Home Team Name'].apply(lambda cells : cells.strip('rn">'))

# Converting floats to integers, so we avoid 1.0 goals of year 2014.0
for col in matches_Df.columns:
    if matches_Df[col].dtype != object:
        matches_Df[col] = matches_Df[col].astype(int)

# We can even create an Second - half Goals
matches_Df["Second-half Home Goals"] = matches_Df["Home Team Goals"] - matches_Df["Half-time Home Goals"]
matches_Df["Second-half Away Goals"] = matches_Df["Away Team Goals"] - matches_Df["Half-time Away Goals"]
# matched_Df.insert(loc = len(matches_Df),
# column = ['Second-half Home Goals', 'Second-half Away Goals'], value = [Second-half Home Goals, Second-half Away Goals] )

print(matches_Df.info())
# 3. A cleaner and more concise data set, now we need to save it

matches_Df.to_csv("C:/Users/HP/Documents/EDA/Python/fifa-world-cup/clean_WorldCupMatches.csv",
                  encoding = "utf-8", index =False)

# 2. Build Data Profiles

#Moving forward we really don't need some vars
drp = ["Datetime", "Win conditions", "RoundID", "Referee", "Assistant 1", "Assistant 2", "MatchID"]
matches_Df.drop(columns = drp, inplace = True)

# 1. Histogram
hist = matches_Df.hist(bins = 10, figsize = (20, 10))
plt.show()
# Attendance is usually between 2500 and 70,000, we should expect very high outliers in the next plots
# Goals scored favour the Home teams, whether before of after the second half
# No games in '42, '46,
# Matches started to compound post 1980, the most recent year 2014(80) has the bulk of games so far.

# 2. Boxplots
drp = ['Year','Attendance']
bplot = matches_Df.drop(columns = drp).boxplot(figsize = (20, 10))
plt.show()
# Year is exempted because of the expected convergence at 2000, attendance pales the outliers from other vars
# It seems there is always an Home Advantage,
# Particularly in total goals, the Half time stats is similar for both ends

# 3. Pairplots
#pp = sns.pairplot(matches_Df)
#plt.show()

## 2. Measuring Vars
# 1. Categorical Variables
print()
for col in matches_Df.columns:
    if matches_Df[col].dtypes == object and matches_Df[col].nunique() < 10:
        print()
        print(f"{matches_Df[col].value_counts()}")
# None of the vars prove to be concise descriptions of events

# 2. Numerical/Continuous Variables
print()
for col in matches_Df.columns:
    if matches_Df[col].dtypes != object and matches_Df[col].nunique() < 25:
        print()
        print(f"{col}: Min_{matches_Df[col].min()} and Max_{matches_Df[col].max()}")

## 3. CrossTabs and Pivot Tables
# 1. CrossTbas: Distribution of Countries by Stage
S_Home_Country = pd.crosstab(matches_Df['Stage'], matches_Df['Home Team Name'])
print(S_Home_Country)
S_Away_Country = pd.crosstab(matches_Df['Stage'], matches_Df['Home Team Name'])
print(S_Home_Country)

# 2. Pivot Table: Distribution of Goals by Stage
S_Home_Goals = matches_Df.pivot_table(['Half-time Home Goals', 'Second-half Home Goals'], ['Stage'])
print(S_Home_Goals, '\n')
S_Away_Goals = matches_Df.pivot_table(['Half-time Away Goals', 'Second-half Away Goals'], ['Stage'])
print(S_Away_Goals, '\n')

## 4. Correlation
matches_corr = matches_Df.corr()
print(matches_corr.abs())

# 4b. Heatmap and Masks
mask = np.zeros_like(matches_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(matches_corr, mask = mask, center = 0, square=True, vmin= -1.2, vmax = 1.2,
            cbar_kws={"shrink": .5}, cmap="YlGnBu", linewidths=.5, annot = True)
plt.show()

# Second Half Goals then to influence Goals than 1/2 time goals
# Attendance gels positively with the Year, re-interating our first assumption from previous charts

# Correlation is enough, perhaps creating pivot tables would be,
# remember that even one of the read data set is one- summary table