import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

curr_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/currency.csv")
print(curr_Df.info())
print(curr_Df.head())

curr_Df['Time'] = curr_Df['Time'].astype(float)