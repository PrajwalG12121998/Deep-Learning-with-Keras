# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:58:08 2020

@author: PRAJWAL
"""

# Import the data
import pandas as pd
df = pd.read_csv("pacific_hurricanes.csv")
df.head() 

df['hurricane'].value_counts()

df['hurricane'].value_counts(normalize=True).loc[0]
