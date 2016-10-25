
# coding: utf-8

# # Practical Work Week #1. Map visualization and US data analysis
# **Authors**: Enrique Herreros and Vincent Stangenberger
# 
# **Date**: 7th of September, 2016
# 
# **Course**: Fundamentals of Data Science

# ## Tasks for this assignment
# 
# Improve the map by:
#  
#  - Scaling to a smaller size and translate to the bottom Alaska
#  - Scaling to a bigger size and translate to the top Hawaii
#  - Colouring the States according to gathered data from Internet
#  - Creating a legend (color bar with the values)
#  - Labeling the names of the States

# **References:**
# 
# *http://kff.org/other/state-indicator/*
# 

# ## Map data manipualtion and visualization

# In[1]:



import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from descartes import PolygonPatch
from shapely.geometry import Polygon, MultiPolygon, shape
import shapely.affinity # <- test this for scaling
import matplotlib.colors as colors
import matplotlib.cm as cmx
import csv
from  matplotlib.pyplot import colorbar

# for juppyter notebooks, it will draw the plots on the notebook itself instead of creating popup windows with the figures
# get_ipython().magic(u'matplotlib inline')


# In[2]:

# Initialize all the variables
S_DIR = 'C:/Users/vincent/Documents/UvA/DataScience/Fundamentals of DS'
BLUE = '#5599ff'
RED = '#ff0000'
total_hospitals = {} # dictionary with data of number of hospitals by state in 2014
total_births = {} # dictionary with data of number of births by state in 2014
deathrate = {} # dictionary with data of death rate per 100.000 by state in 2014
max_total_hospitals = 0
max_total_births = 0
max_total_deaths = 0

# Load the datasets
with open(os.path.join(S_DIR, 'Data/total_hospitals.csv'), 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        total_hospitals[row['name_state']] = row['total_hospitals']
        if int(row['total_hospitals']) > max_total_hospitals and row['total_hospitals'] != None:
            max_total_hospitals = int(row['total_hospitals'])

with open(os.path.join(S_DIR, 'Data/total_births.csv'), 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        total_births[row['name_state']] = row['total_births']
        if int(row['total_births']) > max_total_births and row['total_births'] != None:
            max_total_births = int(row['total_births'])
            
with open(os.path.join(S_DIR, 'Data/deathrate.csv'), 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        deathrate[row['name_state']] = row["death_rate"]
        if float(row['death_rate']) > max_total_deaths and row['death_rate'] != None:
            max_total_deaths = float(row['death_rate'])

# Load geojson map file as json
with open(os.path.join(S_DIR, 'states.geojson')) as rf:
    data = json.load(rf)

# Function for plotting the US map
def plot_map(maximum_value, values, title):
    fig = plt.figure(figsize=(15, 8), dpi=100)
    ax = fig.gca()
    
    cm_ = plt.get_cmap('BuGn') # Name of colormap we want to use
    # Careful with normalizing, it will not automatically calculate percentages, it will only paint absolute values
    cNorm  = colors.Normalize(vmin=0, vmax=maximum_value) # Range of values expected in order to be able to normalize any incoming data
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm_) # Create the scalarMap itself

    for feature in data['features']:
        geometry = feature['geometry']
        s = shape(geometry)
        state_name = feature['properties'][u'STATE_NAME'] # state name
        state_fips = feature['properties'][u'STATE_FIPS'] # example of data to be used
        colorVal = scalarMap.to_rgba(values[state_name])
        if state_name == 'Hawaii': # make it bigger and move it to the right
            s = shapely.affinity.scale(s,xfact=5, yfact=5 )
            s = shapely.affinity.translate(s, xoff=20)
        elif state_name == 'Alaska': # make it a little bit smaller and move it down
            s = shapely.affinity.scale(s,xfact=0.7, yfact=0.7 )
            s = shapely.affinity.translate(s, yoff=-10)
        # print(state_name + " - " + total_hospitals[state_name])
        if geometry['type'] == 'Polygon':
            ax.add_patch(PolygonPatch(s, fc=colorVal,ec=BLUE, alpha=0.5, zorder=2))
        else:
            for g in s.geoms:
                ax.add_patch(PolygonPatch(g,fc=colorVal, ec=BLUE, alpha=0.5, zorder=2))
        plt.text(s.centroid.x, s.centroid.y,state_name, fontsize=6)
                
    ax.axis('scaled')
    plt.axis('off')
    plt.title(title)

    cmmapable = cmx.ScalarMappable(cNorm, cm_)
    cmmapable.set_array(range(0,maximum_value))
    plt.colorbar(cmmapable)

    #plt.title
    plt.show()


# ### Plot Total Hospitals in the US map

# In[3]:

plot_map(max_total_hospitals/20, total_hospitals, "Total Hospitals in the US")


# ### Plot Total Births in the US map

# In[4]:

plot_map(max_total_births/20, total_births, "Total Births in the US")


# ### Plot the ratio Births/Hospitals in the US map

# In[5]:

plot_map(max_total_births/max_total_hospitals,
         {k: float(total_births[k])/float(total_hospitals[k]) for k in total_births.viewkeys() & total_hospitals.viewkeys()},
         "Ratio Births/Hospitals in the US")


# ### Plot Death Rate in the US map

# In[6]:

plot_map(int(max_total_deaths), deathrate, "Death rate in the US")


# ## Data correlation and heatmap

# In[7]:

import numpy as np
from pandas import DataFrame, read_csv
import pandas as pd

df_deathrate = read_csv(S_DIR + '\\Data\\deathrate.csv',index_col="name_state")
df_total_hospitals = read_csv(S_DIR + '\\Data\\total_hospitals.csv',index_col="name_state")
df_total_births = read_csv(S_DIR + '\\Data\\total_births.csv',index_col="name_state")

df_needing_not_receiving_treatmen_alcohol = read_csv(S_DIR + '\\Data\\needing_not_receiving_treatment_alcohol.csv',index_col="name_state")
df_needing_not_receiving_treatmen_illicit_drugs = read_csv(S_DIR + '\\Data\\needing_not_receiving_treatment_illicit_drugs.csv',index_col="name_state")
df_percent_adults_cardiovascular_disease = read_csv(S_DIR + '\\Data\\percent_adults_cardiovascular_disease.csv',index_col="name_state")
df_percent_adults_obese = read_csv(S_DIR + '\\Data\\percent_adults_obese.csv',index_col="name_state")
df_percent_adults_poor_health = read_csv(S_DIR + '\\Data\\percent_adults_poor_health.csv',index_col="name_state")
df_suicide_rate = read_csv(S_DIR + '\\Data\\suicide_rate.csv',index_col="name_state")
df_total_cancer_deaths = read_csv(S_DIR + '\\Data\\total_cancer_deaths.csv',index_col="name_state")
df_total_deaths_firearms = read_csv(S_DIR + '\\Data\\total_deaths_firearms.csv',index_col="name_state")
df_total_hospitals = read_csv(S_DIR + '\\Data\\total_hospitals.csv',index_col="name_state")
df_total_infant_deaths = read_csv(S_DIR + '\\Data\\total_infant_deaths.csv',index_col="name_state")


# In[8]:

df_total_hospitals.index


# In[18]:

df_joined = pd.concat([df_total_births, df_total_hospitals, df_deathrate,df_needing_not_receiving_treatmen_alcohol,
                     df_needing_not_receiving_treatmen_illicit_drugs ,df_percent_adults_cardiovascular_disease,
                     df_percent_adults_obese ,df_percent_adults_poor_health ,df_suicide_rate ,df_total_cancer_deaths,
                     df_total_deaths_firearms], axis=1, join='inner')


# In[23]:

full_df = df_joined.rename(columns={'total_births':'Total births',
                                 'total_hospitals':'Total hospitals', 
                                 'death_rate':'Death rate'})
full_df = full_df.corr()

# Plot it out
fig, ax = plt.subplots()
heatmap = ax.pcolor(full_df, cmap=plt.cm.jet, alpha=0.7)

# Format
fig = plt.gcf()
fig.set_size_inches(8, 11)

# turn off the frame
ax.set_frame_on(False)

# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(full_df.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(full_df.shape[1]) + 0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

# Set the labels

# label source:https://en.wikipedia.org/wiki/Basketball_statistics
labels=["Total births","Total hospitals","Death rate",
            "Teens Ages 12-17 Reporting Needing but Not Recieving Treatment for Alcohol Use",
            "Adults Ages 18+ Reporting Needing but Not Recieving Treatment for Alcohol Use",
            "Teens Needing but Not Recieving Treatment for Illicit Drug Use",
            "Adults Needing but Not Recieving Treatment for Illicit Drug Use",
            "Has Cardiovascular Disease","Adult Overweight/Obesity Rate","Adults Reporting Fair or Poor Health Status",
            "Suicide Rate per 100,000 Individuals","Cancer Death Rate per 100,000","Firearms Death Rate per 100,000"]

# note I could have used nba_sort.columns but made "labels" instead
ax.set_xticklabels(labels, minor=False)
ax.set_yticklabels(full_df.index, minor=False)

# rotate the
plt.xticks(rotation=90)

ax.grid(False)

# Turn off all the ticks
ax = plt.gca()

for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False


# ### Plot with stacked bar Percent of Adults Reporting Poor Mental Health Status, by Race/Ethnicity

# In[14]:

import numpy as np
from pandas import DataFrame, read_csv
import pandas as pd

df_percent_adults_poor_mental_health_by_race = read_csv(S_DIR + '\\Data\\percent_adults_poor_mental_health_by_race.csv')
df_percent_adults_poor_mental_health_by_race = df_percent_adults_poor_mental_health_by_race.replace("NSD", "0")


# In[15]:

# Create the general blog and the "subplots" i.e. the bars
f, ax1 = plt.subplots(1, figsize=(13,7))

# Set the bar width
bar_width = 0.75

# positions of the left bar-boundaries
bar_l = [i+1 for i in range(len(df_percent_adults_poor_mental_health_by_race['White']))] 

# positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i+(bar_width/2) for i in bar_l] 

# Create a bar plot, in position bar_1
ax1.bar(bar_l, 
        # using the pre_score data
        df_percent_adults_poor_mental_health_by_race['White'], 
        # set the width
        width=bar_width,
        # with the label pre score
        label='White', 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#F4561D')

# Create a bar plot, in position bar_1
ax1.bar(bar_l, 
        # using the mid_score data
        df_percent_adults_poor_mental_health_by_race['Black'], 
        # set the width
        width=bar_width,
        # with pre_score on the bottom
        bottom=df_percent_adults_poor_mental_health_by_race['White'], 
        # with the label mid score
        label='Black', 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#F1911E')

# Create a bar plot, in position bar_1
ax1.bar(bar_l, 
        # using the post_score data
        df_percent_adults_poor_mental_health_by_race['Hispanic'], 
        # set the width
        width=bar_width,
        # with pre_score and mid_score on the bottom
        bottom=[float(i)+float(j) for i,j in zip(df_percent_adults_poor_mental_health_by_race['White'],df_percent_adults_poor_mental_health_by_race['Black'])], 
        # with the label post score
        label='Hispanic', 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#F1BD1A')

# Create a bar plot, in position bar_1
ax1.bar(bar_l, 
        # using the post_score data
        df_percent_adults_poor_mental_health_by_race['Asian/Native Hawaiian and Pacific Islander'], 
        # set the width
        width=bar_width,
        # with pre_score and mid_score on the bottom
        bottom=[float(i)+float(j)+float(k) for i,j,k in zip(df_percent_adults_poor_mental_health_by_race['White'],
                                                df_percent_adults_poor_mental_health_by_race['Black'],
                                                df_percent_adults_poor_mental_health_by_race['Hispanic'])], 
        # with the label post score
        label='Asian', 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#AA11DD')

# Create a bar plot, in position bar_1
ax1.bar(bar_l, 
        # using the post_score data
        df_percent_adults_poor_mental_health_by_race['American Indian/Alaska Native'], 
        # set the width
        width=bar_width,
        # with pre_score and mid_score on the bottom
        bottom=[float(i)+float(j)+float(k)+float(l) for i,j,k,l in zip(df_percent_adults_poor_mental_health_by_race['White'],
                                                df_percent_adults_poor_mental_health_by_race['Black'],
                                                df_percent_adults_poor_mental_health_by_race['Hispanic'],
                                                df_percent_adults_poor_mental_health_by_race['Asian/Native Hawaiian and Pacific Islander'])], 
        # with the label post score
        label='American Indian', 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#A1BD1A')

# Create a bar plot, in position bar_1
ax1.bar(bar_l, 
        # using the post_score data
        df_percent_adults_poor_mental_health_by_race['Other'], 
        # set the width
        width=bar_width,
        # with pre_score and mid_score on the bottom
        bottom=[float(i)+float(j)+float(k)+float(l)+float(m) for i,j,k,l,m in zip(df_percent_adults_poor_mental_health_by_race['White'],
                                                df_percent_adults_poor_mental_health_by_race['Black'],
                                                df_percent_adults_poor_mental_health_by_race['Hispanic'],
                                                df_percent_adults_poor_mental_health_by_race['American Indian/Alaska Native'],
                                                df_percent_adults_poor_mental_health_by_race['Asian/Native Hawaiian and Pacific Islander'])], 
                                                
        # with the label post score
        label='Other', 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#FFFD1A')

# set the x ticks with names
plt.xticks(tick_pos, df_percent_adults_poor_mental_health_by_race['name_state'])

# Set the label and legends
ax1.set_ylabel("Total Score")
ax1.set_xlabel("Percent of Adults Reporting Poor Mental Health Status, by Race/Ethnicity")
plt.legend(loc='upper right', bbox_to_anchor=(1.22, 1.01))

locs, labels = plt.xticks()
plt.setp(labels, rotation=90)

# Set a buffer around the edge
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])


# ### Plot with stacked bar Overweight and Obesity Rates for Adults by Race/Ethnicity

# In[16]:

# load data
df_percent_adults_overwight_by_race = read_csv(S_DIR + '\\Data\\percent_overweight_by_race.csv')
df_percent_adults_overwight_by_race = df_percent_adults_overwight_by_race.replace("NSD", "0")


# In[17]:

# Create the general blog and the "subplots" i.e. the bars
f, ax1 = plt.subplots(1, figsize=(13,7))

# Set the bar width
bar_width = 0.75

# positions of the left bar-boundaries
bar_l = [i+1 for i in range(len(df_percent_adults_overwight_by_race['White']))] 

# positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i+(bar_width/2) for i in bar_l] 

# Create a bar plot, in position bar_1
ax1.bar(bar_l, 
        # using the pre_score data
        df_percent_adults_overwight_by_race['White'], 
        # set the width
        width=bar_width,
        # with the label pre score
        label='White', 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#F4561D')

# Create a bar plot, in position bar_1
ax1.bar(bar_l, 
        # using the mid_score data
        df_percent_adults_overwight_by_race['Black'], 
        # set the width
        width=bar_width,
        # with pre_score on the bottom
        bottom=df_percent_adults_overwight_by_race['White'], 
        # with the label mid score
        label='Black', 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#F1911E')

# Create a bar plot, in position bar_1
ax1.bar(bar_l, 
        # using the post_score data
        df_percent_adults_overwight_by_race['Hispanic'], 
        # set the width
        width=bar_width,
        # with pre_score and mid_score on the bottom
        bottom=[float(i)+float(j) for i,j in zip(df_percent_adults_overwight_by_race['White'],
                                                 df_percent_adults_overwight_by_race['Black'])], 
        # with the label post score
        label='Hispanic', 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#F1BD1A')

# Create a bar plot, in position bar_1
ax1.bar(bar_l, 
        # using the post_score data
        df_percent_adults_overwight_by_race['Asian/Native Hawaiian and Pacific Islander'], 
        # set the width
        width=bar_width,
        # with pre_score and mid_score on the bottom
        bottom=[float(i)+float(j)+float(k) for i,j,k in zip(df_percent_adults_overwight_by_race['White'],
                                                df_percent_adults_overwight_by_race['Black'],
                                                df_percent_adults_overwight_by_race['Hispanic'])], 
        # with the label post score
        label='Asian', 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#AA11DD')

# Create a bar plot, in position bar_1
ax1.bar(bar_l, 
        # using the post_score data
        df_percent_adults_overwight_by_race['American Indian/Alaska Native'], 
        # set the width
        width=bar_width,
        # with pre_score and mid_score on the bottom
        bottom=[float(i)+float(j)+float(k)+float(l) for i,j,k,l in zip(df_percent_adults_overwight_by_race['White'],
                                                df_percent_adults_overwight_by_race['Black'],
                                                df_percent_adults_overwight_by_race['Hispanic'],
                                                df_percent_adults_overwight_by_race['Asian/Native Hawaiian and Pacific Islander'])], 
        # with the label post score
        label='American Indian', 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#A1BD1A')

# Create a bar plot, in position bar_1
ax1.bar(bar_l, 
        # using the post_score data
        df_percent_adults_overwight_by_race['Other'], 
        # set the width
        width=bar_width,
        # with pre_score and mid_score on the bottom
        bottom=[float(i)+float(j)+float(k)+float(l)+float(m) for i,j,k,l,m in zip(df_percent_adults_overwight_by_race['White'],
                                                df_percent_adults_overwight_by_race['Black'],
                                                df_percent_adults_overwight_by_race['Hispanic'],
                                                df_percent_adults_overwight_by_race['American Indian/Alaska Native'],
                                                df_percent_adults_overwight_by_race['Asian/Native Hawaiian and Pacific Islander'])], 
                                                
        # with the label post score
        label='Other', 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#FFFD1A')

# set the x ticks with names
plt.xticks(tick_pos, df_percent_adults_overwight_by_race['name_state'])

# Set the label and legends
ax1.set_ylabel("Total Score")
ax1.set_xlabel("Percent of Adults with Overweight and Obesity Rates for Adults by Race/Ethnicity")
plt.legend(loc='upper right', bbox_to_anchor=(1.22, 1.01))

locs, labels = plt.xticks()
plt.setp(labels, rotation=90)

# Set a buffer around the edge
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])

