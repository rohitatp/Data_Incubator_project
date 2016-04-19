# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 21:33:52 2016

@author: raggarwa
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numdifftools as nd

train = pd.read_table("data.txt",sep=',')
train["time_hrs"] = (train["time"]-train["time"].loc[0])/3600

# Visual Insights

plot_xval = train["power"]
plot_yval = train["time_hrs"]
plt.plot(plot_yval,plot_xval)
plt.title("Time plot of power consumed")
plt.xlabel("Time (hours)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

plot_xval = train["power"].loc[3600*24:3600*48]
plot_yval = train["time_hrs"].loc[3600*24:3600*48]
plt.plot(plot_yval,plot_xval)
plt.title("Time plot of power consumed on 2nd day")
plt.xlabel("Time (hours)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

num_bins = 100
plt.hist(train["power"],num_bins)
plt.title("Histogram of power consumed")
plt.xlabel("Power consumed (Watts)")
plt.ylabel("Frequency")
plt.yscale('log')
plt.grid()
plt.show()

num_bins = 100
plt.hist(train["power"],num_bins)
plt.title("Histogram of power consumed")
plt.xlabel("Power consumed (Watts)")
plt.ylabel("Frequency")
plt.grid()
plt.show()

counts, bin_edges = np.histogram(train["power"], bins=num_bins)
cdf = np.cumsum(counts)/np.cumsum(counts).max()
plt.grid()
plt.plot(bin_edges[1:], cdf)
plt.title("CDF of power consumed")
plt.xlabel("Power consumed (Watts)")
plt.ylabel("CDF")
plt.show()

first_derivative = np.gradient(cdf)
plt.grid()
plt.plot(bin_edges[1:], first_derivative)
plt.title("PDF of power consumed")
plt.xlabel("Power consumed (Watts)")
plt.ylabel("PDF")
plt.xlim(0, 10000)
plt.show()

second_derivative = np.gradient(first_derivative)
plt.grid()
plt.plot(bin_edges[1:], second_derivative)
plt.title("Derivative of PDF of power consumed")
plt.xlabel("Power consumed (Watts)")
plt.ylabel("PDF")
plt.xlim(0, 10000)
plt.show()

print("PEAKS:")
for i in range(99):
    if((second_derivative[i]>0) & (second_derivative[i+1] <=0)):
        slope = (second_derivative[i+1] - second_derivative[i])/(bin_edges[1+i+1]-bin_edges[1+i])
        print(-second_derivative[i]/slope + bin_edges[1+i])

print("TROUGHS:")
for i in range(99):
    if((second_derivative[i]<0) & (second_derivative[i+1] >=0)):
        slope = (second_derivative[i+1] - second_derivative[i])/(bin_edges[1+i+1]-bin_edges[1+i])
        print(-second_derivative[i]/slope + bin_edges[1+i])
        
# Isolate pool pump by taking the minimum over each second for the 30 day period.
train["daily_sec"]= (train["time"]-train["time"].loc[0])%(3600*24)
min_daily_series = train["power"].iloc[:3600*24]
max_daily_series = train["time"].iloc[:3600*24]
for i in range(3600*24):
    min_daily_series[i] = 50000 # High number
    max_daily_series[i] = 0 # Low number

#for i in range(1,30):
#    print(i)
#    min_daily_series = np.minimum(min_daily_series,train["power"].iloc[i*3600*24:(i+1)*3600*24])

for i in range(len(train["power"])):
    index = train["daily_sec"].iloc[i]
    min_daily_series[index] = np.minimum(min_daily_series[index],train["power"].iloc[i])
    
for i in range(len(train["power"])):
    index = train["daily_sec"].iloc[i]
    max_daily_series[index] = np.maximum(max_daily_series[index],train["power"].iloc[i])

plt.plot( range(0,3600*24),min_daily_series)
plt.title("Min power consumed over 30 days for each second")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

plt.plot(range(0,3600*24),max_daily_series)
plt.title("Max power consumed over 30 days for each second")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

# Time when water pump in ON on all days = 5398 to 15139 seconds

train["water_pump"] = 0
#For each day, find water cooler start instant and mark it
max_num_days = np.ceil(train["time"].loc[len(train["time"])]/3600/24)
water_pump_final_time = 0
water_pump_start_index = 0
water_pump_start_time = 0
for i in range(len(train["time"])-1):
    found_pump = False
    if((train["power"][i] > 1500) & (train["power"][i+1] > 1500)):
        if(!found_pump):
            water_pump_start_time = train["time"][i] # Actual time 
            water_pump_start_index = i
        water_pump_final_time = i
        found_pump = True
    else:
        if(found_pump & ((water_pump_final_time - water_pump_start_time) < (15139 - 5398))): # Invalid case
            found_pump = False
            water_pump_total_time = 0
            water_pump_start_time = 0
            water_pump_start_index = 0
        elif (found_pump & ((water_pump_final_time - water_pump_start_time) >= (15139 - 5398))): # Valid case
            train["water_pump"].iloc(water_pump_start_index:i) = 1
            water_pump_total_time = 0
            water_pump_start_time = 0
            water_pump_start_index = 0           
            found_pump = False
            
            
    