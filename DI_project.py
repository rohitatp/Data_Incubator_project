# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 21:33:52 2016

@author: raggarwa
"""
#!/usr/bin/python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numdifftools as nd

    # Find all instants in the dataframe with power >= power_appliance for time >= period, and remove power_appliance from the dataframe
    #array_power = numpy.asarray(dataframe["power"])
    #array_power = 1
    #low_values_indices = array_power < power_appliance  # Where values are low
    #array_power[low_values_indices] = 0  # All low values set to 0
    #array_power = np.asarray(dataframe["power"])*0
    #array_power[dataframe["power"] >= power_appliance] = 1  # All low values set to 0
def extract_and_remove(dataframe, power_appliance, period, string_to_add):
    array_valid_power = dataframe["power"] *0
    high_values_indices = (dataframe["power"] >= power_appliance)  # Where values are low
    array_valid_power[high_values_indices] = 1
    last_index_ER = 0
    first_index_ER = 0
    dataframe[string_to_add] = dataframe["power"] *0
    while((first_index_ER<=(len(array_valid_power)-period)) & (period>1)):
        if(min(array_valid_power[first_index_ER:first_index_ER+period-1]) == 1):
            # This is a valid case. Find the last index when array_valid_power = 1 after index
            #print("Lucky: Found occurrence")
            last_index_ER = np.argmax(array_valid_power[first_index_ER+1:] == 0, axis=0)-1
            #print("Data frame check:",dataframe["power"][10000])
            #print("First Index = ",first_index_ER)
            #print("Last index = ",last_index_ER)
            first_index_ER = int(first_index_ER)
            last_index_ER = int(last_index_ER)
            dataframe["power"].loc[first_index_ER:last_index_ER] = dataframe["power"].loc[first_index_ER:last_index_ER]  - power_appliance
            dataframe[string_to_add].loc[first_index_ER:last_index_ER] = 1
            #print("Data frame after check:",dataframe["power"][10000])
            first_index_ER = max(first_index_ER+1,last_index_ER)
        else:
            # Find the last occurrence of 0 in first_index:first_index+period-1 and go to the next element
            #print("first_index = ",first_index_ER)
            first_index_ER = first_index_ER + max(max(np.where(array_valid_power[first_index_ER:first_index_ER+period-1] == 0)))+1
            #print("No luck: First_index = ",first_index_ER)
    if(period==1):
        dataframe["power"].loc[high_values_indices] = dataframe["power"].loc[high_values_indices]  - power_appliance
        dataframe[string_to_add].loc[high_values_indices] = 1
    return dataframe

df = pd.read_table("data.txt",sep=',')
df["time_offset"] = df["time"] - df["time"].iloc[0]
train = df.merge(how='right', on='time_offset', right = pd.DataFrame({'time_offset':np.arange(df.iloc[0]['time_offset'], df.iloc[-1]['time_offset'] + 1, 1)})).sort(columns='time_offset').reset_index().drop(['index'], axis=1)
train = train.fillna(method='pad')
train["time_hrs"] = (train["time"]-train["time"].loc[0])/3600
# Visual Insights
num_days = (train["time_offset"].loc[len( train["time"])-1]+1)/86400 # Since index starts at 0
num_days = num_days.astype(int)
print('Number of days = ', num_days)

plot_xval = train["power"]
plot_yval = train["time_hrs"]
plt.plot(plot_yval,plot_xval)
plt.title("Time plot of power consumed")
plt.xlabel("Time (hours)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

plot_xval = train["power"].loc[:3600*24]
plot_yval = train["time_hrs"].loc[:3600*24]
plt.plot(plot_yval,plot_xval)
plt.title("Time plot of power consumed on 1st day")
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
for i in range(num_bins-1):
    if((second_derivative[i]>0) & (second_derivative[i+1] <=0)):
        slope = (second_derivative[i+1] - second_derivative[i])/(bin_edges[1+i+1]-bin_edges[1+i])
        print(-second_derivative[i]/slope + bin_edges[1+i])

print("TROUGHS:")
for i in range(num_bins-1):
    if((second_derivative[i]<0) & (second_derivative[i+1] >=0)):
        slope = (second_derivative[i+1] - second_derivative[i])/(bin_edges[1+i+1]-bin_edges[1+i])
        print(-second_derivative[i]/slope + bin_edges[1+i])
        
# Isolate pool pump by taking the minimum over each second for the 30 day period.
train["daily_sec"]= (train["time"]-train["time"].loc[0])%(3600*24)

min_daily_series = train["power"].iloc[:3600*24]*0 + 50000
max_daily_series = train["time"].iloc[:3600*24]*0
monthly_mean = [0 for i in range(num_days)]
for i in range(num_days):
    monthly_mean[i] = 0
for i in range(num_days):
    min_daily_series = np.minimum(min_daily_series,train["power"].iloc[i*3600*24:(i+1)*3600*24])
    max_daily_series = np.maximum(max_daily_series,train["power"].iloc[i*3600*24:(i+1)*3600*24])    
    monthly_mean[i] = np.mean(train["power"].iloc[i*3600*24:(i+1)*3600*24])

plt.plot( range(num_days),monthly_mean)
plt.title("Mean power consumed over 31 days for each second")
plt.xlabel("Time (days)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

plt.plot( range(0,3600*24),min_daily_series)
plt.title("Min power consumed over 31 days for each second")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

plt.plot(range(0,3600*24),max_daily_series)
plt.title("Max power consumed over 31 days for each second")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

first_derivative_power = abs(np.gradient(train["power"],edge_order=2))
plt.hist(first_derivative_power[first_derivative_power>500],num_bins)
plt.title("Histogram of power consumption transitions")
plt.xlabel("Power consumed (Watts)")
plt.ylabel("Frequency")
plt.grid()
plt.show()

counts_power, bin_edges_power = np.histogram(first_derivative_power, bins=num_bins)
cdf_power = np.cumsum(counts_power)/np.cumsum(counts_power).max()
plt.grid()
plt.plot(bin_edges_power[1:], cdf_power)
plt.title("CDF of power consumption transitions")
plt.xlabel("Power consumed (Watts)")
plt.ylabel("CDF")
plt.show()

second_derivative_power = np.gradient(cdf_power)
plt.grid()
plt.plot(bin_edges_power[1:], second_derivative_power)
plt.title("Derivative of PDF of power consumption transisions")
plt.xlabel("Power consumed (Watts)")
plt.ylabel("PDF")
plt.show()

waterpump_on_maxval = int(min((min(np.where(min_daily_series > 1500)))))
waterpump_off_minval = int(max((max(np.where(min_daily_series > 1500)))))

print("Max value of the lower index when water pump is ON = ",waterpump_on_maxval)
print("Min value of the higher index when water pump is ON = ",waterpump_off_minval)
# Values are 5398 and 15139
#waterpump_on_maxval = 5398
#waterpump_off_minval = 15139

#plt.plot(range(0,len(train[:2*86400])),train["power"][:2*86400])
plt.plot(range(0,len(train)),train["power"])
plt.title("Before removing pool pump")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

#b = extract_and_remove(train[:2*86400], 1500, waterpump_off_minval - waterpump_on_maxval, "pool_pump")
b = extract_and_remove(train, 1500, waterpump_off_minval - waterpump_on_maxval, "pool_pump")

plt.plot(range(0,len(b["pool_pump"])),b["pool_pump"])
plt.title("Pool pump")
plt.xlabel("Time (seconds)")
plt.ylabel("ON/OFF")
plt.grid()
plt.ylim(-0.5,1.5)
plt.show()

Alg2 = b
Alg2["power_diminished"] = Alg2["power"]
Alg2["AC1"] = 0
Alg2["AC2"] = 0


plt.plot(range(0,len(b["power"])),b["power"])
plt.title("After removing pool pump")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

#c = extract_and_remove(b, 4000, 25*60, "AC2") # 25 minutes
c = extract_and_remove(b, 4000, 1, "AC2") # 25 minutes

plt.plot(range(0,len(c["AC2"])),c["AC2"])
plt.title("AC2")
plt.xlabel("Time (seconds)")
plt.ylabel("ON/OFF")
plt.grid()
plt.ylim(-0.5,1.5)
plt.show()

plt.plot(range(0,len(c["power"])),c["power"])
plt.title("After removing AC2")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

#d = extract_and_remove(c, 2500, 8*60, "AC1") # 8 minutes
d = extract_and_remove(c, 2500, 1, "AC1") # 8 minutes

plt.plot(range(0,len(d["AC1"])),d["AC1"])
plt.title("AC1")
plt.xlabel("Time (seconds)")
plt.ylabel("ON/OFF")
plt.grid()
plt.ylim(-0.5,1.5)
plt.show()

plt.plot(range(0,len(d["power"])),d["power"])
plt.title("After removing AC1")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

#d.to_csv("data_final_algorithm1.csv")
# Time when water pump in ON on all days = 5398 to 15139 seconds

# There is significant residual power that is unaccounted for. It turns out
# that the period for which the values stay up can be less than the values 
# provided in the challenge. So, we are going to take an alrernate approach 
# of using thresholding to quantify which device is ON.

Low_Th_pump_ac1_ac2 = (8000+6500)/2
High_Th_ac1_ac2 = Low_Th_pump_ac1_ac2
Low_Th_ac1_ac2 = (6500+5500)/2
High_Th_pump_ac2 = Low_Th_ac1_ac2
Low_Th_pump_ac2 = (5500+4000)/2
High_Th_ac2 = Low_Th_pump_ac2
Low_Th_ac2 = (4000+2500)/2
High_Th_ac1 = Low_Th_ac2
Low_Th_ac1 = (2500+1500)/2

Alg2["power"] = Alg2["power_diminished"]

plt.hist(Alg2["power"],num_bins)
plt.title("Histogram of power consumed after removing pool pump")
plt.xlabel("Power consumed (Watts)")
plt.ylabel("Frequency")
plt.grid()
plt.show()

counts, bin_edges = np.histogram(Alg2["power"], bins=num_bins)
cdf_afterpump = np.cumsum(counts)/np.cumsum(counts).max()
plt.grid()
plt.plot(bin_edges[1:], cdf_afterpump)
plt.title("CDF of power consumed after removing pool pump")
plt.xlabel("Power consumed (Watts)")
plt.ylabel("CDF")
plt.show()

first_derivative_after_pump = np.gradient(cdf_afterpump)
plt.grid()
plt.plot(bin_edges[1:], first_derivative_after_pump)
plt.title("PDF of power consumed after removing pool pump")
plt.xlabel("Power consumed (Watts)")
plt.ylabel("PDF")
plt.xlim(0, 10000)
plt.show()

second_derivative_after_pump = np.gradient(first_derivative_after_pump)
plt.grid()
plt.plot(bin_edges[1:], second_derivative_after_pump)
plt.title("Derivative of PDF of power consumed after removing pool pump")
plt.xlabel("Power consumed (Watts)")
plt.ylabel("PDF")
plt.xlim(0, 10000)
plt.show()

print("PEAKS:")
for i in range(num_bins-1):
    if((second_derivative[i]>0) & (second_derivative[i+1] <=0)):
        slope = (second_derivative[i+1] - second_derivative[i])/(bin_edges[1+i+1]-bin_edges[1+i])
        print(-second_derivative[i]/slope + bin_edges[1+i])

print("TROUGHS:")
for i in range(num_bins-1):
    if((second_derivative[i]<0) & (second_derivative[i+1] >=0)):
        slope = (second_derivative[i+1] - second_derivative[i])/(bin_edges[1+i+1]-bin_edges[1+i])
        print(-second_derivative[i]/slope + bin_edges[1+i])
        

plt.plot(range(0,len(Alg2["power"])),Alg2["power"])
plt.title("Power After removing pool pump in Alg2")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

plt.plot(range(0,len(Alg2["power_diminished"])),Alg2["power_diminished"])
plt.title("Power diminished After removing pool pump in Alg2")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

#Low_Th_pump_ac1_ac2 = (8000+6500)/2
ind = Alg2["power"] >= Low_Th_pump_ac1_ac2
Alg2["New_Device"] = 0
Alg2["New_Device"].loc[ind] = 1
Alg2["AC1"].loc[ind] = 1
Alg2["AC2"].loc[ind] = 1
Alg2["power_diminished"].loc[ind] = Alg2["power_diminished"].loc[ind] - 8000

print("************ LEN OF New_Device+AC1+AC2", len(ind))
plt.plot(range(0,len(Alg2["power_diminished"])),Alg2["power_diminished"])
plt.title("After removing AC1+AC2+new device in Alg2")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

#High_Th_ac1_ac2 = Low_Th_pump_ac1_ac2
#Low_Th_ac1_ac2 = (6500+5500)/2
indexes = (Alg2["power"] >= Low_Th_ac1_ac2) & (Alg2["power"] < High_Th_ac1_ac2)
print("************ LEN OF AC1+AC2", len(indexes))
Alg2["AC1"].loc[indexes] = 1
Alg2["AC2"].loc[indexes] = 1
Alg2["power_diminished"].loc[indexes] = Alg2["power_diminished"].loc[indexes] - 6500
plt.plot(range(0,len(Alg2["power_diminished"])),Alg2["power_diminished"])
plt.title("After removing AC1+AC2 in Alg2")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

#High_Th_pump_ac2 = Low_Th_ac1_ac2
#Low_Th_pump_ac2 = (5500+4000)/2

indexes_pump_AC2 = (Alg2["power"] >= Low_Th_pump_ac2) & (Alg2["power"] < High_Th_pump_ac2)
print("************ LEN OF New_Device+AC2", len(indexes_pump_AC2))
Alg2["New_Device"].loc[indexes_pump_AC2] = 1
Alg2["AC2"].loc[indexes_pump_AC2] = 1
Alg2["power_diminished"].loc[indexes_pump_AC2] = Alg2["power_diminished"].loc[indexes_pump_AC2] - 5500
plt.plot(range(0,len(Alg2["power_diminished"])),Alg2["power_diminished"])
plt.title("After removing new device+AC2 in Alg2")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

#High_Th_ac2 = Low_Th_pump_ac2
#Low_Th_ac2 = (4000+2500)/2

indexes_AC2 = (Alg2["power"] >= Low_Th_ac2) & (Alg2["power"] < High_Th_ac2)
print("************ LEN OF AC2", len(indexes_AC2))
Alg2["AC2"].loc[indexes_AC2] = 1
Alg2["power_diminished"].loc[indexes_AC2] = Alg2["power_diminished"].loc[indexes_AC2] - 4000
plt.plot(range(0,len(Alg2["power_diminished"])),Alg2["power_diminished"])
plt.title("After removing AC2 in Alg2")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

#High_Th_ac1 = Low_Th_ac2
#Low_Th_ac1 = (2500+1500)/2

indexes_AC1 = (Alg2["power"] >= Low_Th_ac1) & (Alg2["power"] < High_Th_ac1)
print("************ LEN OF AC1", len(indexes_AC1))
Alg2["AC2"].loc[indexes_AC1] = 1
Alg2["power_diminished"].loc[indexes_AC1] = Alg2["power_diminished"].loc[indexes_AC1] - 4000
plt.plot(range(0,len(Alg2["power_diminished"])),Alg2["power_diminished"])
plt.title("After removing AC1 in Alg2")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

Low_Th_New_Device = 1500
indexes_pump = (Alg2["power"] >= Low_Th_New_Device)
print("************ LEN OF pool_pump", len(indexes_pump))
Alg2["New_Device"].loc[indexes_pump] = 1
Alg2["power_diminished"].loc[indexes_pump] = Alg2["power_diminished"].loc[indexes_pump] - 1500
plt.plot(range(0,len(Alg2["power_diminished"])),Alg2["power_diminished"])
#plt.title("After removing pool pump in Alg2")
plt.title("Residual powers unaccounted for")
plt.xlabel("Time (seconds)")
plt.ylabel("Value (in Watts)")
plt.grid()
plt.show()

plt.plot(range(0,len(Alg2["pool_pump"])),Alg2["pool_pump"])
plt.title("Pool Pump instants")
plt.xlabel("Time (seconds)")
plt.ylabel("ON/OFF")
plt.grid()
plt.ylim(-0.5, 1.5)
plt.show()

plt.plot(range(0,len(Alg2["New_Device"])),Alg2["New_Device"])
plt.title("New Device instants")
plt.xlabel("Time (seconds)")
plt.ylabel("ON/OFF")
plt.grid()
plt.ylim(-0.5, 1.5)
plt.show()

plt.plot(range(0,len(Alg2["AC1"])),Alg2["AC1"])
plt.title("AC1 instants")
plt.xlabel("Time (seconds)")
plt.ylabel("ON/OFF")
plt.grid()
plt.ylim(-0.5, 1.5)
plt.show()

plt.plot(range(0,len(Alg2["AC2"])),Alg2["AC2"])
plt.title("AC2 instants")
plt.xlabel("Time (seconds)")
plt.ylabel("ON/OFF")
plt.grid()
plt.ylim(-0.5, 1.5)
plt.show()

DF_final = Alg2.copy()
DF_final.drop('power_diminished', axis=1, inplace=True)
DF_final.drop('power', axis=1, inplace=True)
DF_final.drop('time_offset', axis=1, inplace=True)
DF_final.drop('time_hrs', axis=1, inplace=True)
DF_final.drop('daily_sec', axis=1, inplace=True)

DF_final.to_csv("data_final_algorithm2.csv")