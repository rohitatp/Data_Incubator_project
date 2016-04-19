# Data_Incubator_project

Given aggregate energy consumption data for one home that contains the following main appliances, process the data to extract the energy consumption time series for individual appliances listed above.
•	Central AC 1 - The most common repeating pulse. (At about 2.5 KW amplitude and a width of about 10 minutes)
•	Central AC 2 - Another repeating pulse but less frequent (At about 4 KW amplitude and > 30 minute width)
•	Pool Pump - Runs for a duration of about 3 hours at 1.5 KW amplitude. Starts at the same time every day.
•	Refrigerator - This is the smallest amplitude repeating pulse at < 200 W

Dataset is available at https://bidgely-drupal.s3.amazonaws.com/resources/hiring/data.txt.zip
Same dataset is also available at https://bidgely-drupal.s3.amazonaws.com/resources/hiring/data.txt.gz

Steps:
1. In order to run the DI_project.py, edit the data file and add the following header to it. 
This header will make sure that the dataframe is read by the script under appropriate names.
time,power
2. Keep the data, the .py file in the same folder and run it on python. Alg2 is the algorithm 
that is proposed.
3. Have fun :-)
