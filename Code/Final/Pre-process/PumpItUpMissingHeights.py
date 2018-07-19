
# coding: utf-8

# In[5]:


import pandas
# Read data
data_values = pandas.read_csv("../../../Datasets/Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_values.csv")


# In[9]:


# Find missing gps_height values from longitute and latitude column
#use command --> "pip install --user geocoder" to install Geocoder package
from time import sleep
import geocoder

#Store missing gps heights in a dict {id:gps_height}
gpsHeights = {}

for i in data_values.id:
    if (data_values.loc[data_values.id == i, "gps_height"] == 0).bool():
        latitude = data_values.loc[data_values.id == i, "latitude"]
        longitude = data_values.loc[data_values.id == i, "longitude"]
        skipped = 0
        while True:
            sleep(1)
            #GPS access key 
            geoAccess = geocoder.elevation([latitude, longitude],key="AIzaSyDk3VvbBmuUH5J4vGntZAinU6rynEEKGO4")
            if geoAccess.ok:
                gpsHeights[i] = int(geoAccess.meters)
                break
            else:
                skipped += 1
        #Writing the missing gps heights on to a newGpsheights.csv file
        pandas.DataFrame.from_dict(gpsHeights, orient="index").reset_index().to_csv("../../../Datasets/heights.csv", header=["id", "gps_height"], index=False)

