#!/usr/bin/env python
# coding: utf-8

# ### Capstone Project
# 
# # Opening a Vegetarian Restaurant in Toronto
# 
# ##### by Christine Brachthäuser
# 

# From the Open Data Portal of the City of Toronto the following table with data based on the 2016 Census of 140 Toronto neighborhoods is downloaded. 

# In[1]:


# The code was removed by Watson Studio for sharing.
import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0


# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.

if os.environ.get('RUNTIME_ENV_LOCATION_TYPE') == 'external':
    endpoint_6cb1bd1db7a943ccbd80661dd83649f1 = 'https://s3-api.us-geo.objectstorage.softlayer.net'
else:
    endpoint_6cb1bd1db7a943ccbd80661dd83649f1 = 'https://s3-api.us-geo.objectstorage.service.networklayer.com'

client_6cb1bd1db7a943ccbd80661dd83649f1 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='r4v08f1tiZAD_WU-g7TQsHCsUFknaHrFjUOu83cUGr75',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url=endpoint_6cb1bd1db7a943ccbd80661dd83649f1)

body = client_6cb1bd1db7a943ccbd80661dd83649f1.get_object(Bucket='courseracapstone-donotdelete-pr-ukcq33ldvta5e3',Key='neighbourhood-profiles-2016-csv.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_1 = pd.read_csv(body)
df_data_1.head()


# In[40]:


df_data_1["Characteristic"][989]
df_data_1["Characteristic"][989]=df_data_1["Characteristic"][989].replace('to', ' ')
df_data_1.loc[[989]]


# In[32]:


df_data_1["Characteristic"] = df_data_1["Characteristic"].astype(str)


# In[30]:


df_data_1.loc[[989]]


# In[2]:


print('The table has {} rows in all.'.format(df_data_1.shape[0]))


# In[3]:


import json
import numpy as np

import csv


import requests
from pandas.io.json import json_normalize
from time import sleep

import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as colors

from sklearn.cluster import KMeans 

print('Liberies imported.')


# #### The census data dataframe is now reduced to basic neighborhood characteristics relevant for this project that can be directly extracted from the source table.
# 

# In[4]:


df_profiles = df_data_1.iloc[[2,4,1675,2290,2354],4:].copy()
df_profiles.drop(['City of Toronto'], axis=1, inplace=True)
df_profiles


# #### Cleaning up data for better use later on

# In[5]:


# moving the percentage sign into the index 

df_profiles = df_profiles.replace(["Population Change 2011-2016"], "Population Change 2011-2016 (%)")


# In[6]:


df_profiles = df_profiles.set_index('Characteristic')
df_profiles


# In[7]:


# eliminating the commas and the percent signs from the table

df_profiles = df_profiles.apply(lambda x:x.str.replace(',','').str.replace('%',''))
df_profiles


# In[8]:


# for convenience, the index names are shortened somewhat

df_profiles.index = ['Pop.', 'Pop. Change (%)', 'Higher Education', 'Av. Ret. Income ($)', 'Av. Income ($)']
df_profiles


# In[9]:


# the neighborhoods are moved to the index column and the selected characteristics become regular columns

df_profiles_trans = df_profiles.transpose()
df_profiles_trans = df_profiles_trans.rename_axis('Neighborhood')
df_profiles_trans.head()


# #### In addition to the absolute numbers of people with postsecondary education the respective percentage in regard to the overall population in each neighborhood is calculated and added to the table.

# In[10]:


df_profiles_trans.dtypes


# In[11]:


df_profiles_trans[['Pop.', 'Higher Education']] = df_profiles_trans[['Pop.', 'Higher Education']].astype(int)
df_profiles_trans['Pop. Change (%)'] = df_profiles_trans['Pop. Change (%)'].astype(float)


# In[12]:


df_profiles_trans['Higher Education (%)'] = df_profiles_trans['Higher Education']/df_profiles_trans['Pop.']*100
df_profiles_trans['Higher Education (%)'] = np.round(df_profiles_trans['Higher Education (%)'], decimals=2)
df_profiles_trans.head()


# In[13]:


# changing the order of columns for better readability

columns_names = ['Pop.','Pop. Change (%)','Higher Education','Higher Education (%)','Av. Income ($)','Av. Ret. Income ($)']

df_profiles_trans = df_profiles_trans.reindex(columns = columns_names)

df_profiles_trans.head()


# In[21]:


df_data_1.iloc[989]


# ### Middle and higher incomes

# So far only average income is considered. For a more differentiated picture, the number of people with a personal after-tax income above $30000 #### and the percentage they represent in regard to all after-tax income earners is now calculated so that they can be added to the table later on. 
# 

# In[16]:


df_profiles_income = df_data_1.iloc[986:997, 4:].copy()

df_profiles_income.drop([987], axis=0, inplace=True)
#df_profiles_income.drop(['City of Toronto'], axis=1, inplace=True)
#df_profiles_income = df_profiles_income.set_index('Characteristic')
df_profiles_income.head()


# In[15]:


df_profiles_income.dtypes


# In[14]:


df_profiles_income = df_profiles_income.apply(lambda x:x.str.replace(',',''))


# In[15]:


df_profiles_income = df_profiles_income.astype(int)


# In[16]:


df_profiles_income 


# In[77]:


# summing up the number of people with individual incomes above $30000 in each neighborhood

df_higher_incomes = pd.DataFrame(df_profiles_income.iloc[4:].sum(axis=0), columns = ['Inc. above $30000'])
df_higher_incomes.head()


# In[22]:


df_income_percent = df_profiles_income.copy()
df_income_percent = df_income_percent.div(df_income_percent.iloc[0], axis=1)
df_income_percent = np.round(df_income_percent, decimals = 2)
df_income_percent.head()


# In[23]:


# summing up the share of people with individual incomes above $30000

df_higher_incomes_percent = pd.DataFrame(df_income_percent.iloc[4:].sum(axis=0)*100, columns = ['Inc. above $30000 (%)'])
df_higher_incomes_percent.head()


# ### Middle age population

# The number of people between 30 and 75 years as well as the share of the overall population they represent is now calculated. 

# In[26]:


df_profiles_age = df_data_1.iloc[15:57, 4:].copy()
df_profiles_age.drop(['City of Toronto'], axis=1, inplace = True)
df_profiles_age


# In[27]:


# the following rearrangement is necessary since row 31 is misplaced in the source table;

df_profiles_age = df_profiles_age.reindex([15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,31,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56])
df_profiles_age = df_profiles_age.reset_index()
df_profiles_age


# In[28]:


df_age_men = df_profiles_age.iloc[0:21].copy()
df_age_men.drop(['index'], axis=1, inplace=True)
df_age_men.head()


# In[29]:


df_age_women = df_profiles_age.iloc[21:42].copy()
df_age_women.drop(['index'], axis=1, inplace=True)
df_age_women.head()


# #### Men and women are now merged to one common age group.

# In[30]:


df_age_men_list = df_age_men['Characteristic'].tolist()
df_age_men_list = [s.strip('Male:') for s in df_age_men_list]
df_age_men_list = [s.replace('Male:','') for s in df_age_men_list]

df_age_women_list = df_age_women['Characteristic'].tolist()
df_age_women_list = [s.strip('Female:') for s in df_age_women_list]
df_age_women_list = [s.replace('Female:','') for s in df_age_women_list]


# In[31]:


df_age_men['Characteristic'] = df_age_men_list
df_age_women['Characteristic'] = df_age_women_list


# In[32]:


df_age_men = df_age_men.set_index(['Characteristic'])
df_age_men = df_age_men.astype(int)
df_age_men.head()


# In[31]:


df_age_women = df_age_women.set_index(['Characteristic'])
df_age_women = df_age_women.astype(int)
df_age_women.head()


# In[32]:


# the targeted age group is now reduced to the 30 to 70 year-olds. 

middle_age_men = df_age_men.iloc[6:14].copy()
middle_age_women = df_age_women.iloc[6:14].copy()


# In[33]:


middle_age_men = middle_age_men.astype(int)
middle_age_women = middle_age_women.astype(int)


# In[34]:


middle_age_men.head(2)


# In[35]:


middle_age_women.head(2)


# In[36]:


df_middle_age = middle_age_men + middle_age_women
df_middle_age


# In[ ]:





# In[37]:


df_age = pd.DataFrame(df_middle_age.sum(axis=0), columns=['Pop. 30-70 years'])
df_age.head()


# In[38]:


# now the share of the 30-70-year-olds is calculated

df_age_percent = df_middle_age
df_age_percent = df_age_percent.append(df_profiles.iloc[0])
df_age_percent


# In[39]:


df_age_percent = df_age_percent.astype(int)
df_age_percent = df_age_percent.div(df_age_percent.loc['Pop.'])
df_age_percent = np.round(df_age_percent, decimals = 2)
df_age_percent.drop(['Pop.'], axis=0, inplace=True)


# In[40]:


df_age_p = pd.DataFrame(df_age_percent.sum(axis=0)*100, columns=['Pop. 30-70 years (%)'])
df_age_p.head()


# ### Latitude and longitude coordinates for all neighborhoods

# In[41]:


get_ipython().system('conda install -c conda-forge geopy --yes')
from geopy.geocoders import Nominatim

print('Geocoder imported')


# In[42]:


get_ipython().system('pip install geocoder')
import geocoder


# In[43]:


neighborhoods = df_profiles.columns.tolist()
#neighborhoods


# In[44]:


def get_latlng(neighborhoods_1):

    lat_lng_coords = None

    while (lat_lng_coords is None):
        g = geocoder.arcgis('{}, Toronto, Ontario'.format(neighborhoods_1))
        lat_lng_coords = g.latlng
        
    return lat_lng_coords

print('The latitude of Agincourt North is {} and the longitude is {}.'.format(get_latlng('Agincourt North')[0], get_latlng('Agincourt North')[1]))


# In[45]:


latitudes = [get_latlng(neighborhoods)[0] for neighborhoods in neighborhoods]
sleep(1)

longitudes = [get_latlng(neighborhoods)[1] for neighborhoods in neighborhoods]
sleep(1)


# In[46]:


columns = ['Neighborhood', 'Latitude', 'Longitude']
df_geodata = pd.DataFrame(columns = columns)
df_geodata['Neighborhood'] = neighborhoods
df_geodata['Latitude'] = latitudes
df_geodata['Longitude'] = longitudes
df_geodata.head()
    


# In[66]:


df_geodata.shape


# ## Venue search with the Foursquare database

# In[47]:


CLIENT_ID = 'DHR44SDFIOFZWMVRIVWQFKNYLQHHPGTUN43UJPA5JNHFL0CT'
CLIENT_SECRET = 'QFCZXAPZBW1Q4ERFP1P3SCVC0QXLKAU33J00X0DNRZ5WXGFO'
VERSION = '20180605'

print('Your Credentials:')
print('CLIENT_ID:' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[48]:


address = 'Toronto, ON'
geolocator = Nominatim(user_agent = "toronto_explorer")
location = geolocator.geocode(address)
tor_latitude = location.latitude
tor_longitude = location.longitude
print('The geographical coordinates of Toronto are {}, {}.'.format(tor_latitude, tor_longitude))


# #### Search for vegetarian restaurants in Toronto as a whole with a radius of 20000.
# 

# In[49]:


urlvegy = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius=20000&limit=1000&query=vegetarian / vegan restaurant'.format(
CLIENT_ID,CLIENT_SECRET,VERSION,tor_latitude,tor_longitude)

urlvegy


# In[50]:


resultsvegy = requests.get(urlvegy).json()


# In[51]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[52]:


venues = resultsvegy['response']['groups'][0]['items']
vegy = pd.json_normalize(venues)
filtered_columns = ['venue.name','venue.categories','venue.location.lat','venue.location.lng','venue.location.address']
vegy['categories'] = vegy.apply(get_category_type, axis=1)
vegy.columns = [col.split(".")[-1] for col in vegy.columns]
vegy_restaurants = vegy[['name','categories','lat','lng','address']].copy()
vegy_restaurants.head()


# In[53]:


print('The search revealed that in total there are {} vegetarian restaurants in Toronto.'.format(vegy_restaurants.shape[0]))


# In[54]:


# search for vegetarian restaurants in the higher prices segments

urlvegy3 = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius=20000&limit=1000&query=vegetarian / vegan restaurant&price=3,4'.format(
CLIENT_ID,CLIENT_SECRET,VERSION,tor_latitude,tor_longitude)

urlvegy3


# In[55]:


resultsvegy3 = requests.get(urlvegy3).json()
#resultsvegy


# In[56]:


venues3 = resultsvegy3['response']['groups'][0]['items']
vegy3 = pd.json_normalize(venues3)
filtered_columns = ['venue.name','venue.categories','venue.location.lat','venue.location.lng','venue.location.address']
vegy3['categories'] = vegy3.apply(get_category_type, axis=1)
vegy3.columns = [col.split(".")[-1] for col in vegy3.columns]
vegy3_restaurants = vegy3[['name','categories','lat','lng','address']].copy()
vegy3_restaurants


# In[57]:


vegy_gourmet = vegy3_restaurants.iloc[:,[0,2,3,4,5]]
vegy_gourmet


# In[58]:


print('It turns out that only {} vegetarian restaurants fall into the upper price category.'.format(vegy_gourmet.shape[0]))


# In[ ]:





# In[ ]:





# ### Venue Search

# The following venue search is conducted in regard to the individual neighborhoods to determine how well a vegetarian gourmet restaurant would fit in there. The radius is set to 1500, which seems to be sufficient to get a general impression of the business structure in each neighborhood.
# The venue search is done in several stages. A first inquiry refers to venues generally. The result in form of a json file is converted to a pandas dataframe, which serves as source table for further analysis where specific venue categories are then examined to determine to what extent they are represented in each neighborhood. 

# In[59]:


# I need to change radius and limit;

def getNearbyVenues(names, latitudes, longitudes, radius=1500, LIMIT=1000):
    
    venues_list = []
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
        
        urlvenue = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
                    CLIENT_ID,CLIENT_SECRET,VERSION,lat,lng,radius,LIMIT)
        
        results = requests.get(urlvenue).json()['response']['groups'][0]['items']
        
        venues_list.append([(
            name,
            lat,
            lng,
            v['venue']['name'],
            v['venue']['location']['lat'],
            v['venue']['location']['lng'],
            v['venue']['categories'][0]['name']) for v in results])
    
    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood','Neighborhood Latitude','Neighborhood Longitude','Venue','Venue Latitude','Venue Longitude','Venue Category']
    
    return(nearby_venues)
    


# In[60]:


toronto_venues = getNearbyVenues(names = df_geodata['Neighborhood'],
                                 latitudes = df_geodata['Latitude'],
                                 longitudes = df_geodata['Longitude']
                                 )


# In[61]:


print(toronto_venues.shape)
toronto_venues.head()


# In[62]:


venue_count = toronto_venues.groupby('Neighborhood').count()
neighborhood_venues = venue_count['Venue'].to_frame()
neighborhood_venues = neighborhood_venues.rename(columns={'Venue':'Venues'})
neighborhood_venues.head()


# #### Search for complementary venues

# In[63]:


complement_venues_list = ['Farmers Market','Gourmet Shop','Health Food Store','Organic Grocery','Food & Vegetable Store','Health and Beauty Service','Yoga Studio']

complement_venues = toronto_venues.loc[toronto_venues['Venue Category'].isin(complement_venues_list)].reset_index()
complement_venues.head()


# In[64]:


complement_venues.shape


# In[65]:


complement_count = complement_venues.groupby('Neighborhood').count()
neighborhood_complements = complement_count['Venue'].to_frame()


# In[66]:


neighborhood_complements = neighborhood_complements.rename(columns={'Venue':'Compl. Venues'})
neighborhood_complements.head()            


# In[67]:


# search for gyms and fitness centers

gym_venues = toronto_venues.loc[toronto_venues['Venue Category'].str.contains('Gym', case=False)].reset_index()
gym_venues.head()                            


# In[68]:


gym_venues.shape


# In[69]:


gym_count = gym_venues.groupby('Neighborhood').count()
neighborhood_gyms = gym_count['Venue'].to_frame()
neighborhood_gyms = neighborhood_gyms.rename(columns={'Venue':'Gyms'})
neighborhood_gyms.head()


# #### Search for vegetarian restaurants in each neighborhood

# In[70]:


vegy_venues = toronto_venues[toronto_venues['Venue Category'] == 'Vegetarian / Vegan Restaurant']
vegy_venues.reset_index(drop=True, inplace=True)
vegy_venues.head()


# In[71]:


vegy_venues.shape 


# The discrepancy between this number of 52 vegetarian restaurants and the 94 vegetarian restaurants that were obtained in the search above is due to the relatively small radius. 

# In[72]:


vegy_count = vegy_venues.groupby('Neighborhood').count()
vegy_neighborhoods = vegy_count['Venue'].to_frame()
vegy_neighborhoods = vegy_neighborhoods.rename(columns={'Venue':'Vegy Rests'})
vegy_neighborhoods.head()


# ####  Search for all restaurants per neighborhood

# In[73]:


toronto_restaurants = toronto_venues.loc[toronto_venues['Venue Category'].str.contains('Restaurant', case=False).reset_index(drop=True)]
toronto_restaurants.shape


# In[74]:


restaurant_count = toronto_restaurants.groupby('Neighborhood').count()
neighborhood_restaurants = restaurant_count['Venue'].to_frame()
neighborhood_restaurants = neighborhood_restaurants.rename(columns={'Venue':'Rests'})
neighborhood_restaurants.head()


# In[ ]:





# ## Merging data 

# All the data that has been gathered so far is now merged to a new dataframe that serves as the basis for the cluster analysis.

# In[75]:


df_profiles_trans.head()


# In[78]:


df_profiles_merged = pd.concat([df_profiles_trans,df_higher_incomes,df_higher_incomes_percent,df_age,df_age_p,neighborhood_venues,neighborhood_restaurants,vegy_neighborhoods,neighborhood_complements,neighborhood_gyms], axis=1).fillna(0)
df_profiles_merged.head()


# In[79]:


print('{} categories are now included as selection criteria for the decision-making process.'.format(df_profiles_merged.shape[1]))


# ### Clustering neighborhoods

# In[80]:


from sklearn.preprocessing import StandardScaler

X = df_profiles_merged.values[:,1:]
X = np.nan_to_num(X)

cluster_dataset = StandardScaler().fit_transform(X)
cluster_dataset


# In[81]:


num_cluster = 10

k_means = KMeans(init = "k-means++", n_clusters=num_cluster, n_init=12)
k_means.fit(cluster_dataset)
labels = k_means.labels_
print(labels)


# In[82]:


df_profiles_merged['Labels'] = labels
df_profiles_merged.head()


# In[83]:


df_profiles_merged.groupby('Labels').mean()


# In[84]:


df_cluster_0 = df_profiles_merged.loc[df_profiles_merged['Labels'] == 0]
df_cluster_0


# In[103]:


df_cluster_1 = df_profiles_merged.loc[df_profiles_merged['Labels'] == 1]
df_cluster_1


# In[85]:


df_cluster_2 = df_profiles_merged.loc[df_profiles_merged['Labels'] == 2]
df_cluster_2


# In[86]:


df_cluster_3 = df_profiles_merged.loc[df_profiles_merged['Labels'] == 3]
df_cluster_3


# In[87]:


df_cluster_4 = df_profiles_merged.loc[df_profiles_merged['Labels'] == 4]
df_cluster_4


# In[88]:


df_cluster_5 = df_profiles_merged.loc[df_profiles_merged['Labels'] == 5]
df_cluster_5


# In[89]:


df_cluster_6 = df_profiles_merged.loc[df_profiles_merged['Labels'] == 6]
df_cluster_6


# In[90]:


df_cluster_7 = df_profiles_merged.loc[df_profiles_merged['Labels'] == 7]
df_cluster_7


# In[91]:


df_cluster_8 = df_profiles_merged.loc[df_profiles_merged['Labels'] == 8]
df_cluster_8


# In[92]:


df_cluster_9 = df_profiles_merged.loc[df_profiles_merged['Labels'] == 9]
df_cluster_9


# #### Cluster analysis conclusion
# The cluster analysis singled out the Waterfront Communities - The Island as the most promising candidate as site for the new restaurant.
# 
# 

# ## Plausibility Test

# To see whether the Waterfront Communities really are the best location for the restaurant, the neighborhood is compared to other potential candidate neighborhoods in regard to income and age distribution. For the comparison Casa Loma and Kensington-Chinatown were chosen. Casa Loma belongs to cluster 9 of the most well-off downtown neighborhoods. Kensington-Chinatown belongs to cluster 8 that contains the neighborhoods with the most already existing vegetarian restaurants.  

# ### Income Distribution

# In[17]:


df_profiles_income


# In[18]:


df_income_distribution = df_profiles_income[['Casa Loma','University','Waterfront Communities-The Island']]
df_income_distribution.head()


# In[19]:


df_income_percent = df_income_distribution.div(df_income_distribution.iloc[0], axis=1)
df_income_percent = np.round(df_income_percent, decimals=2)
df_income_percent


# In[20]:


income_list = ['Under 20','Under 20','20-40','20-40','40-60','40-60','60-80','60-80','Over 80']


# In[21]:


df_income_percent.drop(df_income_percent.index[0], inplace=True)


# In[22]:


df_income_percent.index = income_list
df_income_percent.head()


# In[23]:


aggregation_function = {'Casa Loma':'sum','University':'sum','Waterfront Communities-The Island':'sum'}
df_income_percent_grouped = df_income_percent.groupby(df_income_percent.index).aggregate(aggregation_function)

df_income_percent_grouped.head()


# In[24]:


df_income_percent_grouped.index.name = 'Individual Income (000$)'
new_index = ['Under 20','20-40','40-60','60-80','Over 80']
df_income_percent_grouped = df_income_percent_grouped.reindex(new_index).reset_index()
df_income_percent_grouped


# In[25]:


df_income_percent_grouped.plot(x='Individual Income (000$)', kind='bar', stacked=False,
                              title='Distribution of Income (in %) - Neighborhood Comparison', figsize=(10,5), width=0.5)


# ### Population Pyramid

# In[36]:


casa_loma_women = pd.DataFrame(df_age_women['Casa Loma'])
casa_loma_women = casa_loma_women.rename(columns={'Casa Loma':'Females'})
casa_loma_men = pd.DataFrame(df_age_men['Casa Loma'])
casa_loma_men = casa_loma_men.rename(columns={'Casa Loma':'Males'})
casa_loma_men = casa_loma_men.reset_index()

casa_loma = casa_loma_women
casa_loma = casa_loma.reset_index()
casa_loma['Males'] = casa_loma_men['Males']
casa_loma.head()


# In[37]:


y = range(0, len(casa_loma))
x_males = casa_loma['Males']
x_females = casa_loma['Females']
fig,axes = plt.subplots(ncols=2, sharey=True, figsize=(10,6))
fig.patch.set_facecolor('lightgrey')
plt.figtext(.5,.9, 'Population - Casa Loma', fontsize=15, ha='center')
axes[0].barh(y,x_males, align='center', color='orange')
axes[0].set(title = 'Males')
axes[1].barh(y,x_females, align='center', color='purple')
axes[1].set(title = 'Females')
axes[1].grid()
axes[0].set(yticks = y, yticklabels=casa_loma['Characteristic'])
axes[0].invert_xaxis()
axes[0].grid()
plt1=plt.show()


# In[38]:


university_women = pd.DataFrame(df_age_women['University'])
university_women = university_women.rename(columns={'University':'Females'})
university_men = pd.DataFrame(df_age_men['University'])
university_men = university_men.rename(columns={'University':'Males'})
university_men = university_men.reset_index()

university = university_women
university = university.reset_index()
university['Males'] = university_men['Males']
university.head()


# In[39]:


y = range(0, len(university))
x_males = university['Males']
x_females = university['Females']
fig,axes = plt.subplots(ncols=2, sharey=True, figsize=(10,6))
fig.patch.set_facecolor('lightgrey')
plt.figtext(.5,.9, 'Population - University', fontsize=15, ha='center')
axes[0].barh(y,x_males, align='center', color='orange')
axes[0].set(title = 'Males')
axes[1].barh(y,x_females, align='center', color='purple')
axes[1].set(title = 'Females')
axes[1].grid()
axes[0].set(yticks = y, yticklabels=university['Characteristic'])
axes[0].invert_xaxis()
axes[0].grid()
plt1=plt.show()


# In[40]:


waterfront_women = pd.DataFrame(df_age_women['Waterfront Communities-The Island'])
waterfront_women = waterfront_women.rename(columns={'Waterfront Communities-The Island':'Females'})
waterfront_men = pd.DataFrame(df_age_men['Waterfront Communities-The Island'])
waterfront_men = waterfront_men.rename(columns={'Waterfront Communities-The Island':'Males'})
waterfront_men = waterfront_men.reset_index()

waterfront = waterfront_women
waterfront = waterfront.reset_index()
waterfront['Males'] = waterfront_men['Males']
waterfront.head()


# In[41]:


y = range(0, len(waterfront))
x_males = waterfront['Males']
x_females = waterfront['Females']
fig,axes = plt.subplots(ncols=2, sharey=True, figsize=(10,6))
fig.patch.set_facecolor('lightgrey')
plt.figtext(.5,.9, 'Population - Waterfront', fontsize=15, ha='center')
axes[0].barh(y,x_males, align='center', color='orange')
axes[0].set(title = 'Males')
axes[1].barh(y,x_females, align='center', color='purple')
axes[1].set(title = 'Females')
axes[1].grid()
axes[0].set(yticks = y, yticklabels=waterfront['Characteristic'])
axes[0].invert_xaxis()
axes[0].grid()
plt1=plt.show()


# In[ ]:




