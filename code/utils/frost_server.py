
import requests
import json
import pandas as pd
import numpy as np

# import plotly.express as px
# import plotly.graph_objects as go
class FrostServer:
    def __init__(self, 
                 url='https://timeseries.geomar.de/soop/FROST-Server/v1.1/', 
                 thing='Things(3)' # T-Box
                 ):
        self.url = url
        self.thing = thing
        

    def get_content(self, url):
        response = requests.get(url)
        content = response.text
        return json.loads(content)
    
    def get_datastream_url(self):
        content = self.get_content(self.url + self.thing)
        datastream_url = content['Datastreams@iot.navigationLink']
        return datastream_url
    
    def get_position_url(self):
        content = self.get_content(self.url + self.thing)
        position_url = content['Locations@iot.navigationLink']
        return position_url
    

    def get_observations_url(self):
        datastream_url = self.get_datastream_url()
        content_datastream = self.get_content(datastream_url)
        observation_url = content_datastream['value'][0]["Observations@iot.navigationLink"]
        return observation_url
    
    def get_thing_name(self):
        content = self.get_content(self.url + self.thing)
        name_url = content['name']
        return name_url
    
    def print_content(self, content):
        return print(json.dumps(content, indent=4, ensure_ascii=False))
    
    def get_all_observations(self, limit_per_page=1000):
        observation_url = self.get_observations_url()
        params = {
            f"$top": {limit_per_page},  # Limit to 1000 observations per page
            "$orderby": "phenomenonTime asc"  # Sort by phenomenonTime in ascending order
        }
        all_observations = []
        next_link = observation_url

        while next_link:
            response = requests.get(next_link, params=params if next_link == observation_url else None)
            if response.status_code == 200:
                data = response.json()
                all_observations.extend(data["value"])

                    # Check for pagination link
                next_link = data.get("@iot.nextLink")  # Automatically handles pagination
            else:
                print(f"Error: {response.status_code}")
                break
        

        return all_observations

    

    

    

# # %%
# url = 'https://timeseries.geomar.de/soop/FROST-Server/v1.1/'
# t_box = 'Things(3)'

# def get_content(url):
#     response = requests.get(url)
#     content = response.text
#     return json.loads(content)

# content = get_content(url + t_box)
# print(json.dumps(content, indent=4))


# # %%
# datastream_url = content['Datastreams@iot.navigationLink']
# content_datastream = get_content(datastream_url)
# #print(json.dumps(content_datastream, indent=4, ensure_ascii=False))
# observation_url = content_datastream['value'][0]["Observations@iot.navigationLink"]

# # %%
# # Base URL for observations of Datastream(3), sorted by phenomenonTime
# params = {
#     "$top": 1,  # Limit to 1000 observations per page
#     "$orderby": "phenomenonTime asc"  # Sort by phenomenonTime in ascending order
    
# }

# # List to store all observations
# all_observations = []
# next_link = observation_url

# # Loop through all pages until there's no @iot.nextLink
# while next_link:
#     response = requests.get(next_link, params=params if next_link == observation_url else None)
    
#     if response.status_code == 200:
#         data = response.json()
#         all_observations.extend(data["value"])  # Add observations to the list
        
#         # Check for pagination link
#         next_link = data.get("@iot.nextLink")  # Automatically handles pagination
#     else:
#         print(f"Error: {response.status_code}")
#         break

# # print the first 5 observations
# for obs in all_observations[:5]:
#     print(f"Time: {obs['phenomenonTime']}, Result: {obs['result']}")

# print(f"Total Observations Retrieved: {len(all_observations)}")

# # %%
# df_obs = pd.DataFrame(all_observations)
# print(df_obs.head(3))
# print(df_obs.info())

# # %%
# # convert phenomenonTime and resultTime to datetime
# df_obs["phenomenonTime"] = pd.to_datetime(df_obs["phenomenonTime"])
# df_obs["resultTime"] = pd.to_datetime(df_obs["resultTime"])

# # convert result to float
# df_obs["result"] = df_obs["result"].astype(float)





if __name__ == '__main__':
    
    server = FrostServer(thing='Things(3)')
    observation_url = server.get_observations_url()
    content = server.get_content(observation_url)
    all_observations = server.get_all_observations()

    df_obs = pd.DataFrame(all_observations)
    df_obs["phenomenonTime"] = pd.to_datetime(df_obs["phenomenonTime"])
    df_obs["resultTime"] = pd.to_datetime(df_obs["resultTime"])

    # convert result to float
    df_obs["result"] = df_obs["result"].astype(float)

    print(df_obs.head(3))