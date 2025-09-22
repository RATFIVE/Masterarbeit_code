import copernicusmarine
import xarray as xr
import os
from dotenv import load_dotenv

load_dotenv('.env')
username = str(os.getenv("USERNAME"))
password = str(os.getenv("PASSWORD"))

class AdvancedCopernicus:
    def __init__(self):
        self.client = copernicusmarine  # Using composition to access copernicusmarine functions

    def get_dataset(self, dataset_id):
        return self.client.get_dataset(dataset_id)
    
    def get_subset(self, 
                   dataset_id: str, 
                   #dataset_version: str, 
                   variables: list, 
                   minimum_longitude: float, 
                   maximum_longitude: float, 
                   minimum_latitude: float, 
                   maximum_latitude: float, 
                   start_datetime: str, 
                   end_datetime: str,
                   minimum_depth: float = 0.5016462206840515,
                   maximum_depth: float = 0.5016462206840515, 
                   coordinates_selection_method: str = "strict-inside", 
                   disable_progress_bar: bool = False, 
                   username: str = username, 
                   password: str = password,
                   output_filename: str = 'output.nc',
                   delete_file=True,
                   ):
        # Fetch subset data and save to output_filename
        self.client.subset(
            dataset_id=dataset_id,
            #dataset_version=dataset_version,
            variables=variables,
            minimum_longitude=minimum_longitude,
            maximum_longitude=maximum_longitude,
            minimum_latitude=minimum_latitude,
            maximum_latitude=maximum_latitude,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            minimum_depth=minimum_depth,
            maximum_depth=maximum_depth,
            coordinates_selection_method=coordinates_selection_method,
            disable_progress_bar=disable_progress_bar,
            username=username,
            password=password,
            output_filename=output_filename,
            
        )
        # Load the downloaded NetCDF file into an xarray Dataset
        data = xr.open_dataset(output_filename)

        if delete_file:
            self.delete_dataset(output_filename)
        return data
        
    
    def delete_dataset(self, file_name):
        os.remove(file_name)
        #delete all file with .nc extension
        # for file in os.listdir():
        #     if file.endswith(".nc"):
        #         os.remove(file)
        


if __name__ == '__main__':
    import time
    copernicus = AdvancedCopernicus()
    
    subset = copernicus.get_subset(
                dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
                dataset_version="202406",
                variables=["so", "thetao", "vo", "zos", "uo"], 
                minimum_longitude=50,
                maximum_longitude=51,
                minimum_latitude=10,
                maximum_latitude=11,
                start_datetime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                end_datetime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 60*60*24*7)),
                minimum_depth=0.49402499198913574,
                maximum_depth=0.49402499198913574,
                coordinates_selection_method="strict-inside",
                disable_progress_bar=False,
                output_filename='output_filename.nc'
                )
    
    copernicus.delete_dataset('output_filename.nc')