import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import pickle

class LoadData:
    """
    A class to handle loading and preprocessing of geological signal data.
    
    It supports reading raw data from a directory structure or loading 
    pre-processed data from a pickle file.
    """
    def __init__(self, path, save_path=None):
        """
        Initialize the LoadData instance.

        Args:
            path (str): The root directory of the raw data or the path to a .pkl file.
            save_path (str): The directory where the processed .pkl file will be saved.
        """
        self.root_dir = path
        self.save_path = save_path
        self.name_map = {
            'ChangShiShaYan': 'Arkose',
            'CuHuangSha': 'Sandstone',
            'DaLiShi': 'Marble',
            'HuaGangYan': 'Granite',
            'NiHuiYan': 'Marlstone',
            'ShiHuiYan': 'Limestone',
            'YeYan': 'Shale'
        }
        self.data = None

        if Path(self.root_dir).is_file():
            self.load_signal_data()
        else:
            print("data will be saved to ", self.save_path)
            if not self.save_path:
                raise ValueError("save_path must be provided when loading from raw data directory")
            else:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            self.read_signal_data()

#data structure after loading:
# {
#     "r2000": {
#         "Acc_X/Acc_Y/Acc_Z": {
#             "Arkose/Sandstone/Marble/Granite/Marlstone/Limestone/Shale": 10
#         }
#     },
#     "r3000": {
#         "Acc_X/Acc_Y/Acc_Z": {
#             "Arkose/Sandstone/Marble/Granite/Marlstone/Limestone/Shale": 4
#         }
#     },
#     "r4000": {
#         "Acc_X/Acc_Y/Acc_Z": {
#             "Arkose/Sandstone/Marble/Granite/Marlstone/Limestone/Shale": 4
#         }
#     }
# }
        

    def read_signal_data(self):
        data = {}
        
        speeds = ['r2000', 'r3000', 'r4000']
        
        for speed in speeds:
            speed_dir = os.path.join(self.root_dir, speed)
            if os.path.exists(speed_dir):
                data[speed] = {}
                
                for category in tqdm(os.listdir(speed_dir), desc=f"Processing {speed}", ncols=100):
                    category_dir = os.path.join(speed_dir, category)
                    if os.path.isdir(category_dir):
                        label = self.name_map.get(category, category)

                        for file in tqdm(os.listdir(category_dir), desc=f"Files in {category}", ncols=100, leave=False):
                            if file.endswith('.csv'):
                                file_path = os.path.join(category_dir, file)
                                signal_data = self.read_data_only(file_path)
                                
                                for axis in ['Acc_X', 'Acc_Y', 'Acc_Z']:
                                    if axis not in data[speed]:
                                        data[speed][axis] = {}
                                    if label not in data[speed][axis]:
                                        data[speed][axis][label] = []
                                    
                                    data[speed][axis][label].append(signal_data[axis])

        with open(self.save_path, 'wb') as f:
            pickle.dump(data, f)
        self.data = data
        return data
    


    def load_signal_data(self):
        """Load previously processed data from a pickle file."""
        if self.data is None:
            with open(self.root_dir, 'rb') as f:
                self.data = pickle.load(f)
        for speed in self.data:
            for axis in self.data[speed]:
                new_axis_data = {}
                for label, samples in self.data[speed][axis].items():
                    new_label = self.name_map.get(label, label)
                    new_axis_data[new_label] = samples
                self.data[speed][axis] = new_axis_data
        return self.data



    def read_data_only(self, file_name):
            """
            Reads only the data table portion of the WebDAQ file.
            Returns format: { "Acc_X": [data_list], "Acc_Y": [data_list], "Acc_Z": [data_list] }
            """
            path = Path(file_name)
            try:
                # WebDAQ data typically starts from line 6 (index 5)
                df = pd.read_csv(path, skiprows=5)
                
                # Manually rename columns in case of encoding issues or special characters (optional)
                df.columns = ['Sample', 'Time_s', 'Acc_X', 'Acc_Y', 'Acc_Z']
                
                # Drop irrelevant columns
                df = df.drop(columns=['Sample', 'Time_s'])
                
                # Convert to dictionary format: keys are column names, values are lists of data
                return df.to_dict(orient='list')
            
            except Exception as e:
                return {"error": str(e)}

