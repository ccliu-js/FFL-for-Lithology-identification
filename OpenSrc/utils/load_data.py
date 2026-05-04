import os
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm


class LoadData:
    """Load raw WebDAQ CSV files or a preprocessed pickle dataset."""

    def __init__(self, path, save_path=None):
        self.root_dir = path
        self.save_path = save_path
        self.name_map = {
            "ChangShiShaYan": "Arkose",
            "CuHuangSha": "Sandstone",
            "DaLiShi": "Marble",
            "HuaGangYan": "Granite",
            "NiHuiYan": "Marlstone",
            "ShiHuiYan": "Limestone",
            "YeYan": "Shale",
        }
        self.data = None

        if Path(self.root_dir).is_file():
            self.load_signal_data()
        else:
            print("data will be saved to ", self.save_path)
            if not self.save_path:
                raise ValueError(
                    "save_path must be provided when loading from raw data directory"
                )
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.read_signal_data()

    def read_signal_data(self):
        data = {}
        speeds = ["r2000", "r3000", "r4000"]

        for speed in speeds:
            speed_dir = os.path.join(self.root_dir, speed)
            if not os.path.exists(speed_dir):
                continue

            data[speed] = {}
            for category in tqdm(os.listdir(speed_dir), desc=f"Processing {speed}", ncols=100):
                category_dir = os.path.join(speed_dir, category)
                if not os.path.isdir(category_dir):
                    continue

                label = self.name_map.get(category, category)
                for file in tqdm(
                    os.listdir(category_dir),
                    desc=f"Files in {category}",
                    ncols=100,
                    leave=False,
                ):
                    if not file.endswith(".csv"):
                        continue

                    file_path = os.path.join(category_dir, file)
                    signal_data = self.read_data_only(file_path)

                    for axis in ["Acc_X", "Acc_Y", "Acc_Z"]:
                        if axis not in data[speed]:
                            data[speed][axis] = {}
                        if label not in data[speed][axis]:
                            data[speed][axis][label] = []

                        data[speed][axis][label].append(signal_data[axis])

        with open(self.save_path, "wb") as f:
            pickle.dump(data, f)
        self.data = data
        return data

    def load_signal_data(self):
        """Load a processed dataset from a pickle file."""
        if self.data is None:
            with open(self.root_dir, "rb") as f:
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
        """Read only the signal table from a WebDAQ CSV file."""
        path = Path(file_name)
        try:
            df = pd.read_csv(path, skiprows=5)
            df.columns = ["Sample", "Time_s", "Acc_X", "Acc_Y", "Acc_Z"]
            df = df.drop(columns=["Sample", "Time_s"])
            return df.to_dict(orient="list")
        except Exception as e:
            return {"error": str(e)}
