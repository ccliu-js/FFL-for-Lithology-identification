from .load_data import LoadData
from .slice_data import SliceData
import random
import pickle
import os












class ProcessData:
    def __init__(self, data_kwargs=None):
        if data_kwargs is None:
            self.data_kwargs = {
                'data_path': r"A:\lithology\experiment\datasets\raw_data.pkl",
                'data_select_kwargs': {
                    'speed': 'r2000', 
                    'axis': 'Acc_X'
                },  
                'process_kwargs': {
                    'slice_len': 1024,
                    'overlap': 0.0,
                    'drop_last': True,
                    'train_data_single_class_length': 100,
                    'test_data_single_class_length': 100,
                    'savedir': rf"A:\lithology\experiment\datasets",
                },
            }
        else:
            self.data_kwargs = data_kwargs

    def prepare_sore_data(self):
        data_load=LoadData(self.data_kwargs['data_path'])
        data=data_load.data


        selected_data = data[self.data_kwargs['data_select_kwargs']['speed']][self.data_kwargs['data_select_kwargs']['axis']]

        data_slice=SliceData(selected_data, 
                        slice_len=self.data_kwargs['process_kwargs']['slice_len'], 
                        overlap=self.data_kwargs['process_kwargs']['overlap'], 
                        drop_last=self.data_kwargs['process_kwargs']['drop_last'])
        
        sliced_data=data_slice.slice_signal_data()
        train_data = {}
        test_data = {}
        #每类随机打乱切片后的数据，每类取出固定数量的切片作为训练和测试数据，剩余的丢弃
        for class_label, samples in sliced_data.items():
            random.shuffle(samples)
            split_train_point = self.data_kwargs['process_kwargs']['train_data_single_class_length']
            split_test_point = self.data_kwargs['process_kwargs']['test_data_single_class_length']
            if len(samples) < split_train_point + split_test_point:
                print(f"Warning: class {class_label} has only {len(samples)} samples, which is less than the required {split_train_point + split_test_point}.")
                ValueError(f"Class {class_label} has insufficient samples.")

            train_data[class_label] = samples[:split_train_point]
            test_data[class_label] = samples[split_train_point:split_train_point+split_test_point]

        #根据转速和切的点数创建文件夹，并保存切片后的数据
        save_dir = os.path.join(
            self.data_kwargs['process_kwargs']['savedir'],
            f"{self.data_kwargs['process_kwargs']['slice_len']}_{self.data_kwargs['data_select_kwargs']['speed']}_{self.data_kwargs['data_select_kwargs']['axis']}"
        )

        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "test_data.pkl"), 'wb') as f:
            pickle.dump(test_data, f)

        with open(os.path.join(save_dir, "train_data.pkl"), 'wb') as f:
            pickle.dump(train_data, f)

        return train_data, test_data




