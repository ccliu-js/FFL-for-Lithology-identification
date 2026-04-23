from .dataset import Dataset
from .load_data import LoadData
from .slice_data import SliceData

def prepare_data(data_kwargs):
    data_load=LoadData(data_kwargs['data_path'])
    data=data_load.data

    selected_data = data[data_kwargs['data_select_kwargs']['speed']][data_kwargs['data_select_kwargs']['axis']]
    
    data_slice=SliceData(selected_data, 
                         slice_len=data_kwargs['process_kwargs']['slice_len'], 
                         overlap=data_kwargs['process_kwargs']['overlap'], 
                         drop_last=data_kwargs['process_kwargs']['drop_last'])
    
    sliced_data=data_slice.slice_signal_data()

    dataset=Dataset(sliced_data, delete_class=data_kwargs.get('delete_class', None))
    dataset.set_nway_and_q(is_train=True, **data_kwargs['n_way_k_shot_kwargs']['train'])
    dataset.set_nway_and_q(is_train=False, **data_kwargs['n_way_k_shot_kwargs']['test'])
    
    dataset.split_by_sample(ratio=data_kwargs['process_kwargs']['rate'])
    
    return dataset

data_kwargs = {
    'data_path': 'data/SHL_dataset',
    'data_select_kwargs': {
        'speed': 'walking',
        'axis': 'acc_x'
    },
    'process_kwargs': {
        'slice_len': 100,
        'overlap': 50,
        'drop_last': True,
        'rate': 0.8
    },
    'n_way_k_shot_kwargs': {
        'train': {
            'n_way': 5,
            'k_shot': 5
        },
        'test': {
            'n_way': 5,
            'k_shot': 5
        }
    },
    'delete_class': None
}  


