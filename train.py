from OpenSrc.utils import prepare_data
from OpenSrc.NN.BayesianProtoNet import BayesianProtoNet

from OpenSrc.NN.loss import BNNLoss
from OpenSrc.train import Trainer

import torch


data_kwargs={
    'data_path': r"/home/hp/2025/liujunsong/FLI/dataset.pkl",
    'data_select_kwargs': {
        'speed': 'r200', 
        'axis': 'Acc_Y'
    },  
    'process_kwargs': {
        'slice_len': 1024,
        'overlap': 0.0,
        'drop_last': True,
        'rate': 0.8,
    },
    'n_way_k_shot_kwargs': {
        'train': {'n_way': 5, 'k_shot': 10, 'q_shot': 10},
        'test': {'n_way': 7, 'k_shot': 20, 'q_shot': 20}
    },
    'delete_class': None #"Marstone"
}







if __name__ == "__main__":
    dataset = prepare_data(data_kwargs)

    model=BayesianProtoNet()
    loss=BNNLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
    

    trainer=Trainer(model=model, 
                    loss=loss,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    optimizer=optimizer,
                    scheduler=scheduler
                    )
    
    trainer.train(dataloader=dataset, num_epochs=80, num_episodes_per_epoch=40)
