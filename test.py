from OpenSrc.infer import Infer
from OpenSrc.utils.dataset import Dataset
from OpenSrc.utils.slice_data import SliceData
from OpenSrc.utils.load_data import LoadData
import torch





def  test(data,Infer):
    for speed_name, speed_data in data.items():
        for axis_name, axis_data in speed_data.items():
            # if axis_name != "Acc_Y" or speed_name != "r200":
            #     print(f"Skipping {speed_name} - {axis_name}")
            #     continue
            print(f"Testing on {speed_name} - {axis_name}")
            sliced_data=SliceData(axis_data,slice_len=1024,overlap=0.0,drop_last=True)
            sliced_axis_data=sliced_data.slice_signal_data()
            dataset=Dataset(sliced_axis_data)
            dataset.set_nway_and_q(is_train=True, n_way=5, k_shot=10, q_shot=20)
            dataset.set_nway_and_q(is_train=False, n_way=7, k_shot=10, q_shot=90)
            dataset.split_by_sample(ratio=0.8)

            # Infer.infer_evaluate(dataloader=dataset, n_way=7, num_episodes=10)
            Infer.draw(dataloader=dataset, n_way=7, num_episodes=10)


data_path=r"/home/hp/2025/liujunsong/FLI/dataset.pkl"







if __name__ == "__main__":

    #加载全量的数据
    data_load=LoadData(data_path)
    data=data_load.data

    infer=Infer(
        weight_path=r"./best_model.pth",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    
    test(data,infer)

    
