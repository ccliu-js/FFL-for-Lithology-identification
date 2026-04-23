import random
import torch
import numpy as np

class Dataset():
    def __init__(self, data,delete_class=None):
        self.data = data
        self.delete_class=delete_class
        self.train_n_way = None
        self.train_k_shot = None
        self.train_q_shot = None

        self.test_n_way = None
        self.test_k_shot = None
        self.test_q_shot = None

        self.label_dict = {
            'Arkose':0,
            'Sandstone':1,
            'Marble':2,
            'Granite':3,
            'Marlstone':4,
            'Limestone':5,
            'Shale':6
        }


    def split_by_sample(self,ratio=0.8,seed=42):
        random.seed(seed)
        train_data = {}
        test_data = {}
        for class_label, samples in self.data.items():
            random.shuffle(samples)
            split_point = int(len(samples) * ratio)
            train_data[class_label] = samples[:split_point]
            test_data[class_label] = samples[split_point:]
        

        #裁剪测试数据每类固定数量的样本
        class_test_size_per_class=200
        class_train_size_per_class=100
        for class_label, samples in test_data.items():
            if len(samples) > class_test_size_per_class:
                test_data[class_label] = samples[:class_test_size_per_class]
        for class_label, samples in train_data.items():
            if len(samples) > class_train_size_per_class:
                train_data[class_label] = samples[:class_train_size_per_class]


        if self.delete_class:
            if self.delete_class in train_data:
                del train_data[self.delete_class]
                print(f"已从训练数据中删除类别: {self.delete_class}")


        self.train_data = train_data
        self.test_data = test_data

        return train_data, test_data
    
    def set_nway_and_q(self, is_train, n_way, k_shot, q_shot):
        if is_train:
            self.train_n_way = n_way
            self.train_k_shot = k_shot
            self.train_q_shot = q_shot
        else:
            self.test_n_way = n_way
            self.test_k_shot = k_shot
            self.test_q_shot = q_shot


    def get_episode(self, is_train):
        if is_train:
            data = self.train_data
            n_way = self.train_n_way
            k_shot = self.train_k_shot
            q_shot = self.train_q_shot
        else:
            data = self.test_data
            n_way = self.test_n_way
            k_shot = self.test_k_shot
            q_shot = self.test_q_shot

        selected_classes = random.sample(list(data.keys()), n_way)
        support_x, support_y, query_x, query_y = [], [], [], []

        for i, class_label in enumerate(selected_classes):
            samples = data[class_label]
            selected_samples = random.sample(samples, k_shot + q_shot)
            support_samples = selected_samples[:k_shot]
            query_samples = selected_samples[k_shot:k_shot + q_shot]
            support_x.extend(support_samples)
            support_y.extend([i] * k_shot)
            query_x.extend(query_samples)
            query_y.extend([i] * q_shot)




        support_x = torch.stack([torch.as_tensor(x) for x in support_x]).float()
        query_x = torch.stack([torch.as_tensor(x) for x in query_x]).float()



        # 2. 处理标签数据 (Y): 转换为 Long 类型 (分类任务必备)
        support_y = torch.tensor(support_y).long()
        query_y = torch.tensor(query_y).long()
        return support_x, support_y, query_x, query_y


        


