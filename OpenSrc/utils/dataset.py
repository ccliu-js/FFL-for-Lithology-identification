import random

import torch


class Dataset:
    def __init__(self, data, delete_class=None):
        self.data = data
        self.delete_class = delete_class

        self.train_n_way = None
        self.train_k_shot = None
        self.train_q_shot = None

        self.test_n_way = None
        self.test_k_shot = None
        self.test_q_shot = None

    def split_by_sample(self, ratio=0.8, seed=42):
        random.seed(seed)
        train_data = {}
        test_data = {}

        for class_label, samples in self.data.items():
            random.shuffle(samples)
            split_point = int(len(samples) * ratio)
            train_data[class_label] = samples[:split_point]
            test_data[class_label] = samples[split_point:]

        class_test_size_per_class = 200
        class_train_size_per_class = 100

        for class_label, samples in test_data.items():
            if len(samples) > class_test_size_per_class:
                test_data[class_label] = samples[:class_test_size_per_class]

        for class_label, samples in train_data.items():
            if len(samples) > class_train_size_per_class:
                train_data[class_label] = samples[:class_train_size_per_class]

        if self.delete_class:
            for cls in self.delete_class:
                del train_data[cls]


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
        support_y = torch.tensor(support_y).long()
        query_y = torch.tensor(query_y).long()

        return support_x, support_y, query_x, query_y
