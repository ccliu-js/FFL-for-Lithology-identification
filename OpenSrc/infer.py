from OpenSrc.NN.BayesianProtoNet import BayesianProtoNet

import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Infer:
    def __init__(self, weight_path,device):
        self.model = BayesianProtoNet()
        self.model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        self.model.to(device)
        self.device = device


    def set_support(self, support_data, support_labels,labels_dict):
        self.support_data = support_data.to(self.device)
        self.support_labels = support_labels.to(self.device)
        self.labels_dict = labels_dict



    def infer_with_labels(self, query_data, query_labels):
        query_data = query_data.to(self.device)
        query_labels = query_labels.to(self.device)
        with torch.no_grad():
            logits = self.model(self.support_data, self.support_labels, query_data)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu()
    



    def infer_single_sample(self, x_single, fixed_prototypes, n_samples=50):
        # x_single 形状为 [1, input_dim]
        x_single = x_single.to(self.device)
        all_probs = []

        with torch.no_grad():
            for _ in range(n_samples):
                # 提取单样本特征 (由于是贝叶斯，每次提取的特征会微小波动)
                z_query = self.model.backbone(x_single)
                z_query = F.normalize(z_query, p=2, dim=1)

                # 计算该样本与所有固定原型的距离
                # z_query: [1, latent_dim], fixed_prototypes: [n_way, latent_dim]
                dists = torch.cdist(z_query, fixed_prototypes, p=2)
                logits = -(dists**2)
                all_probs.append(F.softmax(logits, dim=-1))

        # 3. 统计贝叶斯结果
        mean_probs = torch.stack(all_probs).mean(0) # [1, n_way]
        confidence, pred_class = torch.max(mean_probs, dim=-1)
        uncertainty = torch.stack(all_probs).std(0).gather(1, pred_class.unsqueeze(1))

        return pred_class.item(), confidence.item(), uncertainty.item()





    def infer_evaluate(self, dataloader, n_way, num_episodes=50):
        """
        针对小样本任务进行指标评估
        num_episodes: 评估的任务总数，通常设为 600 或 1000 以保证统计显著性
        """
        self.model.eval() # 开启评估模式（关闭 Dropout, BatchNorm 等）
        accs = []

        print(f"Starting Evaluation on {num_episodes} episodes...")
        
        with torch.no_grad(): # 禁用梯度计算，节省内存并提速
            for i in range(num_episodes):
                # 1. 采样一个测试任务
                sx, sy, qx, qy = dataloader.get_episode(is_train=False)
                sx = sx.to(self.device)
                sy = sy.to(self.device)
                qx = qx.to(self.device)
                qy = qy.to(self.device)
                
                # 2. 这里的逻辑取决于你的模型 forward 怎么写的
                # 既然是 BNN，我们可以通过多次采样求平均预测来降低随机性
                num_samples = 10 # 评估时可以多采几次样
                all_logits = []
                for _ in range(num_samples):
                    logits, mu_s, logvar_s, mu_q, logvar_q = self.model(sx, sy, qx, n_way)
                    all_logits.append(logits)
                
                # 取平均 logits
                avg_logits = torch.stack(all_logits).mean(dim=0)
                
                # 3. 计算准确率
                preds = torch.argmax(avg_logits, dim=1)
                # 确保标签在同一设备
                qy = qy.to(self.device)
                accuracy = (preds == qy).float().mean().item()
                accs.append(accuracy)

                if (i + 1) % 100 == 0:
                    print(f"Episode [{i+1}/{num_episodes}], Current Mean Acc: {np.mean(accs):.4f}")



        # 4. 计算最终统计指标
        mean_acc = np.mean(accs)
        # 计算 95% 置信区间 (Confidence Interval)
        std_acc = np.std(accs)
        confidence_interval = 1.96 * (std_acc / np.sqrt(num_episodes))




        print("-" * 30)
        print(f"Final Results ({n_way}-way {dataloader.test_k_shot}-shot):")
        print(f"Accuracy: {mean_acc:.4f} ± {confidence_interval:.4f}")
        print("-" * 30)
        
        return mean_acc, confidence_interval
    







    def draw(self, dataloader, n_way, num_episodes=50):
        """
        针对小样本任务进行指标评估
        num_episodes: 评估的任务总数，通常设为 600 或 1000 以保证统计显著性
        """
        self.model.eval() # 开启评估模式（关闭 Dropout, BatchNorm 等）
        accs = []

        print(f"Starting Evaluation on {num_episodes} episodes...")
        sx, sy, qx, qy = dataloader.get_episode(is_train=False)
        sx = sx.to(self.device)
        sy = sy.to(self.device)
        qx = qx.to(self.device)
        qy = qy.to(self.device)
        
        with torch.no_grad(): # 禁用梯度计算，节省内存并提速
            for i in range(num_episodes):
                num_samples = 10 # 评估时可以多采几次样
                all_logits = []
                q_list=[]
                s_list=[]
                for _ in range(num_samples):
                    logits, mu_s, logvar_s, mu_q, logvar_q = self.model(sx, sy, qx, n_way)
                    all_logits.append(logits)

                
                # 取平均 logits
                avg_logits = torch.stack(all_logits).mean(dim=0)

                # #保存拿到的avg_logits
                # np.save(r"A:\lithology\experiment\.input\query_features.npy", np.stack(q_list))
                # np.save(r"A:\lithology\experiment\.input\support_features.npy", np.stack(s_list))
                # #保存sy和qy
                # np.save(r"A:\lithology\experiment\.input\support_labels.npy", sy.cpu().numpy())
                # np.save(r"A:\lithology\experiment\.input\query_labels.npy", qy.cpu().numpy())

                # #保存支持集的方差
                # np.save(r"A:\lithology\experiment\.input\support_logvar.npy", logvar_s.cpu().numpy())

                # return

                # 3. 计算准确率
                preds = torch.argmax(avg_logits, dim=1)
                # 确保标签在同一设备
                qy = qy.to(self.device)
                accuracy = (preds == qy).float().mean().item()
                accs.append(accuracy)

                if (i + 1) % 100 == 0:
                    print(f"Episode [{i+1}/{num_episodes}], Current Mean Acc: {np.mean(accs):.4f}")



        # 4. 计算最终统计指标
        mean_acc = np.mean(accs)
        # 计算 95% 置信区间 (Confidence Interval)
        std_acc = np.std(accs)
        confidence_interval = 1.96 * (std_acc / np.sqrt(num_episodes))




        print("-" * 30)
        print(f"Final Results ({n_way}-way {dataloader.test_k_shot}-shot):")
        print(f"Accuracy: {mean_acc:.4f} ± {confidence_interval:.4f}")
        print("-" * 30)
        
        return mean_acc, confidence_interval








    def compute_confusion_matrix(self,support_x,support_y,query_x,query_y,n_way,device="cuda",num_samples=1,plot=True,k_shot=5):
        self.model.eval()
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)
        with torch.no_grad():
            all_logits = []

            for _ in range(num_samples):
                logits, _, _, _, _ = self.model(support_x, support_y, query_x, n_way)
                all_logits.append(logits)

            # 平均 logits（如果 num_samples > 1）
            avg_logits = torch.stack(all_logits).mean(dim=0)

            preds = torch.argmax(avg_logits, dim=1)

        # 转 CPU + numpy
        y_true = query_y.cpu().numpy()
        y_pred = preds.cpu().numpy()

        cm = confusion_matrix(y_true, y_pred)
        #计算总体准确率
        overall_acc = np.trace(cm) / np.sum(cm)

        #计算每类准确率
        class_acc = cm.diagonal() / cm.sum(axis=1)

        #计算F1分数
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"F1 Score: {f1:.4f}")


        print(f"Overall Accuracy: {overall_acc:.4f}")
        print("Per-Class Accuracy:")
        for i, acc in enumerate(class_acc):
            print(f"  Class {i}: {acc:.4f}")
        if plot:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", vmin=0, vmax=cm.max()*0.6)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{n_way}-way {k_shot}-shot Confusion Matrix")
            plt.show()

        return cm