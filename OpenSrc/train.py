import numpy as np
import torch
from tqdm import tqdm
import numpy as np
import torch


        


class Trainer:
    def __init__(self, model, optimizer, loss, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.scheduler = scheduler

    def train(self, dataloader, num_epochs=100, num_episodes_per_epoch=400):
        print("Starting training...")
        self.model.to(self.device)

        best_acc = 0.0

        for epoch in tqdm(range(num_epochs), desc="Training"):
            epoch_total_losses = []
            epoch_CE_losses = []
            epoch_KL_weight_losses = []
            epoch_KL_latent_losses = []

            train_acc = 0.0
            train_total = 0

            for episode in tqdm(range(num_episodes_per_epoch), desc="Epoch Progress"):
                support_x, support_y, query_x, query_y = dataloader.get_episode(is_train=True)

                sx, sy = support_x.to(self.device), support_y.to(self.device)
                qx, qy = query_x.to(self.device), query_y.to(self.device)

                self.model.train()

                # ===== 1. 计算 loss =====
                total_loss, loss_ce, loss_kl_w, loss_kl_latent = self.loss.KL_and_CE_loss(
                    self.model, qx, qy, sx, sy, dataloader.train_n_way, num_samples=5
                )

                # ===== 2. NaN 检测 =====
                if torch.isnan(total_loss):
                    print(f"NaN detected at Epoch {epoch+1}, Episode {episode+1}")
                    return

                # ===== 3. 反向传播 =====
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # ===== 4. 记录 =====
                epoch_total_losses.append(total_loss.item())
                epoch_CE_losses.append(loss_ce.item())
                epoch_KL_weight_losses.append(loss_kl_w.item())
                epoch_KL_latent_losses.append(loss_kl_latent.item())

                # 计算训练集上的准确率
                with torch.no_grad():
                    logits, _, _, _, _ = self.model(sx, sy, qx, dataloader.train_n_way)
                    preds = torch.argmax(logits, dim=1)
                    train_correct = (preds == qy).sum().item()
                    train_acc += train_correct
                    train_total += qy.size(0)

            # ===== 5. 打印 =====
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print(f"Total Loss: {np.mean(epoch_total_losses):.4f}")
            print(f"CE Loss: {np.mean(epoch_CE_losses):.4f}")
            print(f"KL Weight: {np.mean(epoch_KL_weight_losses):.6f}")
            print(f"KL Latent: {np.mean(epoch_KL_latent_losses):.6f}")
            print(f"Train Acc: {train_acc / train_total:.4f}")

            # ===== 6. 验证 =====
            mean_acc, _ = self.evaluate(dataloader, dataloader.test_n_way, num_episodes=21)

            if mean_acc > best_acc:
                best_acc = mean_acc
                torch.save(self.model.state_dict(), r"best_model.pth")
                print(f"🔥 New best model saved! Acc: {best_acc:.4f}")

            if self.scheduler:
                self.scheduler.step(mean_acc)



    def evaluate(self, dataloader, n_way, num_episodes=10):
        self.model.eval()
        accs = []

        print(f"Starting Evaluation on {num_episodes} episodes...")

        with torch.no_grad():
            for i in tqdm(range(num_episodes), desc="Evaluation"):
                sx, sy, qx, qy = dataloader.get_episode(is_train=False)

                sx, sy = sx.to(self.device), sy.to(self.device)
                qx, qy = qx.to(self.device), qy.to(self.device)

                num_samples = 10
                all_logits = []

                for _ in range(num_samples):
                    logits, _, _, _, _ = self.model(sx, sy, qx, n_way)  # ✅ 改这里
                    all_logits.append(logits)

                avg_logits = torch.stack(all_logits).mean(dim=0)

                preds = torch.argmax(avg_logits, dim=1)
                accuracy = (preds == qy).float().mean().item()
                accs.append(accuracy)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        ci = 1.96 * (std_acc / np.sqrt(num_episodes))

        print(f"Final Results: {mean_acc:.4f} ± {ci:.4f}")

        return mean_acc, ci
    


