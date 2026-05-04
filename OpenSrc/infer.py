import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from OpenSrc.NN.BayesianProtoNet import BayesianProtoNet


class Infer:
    def __init__(self, weight_path, device, model_config=None):
        model_config = model_config or {}
        self.model = BayesianProtoNet(
            scale=model_config.get("proto_net", {}).get("scale", 15.0),
            encoder_config=model_config.get("encoder"),
        )
        self.model.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )
        self.model.to(device)
        self.device = device

    def infer_evaluate(self, dataloader, n_way, num_episodes=50):
        """Evaluate few-shot episodes and report mean accuracy with a 95% CI."""
        self.model.eval()
        accs = []

        print(f"Starting Evaluation on {num_episodes} episodes...")

        with torch.no_grad():
            for i in range(num_episodes):
                sx, sy, qx, qy = dataloader.get_episode(is_train=False)
                sx = sx.to(self.device)
                sy = sy.to(self.device)
                qx = qx.to(self.device)
                qy = qy.to(self.device)

                all_logits = []
                for _ in range(10):
                    logits, _, _, _, _ = self.model(sx, sy, qx, n_way)
                    all_logits.append(logits)

                avg_logits = torch.stack(all_logits).mean(dim=0)
                preds = torch.argmax(avg_logits, dim=1)
                accuracy = (preds == qy).float().mean().item()
                accs.append(accuracy)

                if (i + 1) % 100 == 0:
                    print(
                        f"Episode [{i + 1}/{num_episodes}], "
                        f"Current Mean Acc: {np.mean(accs):.4f}"
                    )

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        confidence_interval = 1.96 * (std_acc / np.sqrt(num_episodes))

        print("-" * 30)
        print(f"Final Results ({n_way}-way {dataloader.test_k_shot}-shot):")
        print(f"Accuracy: {mean_acc:.4f} +/- {confidence_interval:.4f}")
        print("-" * 30)

        return mean_acc, confidence_interval

    def compute_confusion_matrix(
        self,
        support_x,
        support_y,
        query_x,
        query_y,
        n_way,
        idx2label=None,
        device=None,
        num_samples=10,
        plot=True,
        k_shot=5,
        k=5,
        aix="Acc_Z",
        speed="r200",
    ):
        self.model.eval()
        if device is None:
            device = self.device

        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

        with torch.no_grad():
            all_logits = []
            for _ in range(num_samples):
                logits, _, _, _, _ = self.model(
                    support_x,
                    support_y,
                    query_x,
                    n_way,
                )
                all_logits.append(logits)

            avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)
            preds = torch.argmax(avg_logits, dim=1)

        y_true = query_y.cpu().numpy()
        y_pred = preds.cpu().numpy()
        labels = list(range(n_way))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        overall_acc = np.trace(cm) / np.maximum(np.sum(cm), 1)
        class_acc = np.divide(
            cm.diagonal(),
            cm.sum(axis=1),
            out=np.zeros_like(cm.diagonal(), dtype=float),
            where=cm.sum(axis=1) != 0,
        )

        print(f"Overall Accuracy: {overall_acc:.4f}")

        if idx2label is None:
            class_names = [str(i) for i in labels]
        else:
            class_names = [idx2label[i] for i in labels]

        print("Per-Class Accuracy:")
        for i, acc in enumerate(class_acc):
            print(f"  {class_names[i]}: {acc:.4f}")

        if plot:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
            )

            plt.xlabel("Predicted rock class")
            plt.ylabel("True rock class")
            plt.title("Confusion Matrix")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(
                f"{aix}_{speed}_{n_way}way_{k}shot.png",
                dpi=600,
                bbox_inches="tight",
            )
            plt.close()

        return cm, overall_acc, class_acc
