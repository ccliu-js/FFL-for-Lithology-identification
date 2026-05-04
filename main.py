import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch

from OpenSrc.NN.BayesianProtoNet import BayesianProtoNet
from OpenSrc.NN.loss import BNNLoss
from OpenSrc.infer import Infer
from OpenSrc.train import Trainer
from OpenSrc.utils import prepare_data
from OpenSrc.utils.dataset import Dataset
from OpenSrc.utils.load_data import LoadData
from OpenSrc.utils.slice_data import SliceData


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def parse_scalar(value):
    value = value.strip()
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value.strip("\"'")


def load_simple_yaml(config_path):
    root = {}
    stack = [(-1, root)]

    with open(config_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue

            indent = len(line) - len(line.lstrip(" "))
            key, _, raw_value = line.strip().partition(":")

            while stack and indent <= stack[-1][0]:
                stack.pop()

            parent = stack[-1][1]
            if raw_value.strip():
                parent[key] = parse_scalar(raw_value)
            else:
                parent[key] = {}
                stack.append((indent, parent[key]))

    return root


def load_config(config_path):
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        return load_simple_yaml(config_path)


def resolve_device(device_name):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_task_data_config(config, task_name):
    common_data = config["common"].get("data", {})
    task_data = config.get(task_name, {}).get("data", {})
    return {
        "speed": task_data.get("speed", common_data.get("speed")),
        "axis": task_data.get("axis", common_data.get("axis")),
    }


def build_train_data_kwargs(config):
    common = config["common"]
    train = config["train"]
    data_config = get_task_data_config(config, "train")

    return {
        "data_path": common["paths"]["data"],
        "data_select_kwargs": {
            "speed": data_config["speed"],
            "axis": data_config["axis"],
        },
        "process_kwargs": {
            "slice_len": common["slicing"]["slice_len"],
            "overlap": common["slicing"]["overlap"],
            "drop_last": common["slicing"]["drop_last"],
            "rate": common["split"]["ratio"],
        },
        "n_way_k_shot_kwargs": train["episodes"],
        "delete_class": train["data"].get("delete_class"),
    }


def load_sliced_axis_data(data_path, speed, axis, slicing_config):
    data = LoadData(data_path).data
    selected_data = data[speed][axis]
    slicer = SliceData(
        selected_data,
        slice_len=slicing_config["slice_len"],
        overlap=slicing_config["overlap"],
        drop_last=slicing_config["drop_last"],
    )
    return slicer.slice_signal_data()


def set_support_and_query(data, support_size, query_size):
    support_x, support_y = [], []
    query_x, query_y = [], []
    label2idx = {label: i for i, label in enumerate(data.keys())}
    idx2label = {i: label for label, i in label2idx.items()}

    for label, samples in data.items():
        if len(samples) < support_size + 1:
            continue

        shuffled_samples = random.sample(samples, len(samples))
        support_samples = shuffled_samples[:support_size]
        query_samples = shuffled_samples[support_size:support_size + query_size]

        support_x.extend(support_samples)
        support_y.extend([label2idx[label]] * len(support_samples))
        query_x.extend(query_samples)
        query_y.extend([label2idx[label]] * len(query_samples))

    support_x = torch.tensor(support_x, dtype=torch.float32)
    support_y = torch.tensor(support_y, dtype=torch.long)
    query_x = torch.tensor(query_x, dtype=torch.float32)
    query_y = torch.tensor(query_y, dtype=torch.long)

    return support_x, support_y, query_x, query_y, idx2label


def build_model(model_config):
    model_config = model_config or {}
    return BayesianProtoNet(
        scale=model_config.get("proto_net", {}).get("scale", 15.0),
        encoder_config=model_config.get("encoder"),
    )


def run_train(config):
    common = config["common"]
    train_config = config["train"]
    opt_config = train_config["optimization"]
    scheduler_config = opt_config["scheduler"]

    set_seed(common["runtime"]["seed"])
    device = resolve_device(common["runtime"]["device"])
    dataset = prepare_data(build_train_data_kwargs(config))

    model = build_model(config.get("model"))
    loss = BNNLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_config["learning_rate"],
        weight_decay=opt_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_config["factor"],
        patience=scheduler_config["patience"],
        min_lr=scheduler_config["min_lr"],
    )

    trainer = Trainer(
        model=model,
        loss=loss,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        save_path=common["paths"]["weights"],
    )
    trainer.train(
        dataloader=dataset,
        num_epochs=opt_config["num_epochs"],
        num_episodes_per_epoch=opt_config["episodes_per_epoch"],
    )


def run_infer_multi(config):
    common = config["common"]
    infer_config = config["infer_multi"]
    data_config = get_task_data_config(config, "infer_multi")
    exp_config = infer_config["experiment"]

    set_seed(common["runtime"]["seed"])
    device = resolve_device(common["runtime"]["device"])
    data = load_sliced_axis_data(
        data_path=common["paths"]["data"],
        speed=data_config["speed"],
        axis=data_config["axis"],
        slicing_config=common["slicing"],
    )
    infer = Infer(common["paths"]["weights"], device, model_config=config.get("model"))

    overall_acc_list = []
    class_acc_list = []
    cm_list = []

    for repeat_idx in range(exp_config["repeat"]):
        print(f"\n========== Repeat {repeat_idx + 1}/{exp_config['repeat']} ==========")

        support_x, support_y, query_x, query_y, idx2label = set_support_and_query(
            data,
            support_size=exp_config["k_shot"],
            query_size=exp_config["query_size"],
        )

        cm, overall_acc, class_acc = infer.compute_confusion_matrix(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            n_way=exp_config["n_way"],
            idx2label=idx2label,
            device=device,
            num_samples=exp_config["monte_carlo_samples"],
            plot=exp_config["plot"],
            k_shot=exp_config["k_shot"],
            k=exp_config["figure_k_label"],
            aix=data_config["axis"],
            speed=data_config["speed"],
        )

        overall_acc_list.append(overall_acc)
        class_acc_list.append(class_acc)
        cm_list.append(cm)

    overall_acc_array = np.array(overall_acc_list)
    class_acc_array = np.array(class_acc_list)
    cm_array = np.array(cm_list)

    print("\n========== Final Result ==========")
    print(f"Repeat Times: {exp_config['repeat']}")
    print(f"Overall Accuracy Mean: {np.mean(overall_acc_array):.4f}")
    print(f"Overall Accuracy Variance: {np.var(overall_acc_array):.6f}")
    print(f"Overall Accuracy Std: {np.std(overall_acc_array):.4f}")

    print("\nEach Repeat Overall Accuracy:")
    for i, acc in enumerate(overall_acc_array):
        print(f"  Repeat {i + 1}: {acc:.4f}")

    mean_class_acc = np.mean(class_acc_array, axis=0)
    var_class_acc = np.var(class_acc_array, axis=0)
    std_class_acc = np.std(class_acc_array, axis=0)

    print("\nPer-Class Accuracy Mean / Variance / Std:")
    for i in range(exp_config["n_way"]):
        class_name = idx2label[i]
        print(
            f"  {class_name}: "
            f"mean={mean_class_acc[i]:.4f}, "
            f"var={var_class_acc[i]:.6f}, "
            f"std={std_class_acc[i]:.4f}"
        )

    print("\nMean Confusion Matrix:")
    print(np.mean(cm_array, axis=0))


def run_test_diff_speed(config):
    common = config["common"]
    test_config = config["test_diff_speed"]
    data_config = get_task_data_config(config, "test_diff_speed")

    set_seed(common["runtime"]["seed"])
    device = resolve_device(common["runtime"]["device"])
    data = LoadData(common["paths"]["data"]).data
    infer = Infer(common["paths"]["weights"], device, model_config=config.get("model"))

    for speed_name, speed_data in data.items():
        for axis_name, axis_data in speed_data.items():
            if axis_name != data_config["axis"]:
                continue

            print(f"Testing on {speed_name} - {axis_name}")
            sliced_data = SliceData(
                axis_data,
                slice_len=common["slicing"]["slice_len"],
                overlap=common["slicing"]["overlap"],
                drop_last=common["slicing"]["drop_last"],
            ).slice_signal_data()

            dataset = Dataset(sliced_data)
            dataset.set_nway_and_q(is_train=True, **test_config["episodes"]["train"])
            dataset.set_nway_and_q(is_train=False, **test_config["episodes"]["test"])
            dataset.split_by_sample(ratio=common["split"]["ratio"])

            infer.infer_evaluate(
                dataloader=dataset,
                n_way=test_config["evaluation"]["n_way"],
                num_episodes=test_config["evaluation"]["num_episodes"],
            )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run FLI training and evaluation experiments."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the YAML configuration file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Train the Bayesian ProtoNet model.")
    subparsers.add_parser(
        "infer-multi",
        help="Run repeated evaluation and draw confusion matrices.",
    )
    subparsers.add_parser(
        "test-diff-speed",
        help="Evaluate the checkpoint across different speeds.",
    )

    return parser


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)

    if args.command == "train":
        run_train(config)
    elif args.command == "infer-multi":
        run_infer_multi(config)
    elif args.command == "test-diff-speed":
        run_test_diff_speed(config)


if __name__ == "__main__":
    main()
