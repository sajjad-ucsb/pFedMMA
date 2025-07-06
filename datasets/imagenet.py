import os
import pickle
import random
from collections import defaultdict

from Dassl.dassl.data.datasets.base_dataset import Datum, DatasetBase
from Dassl.dassl.utils import mkdir_if_missing
from .oxford_pets import OxfordPets


class ImageNet(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_ImageNet.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.baseline_dir = os.path.join(self.dataset_dir, "baseline")
        mkdir_if_missing(self.split_fewshot_dir)

        from torchvision.datasets import ImageFolder
        from torchvision import transforms

        # Default preprocess
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        full_train = ImageFolder(os.path.join(self.image_dir, "train"), transform=preprocess)
        full_test = ImageFolder(os.path.join(self.image_dir, "val"), transform=preprocess)

        # Wrap in Datum
        def wrap_dataset(dataset):
            items = []
            for impath, label in dataset.samples:
                classname = dataset.classes[label]
                items.append(Datum(impath=impath, label=label, classname=classname))
            return items

        total_train = wrap_dataset(full_train)
        test = wrap_dataset(full_test)
        val = random.sample(total_train, int(0.2 * len(total_train)))  # optional 20% val
        train = list(set(total_train) - set(val))

        num_shots = cfg.DATASET.NUM_SHOTS
        backbone = cfg.MODEL.HEAD.NAME
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, backbone, f"shot_{num_shots}-seed_{seed}.pkl")
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        if cfg.DATASET.USERS > 0 and cfg.DATASET.SUBSAMPLE_CLASSES != 'all':
            federated_train_x = self.generate_federated_fewshot_dataset(
                train, num_shots=num_shots, num_users=cfg.DATASET.USERS,
                is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE
            )
            if cfg.DATASET.USEALL:
                federated_test_x = defaultdict(list)
                for idx in range(cfg.DATASET.USERS):
                    federated_test_x[idx] = test
            else:
                federated_test_x = self.generate_federated_dataset(
                    test, num_shots=num_shots, num_users=cfg.DATASET.USERS,
                    is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE
                )
        elif cfg.DATASET.USERS > 0 and cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_dataset(
                total_train, num_shots=num_shots, num_users=cfg.DATASET.USERS,
                is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE
            )
            federated_test_x = self.generate_federated_dataset(
                test, num_shots=num_shots, num_users=cfg.DATASET.USERS,
                is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE
            )
        elif cfg.DATASET.USERS > 0 and not cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_fewshot_dataset(
                total_train, num_shots=num_shots, num_users=cfg.DATASET.USERS,
                is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE
            )
            federated_test_x = self.generate_federated_dataset(
                test, num_shots=num_shots, num_users=cfg.DATASET.USERS,
                is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE
            )
        else:
            federated_train_x = None
            federated_test_x = None

        super().__init__(
            train_x=train, val=val, test=test,
            federated_train_x=federated_train_x, federated_test_x=federated_test_x
        )
