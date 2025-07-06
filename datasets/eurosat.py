import os
import pickle
from collections import defaultdict

from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum
from Dassl.dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

template = ['a photo of a {}.']

NEW_CLASSNAMES = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake'
}

# @DATASET_REGISTRY.register()
class EuroSAT(DatasetBase):
    dataset_dir = 'eurosat'
    template = template

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '2750')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_EuroSAT.json')
        self.split_fewshot_dir = os.path.join(self.dataset_dir, 'split_fewshot')
        self.baseline_dir = os.path.join(self.dataset_dir, 'baseline')
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            total_train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            total_train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CLASSNAMES)
            OxfordPets.save_split(total_train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        backbone = cfg.MODEL.HEAD.NAME
        seed = cfg.SEED

        if num_shots >= 1:
            if cfg.TRAINER.NAME == "Baseline":
                preprocessed = os.path.join(self.baseline_dir, backbone, f"shot_{num_shots}-seed_{seed}.pkl")
            else:
                preprocessed = os.path.join(self.split_fewshot_dir, backbone, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(total_train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Update classnames to human-friendly ones
        train = self.update_classname(train)
        val = self.update_classname(val)
        test = self.update_classname(test)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        if cfg.DATASET.USERS > 0:
            if cfg.DATASET.SUBSAMPLE_CLASSES != 'all':
                federated_train_x = self.generate_federated_fewshot_dataset(train, num_shots=num_shots,
                                                                            num_users=cfg.DATASET.USERS,
                                                                            is_iid=cfg.DATASET.IID,
                                                                            repeat_rate=cfg.DATASET.REPEATRATE)
                if cfg.DATASET.USEALL:
                    federated_test_x = defaultdict(list)
                    for idx in range(cfg.DATASET.USERS):
                        federated_test_x[idx] = test
                else:
                    federated_test_x = self.generate_federated_dataset(test, num_shots=num_shots,
                                                                       num_users=cfg.DATASET.USERS,
                                                                       is_iid=cfg.DATASET.IID,
                                                                       repeat_rate=cfg.DATASET.REPEATRATE)
            else:
                federated_train_x = self.generate_federated_dataset(train, num_shots=num_shots,
                                                                    num_users=cfg.DATASET.USERS,
                                                                    is_iid=cfg.DATASET.IID,
                                                                    repeat_rate=cfg.DATASET.REPEATRATE)
                federated_test_x = self.generate_federated_dataset(test, num_shots=num_shots,
                                                                   num_users=cfg.DATASET.USERS,
                                                                   is_iid=cfg.DATASET.IID,
                                                                   repeat_rate=cfg.DATASET.REPEATRATE)


        else:
            federated_train_x = None
            federated_test_x = None

        super().__init__(train_x=train, federated_train_x=federated_train_x,
                         val=val, federated_test_x=federated_test_x, test=test)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CLASSNAMES.get(cname_old, cname_old)
            item_new = Datum(
                impath=item_old.impath,
                label=item_old.label,
                classname=cname_new
            )
            dataset_new.append(item_new)
        return dataset_new
