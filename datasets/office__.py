import os
from data_utils import prepare_data_office

class Office:
    dataset_dir = "office_caltech_10"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.num_classes = 10

        # Leave-One-Domain-Out is always enabled
        target_domain = cfg.DATASET.TARGET_DOMAIN.lower()
        print(f"LODO setting: Target domain for testing is '{target_domain}'")

        train_set, test_set, classnames, lab2cname = prepare_data_office(cfg, root)

        self.federated_train_x = train_set
        self.federated_test_x = test_set
        self.lab2cname = lab2cname
        self.classnames = classnames
