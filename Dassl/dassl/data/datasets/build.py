from Dassl.dassl.utils import Registry, check_availability
# from datasets.caltech101 import Caltech101
from datasets.cifar100 import Cifar100
from datasets.cifar10 import Cifar10
from datasets.oxford_flowers import OxfordFlowers
from datasets.oxford_pets import OxfordPets
from datasets.food101 import Food101
from datasets.dtd import DescribableTextures
from datasets.domainnet import DomainNet
from datasets.office import Office
from datasets.ucf101 import UCF101
from datasets.imagenet import ImageNet
from datasets.sun397 import SUN397
from datasets.fgvc import FGVC
from datasets.eurosat import EuroSAT
# from datasets.digits import Digits

DATASET_REGISTRY = Registry("DATASET")
# DATASET_REGISTRY.register(Caltech101)
DATASET_REGISTRY.register(Cifar100)
DATASET_REGISTRY.register(Cifar10)
DATASET_REGISTRY.register(OxfordFlowers)
DATASET_REGISTRY.register(OxfordPets)
DATASET_REGISTRY.register(Food101)
DATASET_REGISTRY.register(DescribableTextures)
DATASET_REGISTRY.register(DomainNet)
DATASET_REGISTRY.register(Office)
DATASET_REGISTRY.register(UCF101)
DATASET_REGISTRY.register(ImageNet)
DATASET_REGISTRY.register(SUN397)
DATASET_REGISTRY.register(FGVC)
DATASET_REGISTRY.register(EuroSAT)
# DATASET_REGISTRY.register(Digits)

def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)
