import os
import tqdm
import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import os
import os.path
from typing import Any
from PIL import Image
import random
import platform
import torchvision.utils
from .download_datasets import download_aircraft, download_pets
from .autoaugment import CIFAR10Policy

import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch


class ProcessedDataset(torch.utils.data.Dataset):
    """
    Not used in the current implementation.
    It can reduce the time to load the dataset by pre-processing the dataset and saving it to a file.
    But it costs too much memory to store the pre-processed dataset in CPU, often leading to memory errors.
    """
    def __init__(self, dataset, save_path:str):
        self.dataset = dataset
        self.save_path = save_path
        self.data = []
        self.targets = []
        if os.path.exists(save_path):
            preprocessed_dataset = torch.load(save_path)
            self.data, self.targets = preprocessed_dataset['data'], preprocessed_dataset['targets']
        else:
            self.initialize()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        return img, target

    def initialize(self):
        for img, target in tqdm.tqdm(self.dataset, ncols=120, desc="Preprocessing dataset"):
            self.data.append(img)
            self.targets.append(target)
        torch.save({'data': self.data, 'targets': self.targets}, self.save_path)


def make_dataset(dir, image_ids, targets):
    assert len(image_ids) == len(targets)
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (
            os.path.join(dir, "data", "images", "%s.jpg" % image_ids[i]),
            targets[i],
        )
        images.append(item)
    return images


def find_classes(classes_file):
    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, "r")
    for line in f:
        split_line = line.split(" ")
        image_ids.append(split_line[0])
        targets.append(" ".join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class PetDataset(Dataset):
    def __init__(self, root, save_path, train=True, num_cl=37, val_split=0.2, transforms=None):
        self.data = torch.load(
            os.path.join(
                root,
                "{}{}.pth".format(
                    "train" if train else "test",
                    int(100 * (1 - val_split)) if train else int(100 * val_split),
                ),
            )
        )
        self.is_train = train
        self.len = len(self.data)
        self.transform = transforms
        self.save_path = save_path
        self.img_list = []
        self.label_list = []
        if not os.path.exists(save_path):
            self.preprocess()
        else:
            preprocessed_dataset = torch.load(save_path)
            self.img_list, self.label_list = preprocessed_dataset['data'], preprocessed_dataset['targets']

    def preprocess(self):
        print(f">>> Pre-processing pets dataset for {'train' if self.is_train else 'test'}.")
        bar = tqdm.tqdm(range(len(self.data)), ncols=120)
        for i in bar:
            img, label = self.data[i]
            if self.transform:
                img = self.transform(img)
            self.img_list.append(img)
            self.label_list.append(label)
        torch.save({'data': self.img_list, 'targets': self.label_list}, self.save_path)

    def __getitem__(self, index):
        img, label = self.img_list[index], self.label_list[index]
        return img, label

    def __len__(self):
        return self.len


class FGVCAircraft(Dataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft>`_ Dataset.
    Args:
        root (string): Root directory path to dataset.
        class_type (string, optional): The level of FGVC-Aircraft fine-grain classification
            to label data with (i.e., ``variant``, ``family``, or ``manufacturer``).
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    url = "http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
    class_types = ("variant", "family", "manufacturer")
    splits = ("train", "val", "trainval", "test")

    def __init__(
        self,
        root,
        save_path,
        class_type="variant",
        split="train",
        transform=None,
        target_transform=None,
        loader=default_loader,
        download=False,
    ):
        if split not in self.splits:
            raise ValueError(
                'Split "{}" not found. Valid splits are: {}'.format(
                    split,
                    ", ".join(self.splits),
                )
            )
        if class_type not in self.class_types:
            raise ValueError(
                'Class type "{}" not found. Valid class types are: {}'.format(
                    class_type,
                    ", ".join(self.class_types),
                )
            )
        self.root = os.path.expanduser(root)
        self.root = os.path.join(self.root, "fgvc-aircraft-2013b")
        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(
            self.root, "data", "images_%s_%s.txt" % (self.class_type, self.split)
        )

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = find_classes(self.classes_file)
        samples = make_dataset(self.root, image_ids, targets)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.sample_list = []
        self.target_list = []
        self.save_path = save_path
        if os.path.exists(save_path):
            preprocessed_dataset = torch.load(save_path)
            self.sample_list, self.target_list = preprocessed_dataset['data'], preprocessed_dataset['targets']
        else:
            self.preprocess()

    def preprocess(self):
        print(f">>> Pre-processing aircraft dataset for {self.split}.")
        bar = tqdm.tqdm(range(len(self.samples)), ncols=120)
        for i in bar:
            path, target = self.samples[i]
            sample = pil_loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            self.sample_list.append(sample)
            self.target_list.append(target)
        torch.save({'data': self.sample_list, 'targets': self.target_list}, self.save_path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.sample_list[index]
        target = self.target_list[index]
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.root, "data", "images")
        ) and os.path.exists(self.classes_file)

    def download(self):
        """Download the FGVC-Aircraft data if it doesn't exist already."""
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print("Downloading %s ... (may take a few minutes)" % self.url)
        parent_dir = os.path.abspath(os.path.join(self.root, os.pardir))
        tar_name = self.url.rpartition("/")[-1]
        tar_path = os.path.join(parent_dir, tar_name)
        data = urllib.request.urlopen(self.url)

        # download .tar.gz file
        with open(tar_path, "wb") as f:
            f.write(data.read())

        # extract .tar.gz to PARENT_DIR/fgvc-aircraft-2013b
        data_folder = tar_path.strip(".tar.gz")
        print(
            "Extracting %s to %s ... (may take a few minutes)" % (tar_path, data_folder)
        )
        tar = tarfile.open(tar_path)
        tar.extractall(parent_dir)

        # if necessary, rename data folder to self.root
        if not os.path.samefile(data_folder, self.root):
            print("Renaming %s to %s ..." % (data_folder, self.root))
            os.rename(data_folder, self.root)

        # delete .tar.gz file
        print("Deleting %s ..." % tar_path)
        os.remove(tar_path)

        print("Done!")


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def get_dataset(
    data_name,
    batch_size,
    img_size,
    autoaugment,
    cutout,
    cutout_length,
    data_path="./eval_architecture/datasets/data/",
):
    # 数据集预处理
    train_transform, valid_transform = _data_transforms(
        data_name, img_size, autoaugment, cutout, cutout_length
    )
    # 加载dataset
    if data_name == "cifar100":
        train_data = torchvision.datasets.CIFAR100(
            root=data_path + 'cifar100', train=True, download=True, transform=train_transform
        )
        # train_data = ProcessedDataset(
        #     dataset=train_data, save_path="eval_architecture/datasets/data/processed/processed_cifar100_train.pth"
        # )
        valid_data = torchvision.datasets.CIFAR100(
            root=data_path + "cifar100",
            train=False,
            download=True,
            transform=valid_transform,
        )
        # valid_data = ProcessedDataset(
        #     dataset=valid_data, save_path="eval_architecture/datasets/data/processed/processed_cifar100_test.pth"
        # )
        num_class = 100
    elif data_name == "cifar10":
        train_data = torchvision.datasets.CIFAR10(
            root=data_path + "cifar10",
            train=True,
            download=True,
            transform=train_transform,
        )
        # train_data = ProcessedDataset(
        #     dataset=train_data, save_path="eval_architecture/datasets/data/processed/processed_cifar10_train.pth"
        # )
        valid_data = torchvision.datasets.CIFAR10(
            root=data_path + "cifar10",
            train=False,
            download=True,
            transform=valid_transform,
        )
        # valid_data = ProcessedDataset(
        #     dataset=valid_data, save_path="eval_architecture/datasets/data/processed/processed_cifar10_test.pth"
        # )
        num_class = 10
    elif data_name.startswith("aircraft"):
        """
        按照manufacturer进行划分，将飞机分为不同的具体型号，可分为30个类别（如波音787-900、波音787-1000）
        按照family进行划分，将飞机分为不同的系列或类别，可分为70个类别（如空客350-1000、空客330-900）
        按照variant进行划分，将飞机分为不同的制造商，可分为30个类别（如波音、空客、巴航工）
        """
        class_type = "variant"
        train_data = FGVCAircraft(
            root=data_path + "aircraft",
            save_path="eval_architecture/datasets/data/processed/processed_aircraft_train.pth",
            class_type=class_type,
            split="trainval",
            transform=train_transform,
            download=True,
        )
        valid_data = FGVCAircraft(
            root=data_path + "aircraft",
            save_path="eval_architecture/datasets/data/processed/processed_aircraft_test.pth",
            class_type=class_type,
            split="test",
            transform=valid_transform,
            download=True,
        )
        if class_type == "manufacturer":
            num_class = 30
        elif class_type == "family":
            num_class = 70
        elif class_type == "variant":
            num_class = 100
    elif data_name.startswith("pets"):
        train_data = PetDataset(
            root=data_path + "pets",
            save_path="eval_architecture/datasets/data/processed/processed_pets_train.pth",
            train=True,
            num_cl=37,
            val_split=0.15,
            transforms=train_transform,
        )
        valid_data = PetDataset(
            root=data_path + "pets",
            save_path="eval_architecture/datasets/data/processed/processed_pets_test.pth",
            train=False,
            num_cl=37,
            val_split=0.15,
            transforms=valid_transform,
        )
        num_class = 37
    else:
        raise KeyError

    # 加载dataloader
    system = platform.system()
    if system == "Windows":
        print("Windows system detected. Using single process data loader.")
        train_queue = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        valid_queue = torch.utils.data.DataLoader(
            valid_data,
            batch_size=200,
            shuffle=False,
            pin_memory=True,
        )
    elif system == "Linux":
        print("Linux system detected. Using multi-process data loader.")
        train_queue = torch.utils.data.DataLoader(
            train_data,
            num_workers=4,  # 设置加载数据的子进程数量
            prefetch_factor=3,  # 每个子进程预取的批次数量
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        valid_queue = torch.utils.data.DataLoader(
            valid_data,
            num_workers=4,
            prefetch_factor=3,
            batch_size=200,
            shuffle=False,
            pin_memory=True,
        )
    else:
        raise ValueError("Unsupported OS: {}".format(system))
    return train_queue, valid_queue, num_class


def _data_transforms(data_name, img_size, autoaugment, cutout, cutout_length):
    if "cifar" in data_name:
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]
    elif "aircraft" in data_name:
        norm_mean = [0.48933587508932375, 0.5183537408957618, 0.5387914411673883]
        norm_std = [0.22388883112804625, 0.21641635409388751, 0.24615605842636115]
    elif "pets" in data_name:
        norm_mean = [0.4828895122298728, 0.4448394893850807, 0.39566558230789783]
        norm_std = [0.25925664613996574, 0.2532760018681693, 0.25981017205097917]
    else:
        raise KeyError

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size), interpolation=Image.BICUBIC, antialias=True
            ),  # BICUBIC interpolation
            transforms.RandomHorizontalFlip(),
        ]
    )

    if autoaugment:
        train_transform.transforms.append(CIFAR10Policy())

    train_transform.transforms.append(transforms.ToTensor())

    if cutout:
        train_transform.transforms.append(Cutout(cutout_length))

    train_transform.transforms.append(transforms.Normalize(norm_mean, norm_std))

    valid_transform = transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size), interpolation=Image.BICUBIC, antialias=True
            ),  # BICUBIC interpolation
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    return train_transform, valid_transform
