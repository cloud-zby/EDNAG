import os
import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import os
import os.path
from typing import Any
from PIL import Image
from meta_acc_predictor.download_datasets import download_aircraft, download_pets


Dataset2Class = {'cifar10': 10,
                 'cifar100': 100,
                 'mnist': 10,
                 'svhn': 10,
                 'aircraft': 30,
                 'pets': 37}

def make_dataset(dir, image_ids, targets):
  assert (len(image_ids) == len(targets))
  images = []
  dir = os.path.expanduser(dir)
  for i in range(len(image_ids)):
    item = (os.path.join(dir, 'data', 'images',
                         '%s.jpg' % image_ids[i]), targets[i])
    images.append(item)
  return images


def find_classes(classes_file):
  # read classes file, separating out image IDs and class names
  image_ids = []
  targets = []
  f = open(classes_file, 'r')
  for line in f:
    split_line = line.split(' ')
    image_ids.append(split_line[0])
    targets.append(' '.join(split_line[1:]))
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
  def __init__(self, root, train=True, num_cl=37, val_split=0.2, transforms=None):
    self.data = torch.load(os.path.join(root,'pets','{}{}.pth'.format('train' if train else 'test',
                                                               int(100*(1-val_split)) if train else int(100*val_split))))
    self.len = len(self.data)
    self.transform = transforms
    
    self.img_list = []
    self.label_list = []
    for i in range(len(self.data)):
        img, label = self.data[i]
        if self.transform:
            img = self.transform(img)
        self.img_list.append(img)
        self.label_list.append(label)

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
  url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
  class_types = ('variant', 'family', 'manufacturer')
  splits = ('train', 'val', 'trainval', 'test')
  
  def __init__(self, root, class_type='variant', split='train', transform=None,
               target_transform=None, loader=default_loader, download=False):
    if split not in self.splits:
      raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
        split, ', '.join(self.splits),
      ))
    if class_type not in self.class_types:
      raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
        class_type, ', '.join(self.class_types),
      ))
    self.root = os.path.expanduser(root)
    self.root = os.path.join(self.root, 'fgvc-aircraft-2013b')
    self.class_type = class_type
    self.split = split
    self.classes_file = os.path.join(self.root, 'data',
                                     'images_%s_%s.txt' % (self.class_type, self.split))

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
    for i in range(len(samples)):
        path, target = self.samples[i]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        self.sample_list.append(sample)
        self.target_list.append(target)
  
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
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Root Location: {}\n'.format(self.root)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    tmp = '    Target Transforms (if any): '
    fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    return fmt_str
  
  def _check_exists(self):
    return os.path.exists(os.path.join(self.root, 'data', 'images')) and \
           os.path.exists(self.classes_file)
  
  def download(self):
    """Download the FGVC-Aircraft data if it doesn't exist already."""
    from six.moves import urllib
    import tarfile
    
    if self._check_exists():
      return
    
    # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
    print('Downloading %s ... (may take a few minutes)' % self.url)
    parent_dir = os.path.abspath(os.path.join(self.root, os.pardir))
    tar_name = self.url.rpartition('/')[-1]
    tar_path = os.path.join(parent_dir, tar_name)
    data = urllib.request.urlopen(self.url)
    
    # download .tar.gz file
    with open(tar_path, 'wb') as f:
      f.write(data.read())
    
    # extract .tar.gz to PARENT_DIR/fgvc-aircraft-2013b
    data_folder = tar_path.strip('.tar.gz')
    print('Extracting %s to %s ... (may take a few minutes)' % (tar_path, data_folder))
    tar = tarfile.open(tar_path)
    tar.extractall(parent_dir)
    
    # if necessary, rename data folder to self.root
    if not os.path.samefile(data_folder, self.root):
      print('Renaming %s to %s ...' % (data_folder, self.root))
      os.rename(data_folder, self.root)
    
    # delete .tar.gz file
    print('Deleting %s ...' % tar_path)
    os.remove(tar_path)
    
    print('Done!')


class CUTOUT(object):

    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return ('{name}(length={length})'.format(name=self.__class__.__name__, **self.__dict__))

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def get_datasets(name, cutout, root='meta_acc_predictor/data'):
    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name.startswith('mnist'):
        mean, std = [0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]
    elif name.startswith('svhn'):
        mean, std = [0.4376821, 0.4437697, 0.47280442], [
            0.19803012, 0.20101562, 0.19703614]
    elif name.startswith('aircraft'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name.startswith('pets'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    # Data Argumentation
    if name == 'cifar10' or name == 'cifar100':
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                 transforms.Normalize(mean, std)]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 32, 32)
    elif name.startswith('cub200'):
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        xshape = (1, 3, 32, 32)
    elif name.startswith('mnist'):
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean, std)
        ])
        xshape = (1, 3, 32, 32)
    elif name.startswith('svhn'):
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        xshape = (1, 3, 32, 32)
    elif name.startswith('aircraft'):
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        xshape = (1, 3, 32, 32)
    elif name.startswith('pets'):
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        xshape = (1, 3, 32, 32)
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == 'cifar10':
        train_data = dset.CIFAR10(
            root, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(
            root, train=False, transform=test_transform, download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'cifar100':
        train_data = dset.CIFAR100(
            root, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(
            root, train=False, transform=test_transform, download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'mnist':
        train_data = dset.MNIST(
            root, train=True, transform=train_transform, download=True)
        test_data = dset.MNIST(
            root, train=False, transform=test_transform, download=True)
        assert len(train_data) == 60000 and len(test_data) == 10000
    elif name == 'svhn':
        train_data = dset.SVHN(root, split='train',
                               transform=train_transform, download=True)
        test_data = dset.SVHN(root, split='test',
                              transform=test_transform, download=True)
        assert len(train_data) == 73257 and len(test_data) == 26032
    elif name == 'aircraft':
        if os.path.exists('./meta_acc_predictor/data/fgvc-aircraft-2013b.tar.gz') == False:
            download_aircraft()
        train_data = FGVCAircraft(root, class_type='manufacturer', split='trainval',
                                  transform=train_transform, download=False)
        test_data = FGVCAircraft(root, class_type='manufacturer', split='test',
                                 transform=test_transform, download=False)
        assert len(train_data) == 6667 and len(test_data) == 3333
    elif name == 'pets':
        if os.path.exists('./meta_acc_predictor/data/pets/train85.pth') == False:
            download_pets()
        if os.path.exists('./meta_acc_predictor/data/pets/test15.pth') == False:
            download_pets()
        train_data = PetDataset(root, train=True, num_cl=37,
                                val_split=0.15, transforms=train_transform)
        test_data = PetDataset(root, train=False, num_cl=37,
                               val_split=0.15, transforms=test_transform)
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num

