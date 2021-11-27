import numpy as np
import os
import cv2
import json
import torch
import torchvision


class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class BaseDataLoader:
    def __init__(self, batch_size=1, train=True, shuffle=True, drop_last=False):
        pass

    def get_loader(self, loader, prob):
        raise NotImplementedError

    def get_labels(self, task):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def num_channels(self):
        raise NotImplementedError

    @property
    def num_classes_single(self):
        raise NotImplementedError

    @property
    def num_classes_multi(self):
        raise NotImplementedError


class CIFAR10Loader(BaseDataLoader):
    def __init__(self, batch_size=128, train=True, shuffle=True, drop_last=False):
        super(CIFAR10Loader, self).__init__(batch_size, train, shuffle, drop_last)
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )

        dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                               download=True, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      drop_last=drop_last)
        self.task_dataloader = None

        self._len = 50000 if train else 10000
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last


    def _create_TaskDataLoaders(self):
        images = []
        labels = []

        for batch_images, batch_labels in self.dataloader:
            for i in batch_images:
                images.append(i)
            for l in batch_labels:
                labels.append(l)

        self.task_dataloader = []
        for t in range(10):
            #labels is a LongTensor (c==t).long() converts ByteTensor to LongTensor.
            # creates a sparse Tensor of 1s and 0s. 1 means true for this task.
            dataset = CustomDataset(data=images.copy(), labels=[(c == t).long() for c in labels])
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=self.shuffle,
                                                     drop_last=self.drop_last)
            self.task_dataloader.append(dataloader)


    def get_loader(self, loader='standard', prob='uniform'):
        if loader == 'standard':
            return self.dataloader

        if self.task_dataloader is None:
            self._create_TaskDataLoaders()

        if loader == 'multi-task':
            return MultiTaskDataLoader(self.task_dataloader, prob)
        else:
            assert loader in list(range(10)), 'Unknown loader: {}'.format(loader)
            return self.task_dataloader[loader]


    def get_labels(self, task='standard'):
        if task == 'standard':
            return list(range(10))
        else:
            assert task in list(range(10)), 'Unknown task: {}'.format(task)
            labels = [0 for _ in range(10)]
            labels[task] = 1
            return labels


    def __iter__(self):
        return iter(self.dataloader)


    def __len__(self):
        return self._len


    @property
    def num_channels(self):
        return 3


    @property
    def num_classes_single(self):
        return 10


    @property
    def num_classes_multi(self):
        return [2 for _ in range(10)]


class CIFAR100Loader(BaseDataLoader):
    def __init__(self, batch_size=128, train=True, shuffle=True, drop_last=False):
        super(CIFAR100Loader, self).__init__(batch_size, train, shuffle, drop_last)
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
        )

        dataset = torchvision.datasets.CIFAR100(root='./data', train=train,
                                               download=True, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      drop_last=drop_last)
        self.task_dataloader = None
        self.labels = None

        self._len = 50000 if train else 10000
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last


    def _create_TaskDataLoaders(self):
        with open('CIFAR100_fine2coarse.json', 'r') as f:
            data_info = json.load(f)

        images = [[] for _ in range(20)]
        labels = [[] for _ in range(20)]

        for batch_images, batch_labels in self.dataloader:
            for i, l in zip(batch_images, batch_labels):
                images[data_info['task'][l]].append(i)
                labels[data_info['task'][l]].append(data_info['subclass'][l])

        self.task_dataloader = []
        for task_images, task_labels in zip(images, labels):
            dataset = CustomDataset(data=task_images, labels=task_labels)
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=self.shuffle,
                                                     drop_last=self.drop_last)
            self.task_dataloader.append(dataloader)


    def get_loader(self, loader='standard', prob='uniform'):
        if loader == 'standard':
            return self.dataloader

        if self.task_dataloader is None:
            self._create_TaskDataLoaders()

        if loader == 'multi-task':
            return MultiTaskDataLoader(self.task_dataloader, prob)
        else:
            assert loader in list(range(20)), 'Unknown loader: {}'.format(loader)
            return self.task_dataloader[loader]


    def _create_labels(self):
        with open('CIFAR100_fine2coarse.json', 'r') as f:
            data_info = json.load(f)

        self.labels = [[] for _ in range(20)]
        for i, t in enumerate(data_info['task']):
            self.labels[t].append(i)


    def get_labels(self, task='standard'):
        if task == 'standard':
            return list(range(100))
        else:
            assert task in list(range(20)), 'Unknown task: {}'.format(task)
            if self.labels is None:
                self._create_labels()
            return self.labels[task]


    def __iter__(self):
        return iter(self.dataloader)


    def __len__(self):
        return self._len


    @property
    def num_channels(self):
        return 3


    @property
    def num_classes_single(self):
        return 100


    @property
    def num_classes_multi(self):
        return [5 for _ in range(20)]


class OmniglotLoader(BaseDataLoader):
    def __init__(self, batch_size=128, train=True, shuffle=True, drop_last=False):
        super(OmniglotLoader, self).__init__(batch_size, train, shuffle, drop_last)
        omniglot_path = './data/omniglot'

        if os.path.isdir(omniglot_path):
            print('Files already downloaded and verified')
        else:
            raise FileNotFoundError('Omniglot dataset not found. Please download it and put it under \'{}\''.format(omniglot_path))

        images = []
        labels = []
        self._len = 0
        self.task_dataloader = []
        self.num_classes = []

        for p in [os.path.join(omniglot_path, 'images_background'), os.path.join(omniglot_path, 'images_evaluation')]:
            for task_path in sorted(os.listdir(p)):
                task_path = os.path.join(p, task_path)
                task_images = []
                task_labels = []
                for i, cls_path in enumerate(sorted(os.listdir(task_path))):
                    cls_path = os.path.join(task_path, cls_path)
                    ims = [cv2.imread(os.path.join(cls_path, filename), cv2.IMREAD_GRAYSCALE) / 255 for filename in sorted(os.listdir(cls_path))]

                    if train:
                        ims = ims[:int(len(ims)*0.8)]
                    else:
                        ims = ims[int(len(ims)*0.8):]

                    self._len += len(ims)
                    task_images += ims
                    task_labels += [i for _ in range(len(ims))]

                task_images = np.expand_dims(task_images, 1)
                dataset = CustomDataset(data=torch.Tensor(task_images).float(), labels=torch.Tensor(task_labels).long())
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=batch_size,
                                                         shuffle=shuffle,
                                                         drop_last=drop_last)
                self.task_dataloader.append(dataloader)

                self.num_classes.append(len(np.unique(task_labels)))

                images.append(task_images)
                labels.append(task_labels)

        images = np.concatenate(images)
        labels = np.concatenate(labels)

        new_label = 0
        new_labels = [new_label]
        for prev_label, label in zip(labels[:-1], labels[1:]):
            if prev_label != label:
                new_label += 1
            new_labels.append(new_label)

        new_labels = torch.Tensor(new_labels).long()
        images = torch.from_numpy(images).float()

        dataset = CustomDataset(data=images, labels=new_labels)
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      drop_last=drop_last)

        self.labels = []
        cnter = 0
        for num_classes in self.num_classes:
            self.labels.append(list(range(cnter, cnter + num_classes)))
            cnter += num_classes

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last


    def get_loader(self, loader='standard', prob='uniform'):
        if loader == 'standard':
            return self.dataloader

        if loader == 'multi-task':
            return MultiTaskDataLoader(self.task_dataloader, prob)
        else:
            assert loader in list(range(50)), 'Unknown loader: {}'.format(loader)
            return self.task_dataloader[loader]


    def get_labels(self, task='standard'):
        if task == 'standard':
            return list(range(50))
        else:
            assert task in list(range(50)), 'Unknown task: {}'.format(task)
            return self.labels[task]


    def __iter__(self):
        return iter(self.dataloader)


    def __len__(self):
        return self._len


    @property
    def num_channels(self):
        return 1


    @property
    def num_classes_single(self):
        return sum(self.num_classes)


    @property
    def num_classes_multi(self):
        return self.num_classes


class MultiTaskDataLoader:
    def __init__(self, dataloaders, prob='uniform'):
        self.dataloaders = dataloaders
        self.iters = [iter(loader) for loader in self.dataloaders]

        if prob == 'uniform':
            self.prob = np.ones(len(self.dataloaders)) / len(self.dataloaders)
        else:
            self.prob = prob

        self.size = sum([len(d) for d in self.dataloaders])
        self.step = 0


    def __iter__(self):
        return self


    def __next__(self):
        if self.step >= self.size:
            self.step = 0
            raise StopIteration

        task = np.random.choice(list(range(len(self.dataloaders))), p=self.prob)

        try:
            data, labels = self.iters[task].__next__()
        except StopIteration:
            self.iters[task] = iter(self.dataloaders[task])
            data, labels = self.iters[task].__next__()

        self.step += 1

        return data, labels, task

class MultiTaskSequentialDataLoader:
    def __init__(self, dataloaders, task_mixing_ratio):
        self.dataloaders = dataloaders
        self.iters = [iter(loader) for loader in self.dataloaders]

        #if the TMR is 0, all tasks will be trained sequentially (i.e AAA...BBB...CCC...DDD...)
        #if the TMR is 1, all tasks will be trained in reverse order (i.e ...DDD...CCC...BBB...AAA)
        #if TMR is 0.5, tasks will be mixed perfectly (i.e ABCD..ABCD...ABCD...)
        self.task_mixing_ratio = task_mixing_ratio

        #num of images in each class in dataset
        self.task_sizes = [len(d) for d in self.dataloaders]

        #num of tasks
        self.task_count = len(task_sizes)

        # num of images in entire dataset
        self.size = sum(task_sizes)

        #convert TMR to prob distribution
        self.probs = []
        self.__set_task_probs__()

        self.step = 0

    #returns a list of probabilities, one for each task, given the TMR
    def __set_task_probs__():
        self.probs = []
        #tmr between 0 and 1
        
        #for now, will only consider 0, 0.25, 0.5, 0.75, and 1
        bucket = self.task_mixing_ratio // 0.25
        #bucket 0 = 0, 1 = 0.25, 2 = 0.5, 3 = 0.75, 4 = 1, other = 0.5

        if (bucket == 0):
            self.probs = [1]
            for i in range(1, self.task_count):
                self.probs.append(0)
        elif (bucket == 1):
            #half decreasing (should sum to roughly 1)
            self.probs = [1/(2**i) for i in range(1, self.task_count + 1)]
        elif (bucket == 3):
            self.probs = [1/(2**i) for i in range(1, self.task_count + 1)]
            self.probs.reverse()
        elif (bucket == 4):
            for i in range(0, self.task_count-1):
                self.probs.append(0)
            self.probs.append(1)
        else: #bucket = 2 or other (default)
            self.probs = [1/(self.task_count) for _ in range(self.task_count)]

    def __iter__(self):
        return self

    def __next__(self):
        if self.step >= self.size:
            self.step = 0
            raise StopIteration

        task = np.random.choice(list(range(len(self.dataloaders))), p=self.probs)

        try:
            data, labels = self.iters[task].__next__()
        except StopIteration:
            self.iters[task] = iter(self.dataloaders[task])
            data, labels = self.iters[task].__next__()

        self.step += 1

        return data, labels, task        
