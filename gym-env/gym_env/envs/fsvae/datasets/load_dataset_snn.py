import os
import torchvision.datasets
import torchvision.transforms as transforms
import torch
import global_v as glv

from .usbEventDataset import USBEventDataset
from .eventDataset import EventDataset

def load_mnist(data_path):
    print("loading MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])
    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def load_fashionmnist(data_path):
    print("loading Fashion MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.FashionMNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.FashionMNIST(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    return trainloader, testloader

def load_cifar10(data_path):
    print("loading CIFAR10")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.CIFAR10(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.CIFAR10(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
    return trainloader, testloader

def load_celebA(data_path):
    print("loading CelebA")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']
    
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange])

    trainset = torchvision.datasets.CelebA(root=data_path, 
                                            split='train', 
                                            download=True, 
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers=8, pin_memory=True)

    testset = torchvision.datasets.CelebA(root=data_path, 
                                            split='test', 
                                            download=True, 
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size, 
                                            shuffle=False, num_workers=8, pin_memory=True)
    return trainloader, testloader


def load_usb_events(data_path):
    print("loading usb_events")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']
    dataset = glv.network_config['dataset']
    
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(148),
        # transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
        ])

    
    dataset_path = os.path.join(data_path, dataset)
    data_info_path = os.path.join(dataset_path, "data_info.txt")
    # events_path = os.path.join(dataset_path, "events")
    events_path = os.path.join(dataset_path, "cropped_event_imgs")


    train_dataset = USBEventDataset(data_info_path, events_path, split='train', transform=transform)
    test_dataset = USBEventDataset(data_info_path, events_path, split='test', transform=transform)

    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers=8, pin_memory=False, drop_last=False)

    testloader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False, num_workers=8, pin_memory=False, drop_last=False)

    return trainloader, testloader

def load_events(data_path):
    print("loading events")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']
    dataset = glv.network_config['dataset']
    
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(148),
        # transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
        ])

    
    dataset_path = os.path.join(data_path, dataset)
    data_info_path = os.path.join(dataset_path, "data_info.txt")
    # events_path = os.path.join(dataset_path, "events")
    # events_path = os.path.join(dataset_path, "cropped_event_imgs")
    # events_path = os.path.join(dataset_path, "grey_cropped_event_imgs")
    events_path = os.path.join(dataset_path, "small_grey_cropped_event_imgs")



    train_dataset = EventDataset(data_info_path, events_path, split='train', transform=transform)
    test_dataset = EventDataset(data_info_path, events_path, split='test', transform=transform)

    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers=2, pin_memory=False, drop_last=False)

    testloader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False, num_workers=2, pin_memory=False, drop_last=False)

    return trainloader, testloader


def load_hole(data_path):
    print("loading events")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']
    dataset = glv.network_config['dataset']
    
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(148),
        # transforms.Resize((input_size,input_size)),
        # transforms.RandomAffine([-180, 180], [0.5, 0.5], [0.3, 1.1], fill=127),
        transforms.RandomAffine([-180, 180], translate=[0.5, 0.5], fill=127),
        transforms.ToTensor(),
        SetRange
        ])

    
    dataset_path = os.path.join(data_path, dataset)
    data_info_path = os.path.join(dataset_path, "data_info.txt")
    # events_path = os.path.join(dataset_path, "events")
    # events_path = os.path.join(dataset_path, "cropped_event_imgs")
    # events_path = os.path.join(dataset_path, "grey_cropped_event_imgs")
    # events_path = os.path.join(dataset_path, "small_grey_cropped_event_imgs")


    events_path = os.path.join(dataset_path, "preproc")

    



    # train_dataset = EventDataset(data_info_path, events_path, split='train', transform=transform)
    test_dataset = EventDataset(data_info_path, events_path, split='test', transform=transform)

    # trainloader = torch.utils.data.DataLoader(train_dataset, 
    #                                         batch_size=batch_size, 
    #                                         shuffle=True, num_workers=2, pin_memory=False, drop_last=False)

    trainloader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers=8, pin_memory=True, drop_last=False)

    testloader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    return trainloader, testloader