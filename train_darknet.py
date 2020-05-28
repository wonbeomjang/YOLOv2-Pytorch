import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torchvision.datasets.cifar import CIFAR100
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm
from models.darknet19 import DarkNet19
from utils.utils import AverageMeter


def train(net, criterion, optimizer, train_loader, val_loader, config, scheduler=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss_avg = AverageMeter()
    val_loss_avg = AverageMeter()

    train_step = len(train_loader)
    val_step = len(test_loader)

    writer = SummaryWriter(config.log_path)
    criterion.to(device)
    net.train()
    net.to(device)
    if scheduler:
        scheduler.step(config.epoch)

    for epoch in range(config.epoch, config.num_epoch):
        train_loss_avg.reset()
        val_loss_avg.reset()
        iter = tqdm(enumerate(train_loader))
        iter.set_description(f'Train Step in {epoch} total step: {train_step}')

        for step, (images, targets) in iter:
            images = images.to(device)
            targets = targets.to(device)

            preds = net(images)

            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_avg.update(loss.item())

        iter = tqdm(enumerate(val_loader))
        iter.set_description(f'Validation Step in {epoch} total step: {val_step}')
        for step, (images, targets) in iter:
            images = images.to(device)
            targets = targets.to(device)

            preds = net(images)
            loss = criterion(preds, targets)

            val_loss_avg.update(loss.item())
        writer.add_scalars('DarkNet19/loss', {'train': train_loss_avg.avg, 'validation': val_loss_avg.avg}, epoch)
        writer.add_scalar('DarkNet19/LearningRate', optimizer.param_groups[0]['lr'], epoch)

        torch.save(net.state_dict(), f'{config.checkpoint_dir}/DarkNet19-{epoch}.pth')
        if scheduler:
            scheduler.step()


def test(net, data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iter = tqdm(enumerate(data_loader))
    iter.set_description(f'Testing')
    correct = 0
    num_sample = 0
    net.eval()
    net.to(device)

    for step, (images, targets) in iter:
        images = images.to(device)
        targets = targets.to(device)

        preds: torch.Tensor = net(images)
        correct += (preds.argmax(1) == targets).sum().item()
        num_sample += targets.size(0)

    print(f'Final Accuracy: {correct / num_sample * 100: .4f}%')

    return correct / num_sample * 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--num_classes', type=int, default=2, help='number of model output channels')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=166, help='number of epochs to train for')
    parser.add_argument('--num_epoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
    parser.add_argument('--log_path', default='logs', help="path to saved models (to continue training)")
    parser.add_argument('--num_workers', type=int, default=5, help="")
    parser.add_argument('--dataset', default='dataset', help="")
    parser.add_argument('--train_dir', default='dataset/train', help="")
    parser.add_argument('--val_dir', default='dataset/val', help="")

    config = parser.parse_args()

    if not os.path.exists(config.dataset):
        os.makedirs(config.dataset)
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    net = DarkNet19(100)

    train_set = CIFAR100(config.dataset, train=True, transform=transform, download=True)
    test_set = CIFAR100(config.dataset, train=False, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]), download=True)
    train_set, val_set = random_split(train_set, [int(len(train_set) * 0.9), len(train_set) - int(len(train_set) * 0.9)])

    # train_set = ImageFolder(config.train_dir, transform=transform)
    # val_set = ImageFolder(config.val_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, config.batch_size, True, num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, config.batch_size, True, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, config.batch_size, False, num_workers=config.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    if os.path.exists(f'{config.checkpoint_dir}/DarNekt19-{config.epoch - 1}.pth'):
        print('[*] Load parameter')
        net.load_state_dict(torch.load(f'{config.checkpoint_dir}/DarkNet19-{config.epoch - 1}.pth'))

    train(net, criterion, optimizer, train_loader, val_loader, config)
    with torch.no_grad():
        writer = SummaryWriter(config.log_path)
        for epoch in range(config.num_epoch):
            net.load_state_dict(torch.load(f'{config.checkpoint_dir}/DarkNet19-{epoch}.pth'))
            writer.add_scalar('DarkNet19/Accuracy', test(net, test_loader), epoch)

    os.system('shutdown -s')
