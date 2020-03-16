import torchvision
import torchvision.transforms as transforms
import torch
import torch.utils.data
import resnet
from torch.autograd import Variable
from torch import nn
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import early_stop
from tqdm import tqdm

import os,sys
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_globa_step=0
val_globa_step=0

wd=1e-50
learning_rate=1e-4
epochs=100
batch_size=24
torch.backends.cudnn.benchmark = True
transform= transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ])

transform_test= transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]


def split_Train_Val_Data(data_dir, ratio):
    """ the sum of ratio must equal to 1"""
    dataset = ImageFolder(data_dir)  # data_dir精确到分类目录的上一级
    character = [[] for i in range(len(dataset.classes))]
    # print(dataset.class_to_idx)
    for x, y in dataset.samples:  # 将数据按类标存放
        character[y].append(x)
    # print(dataset.samples)

    train_inputs, val_inputs = [], []
    train_labels, val_labels = [], []
    for i, data in enumerate(character):  # data为一类图片
        num_sample_train = int(len(data) * ratio[0])
        # print(num_sample_train)
        num_sample_val = int(len(data) * ratio[1])
        num_val_index = num_sample_train + num_sample_val
        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num_val_index]:
            val_inputs.append(str(x))
            val_labels.append(i)
    # print(len(train_inputs))
    train_dataloader = DataLoader(MyDataset(train_inputs, train_labels, transform),
                                  batch_size=batch_size, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(MyDataset(val_inputs, val_labels, transform_test),
                                batch_size=batch_size, shuffle=False, num_workers=16)

    return train_dataloader, val_dataloader


data_dir = '/256_ObjectCategories'
trainloader, testloader = split_Train_Val_Data(data_dir, [0.95, 0.05])

n = resnet.resnet101().cuda()

weight_p, bias_p = [],[]
for name, p in n.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]

trans_params = list(map(id, n.trans_conv.parameters()))
class_params = list(map(id, n.group2.parameters()))

base_params = filter(lambda p: id(p) not in trans_params,
                     n.parameters())
base_params = filter(lambda p: id(p) not in class_params,
                     base_params)



loss1 =nn.MSELoss()
loss1.cuda()
loss2=nn.CrossEntropyLoss()
loss2.cuda()
optimizer = torch.optim.Adam([{'params': base_params},
                              {'params':n.trans_conv.parameters(),'lr':learning_rate},
                              {'params':n.group2.parameters(),'lr':learning_rate}],
                      lr=learning_rate,weight_decay=wd)

opt = torch.optim.Adam([{'params': base_params},
                          {'params':n.trans_conv.parameters(),'lr':learning_rate}],
                      lr=learning_rate,weight_decay=wd)

if os.path.exists('bestmodel_params.pkl'):
    checkpoint = torch.load('bestmodel_params.pkl')
    n.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    opt.load_state_dict(checkpoint['opt_state_dict2'])

sch=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1,patience=15)
schopt=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.1,patience=15)
es=early_stop.EarlyStopping('max',patience=20)
for epoch in range(epochs):
        loadertrain = tqdm(trainloader, desc='{} E{:03d}'.format('train', epoch), ncols=0)
        loadertest = tqdm(testloader, desc='{} E{:03d}'.format('test', epoch), ncols=0)
        epoch_loss = 0.0
        correct=0.0
        total=0.0
        total2=0.0
        correct2=0.0
        for x_train, y_train in loadertrain:
            n.train()

            x_train, y_train = Variable(x_train.cuda()),Variable(y_train.cuda())
            x_noise=torch.FloatTensor(x_train.size(0),3,224,224).uniform_(-0.03,0.03)
            x_noise=torch.clamp(x_noise,-0.03,0.03)
            x_train_noise=x_train+Variable(x_noise.cuda())
            y_pre,c_pre = n(x_train_noise)

            y_pre=y_pre.cuda()

            n.zero_grad()
            optimizer.zero_grad()
            loss = loss1(torch.mul(y_pre,1.0), torch.mul( x_train,1.0))
            if loss.item()>1.0:
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(n.parameters(), 5.0)
                opt.step()
                epoch_loss += loss.data.item()
                _, predicted = torch.max(c_pre.data, 1)
                total += y_train.size(0)
                correct += predicted.eq(y_train.data).cuda().sum()
                torch.cuda.empty_cache()
            else:
                loss_cl=loss2(c_pre,y_train)

                loss_sum=torch.mul(loss,1/1)+loss_cl
                loss_sum.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(n.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss_sum.data.item()
                _, predicted = torch.max(c_pre.data, 1)
                total += y_train.size(0)
                correct += predicted.eq(y_train.data).cuda().sum()

                train_globa_step+=1
                torch.cuda.empty_cache()
                if loss.item()<1.0:

                    y_pre2, c_pre2 = n(y_pre)
                    y_pre2 = y_pre2.cuda()

                    n.zero_grad()
                    optimizer.zero_grad()
                    lossreg2 = loss1(torch.mul(y_pre2, 1.0), torch.mul( x_train, 1.0))
                    loss_cl2 = loss2(c_pre2, y_train)
                    _, predicted2 = torch.max(c_pre2.data, 1)
                    total2 += y_train.size(0)
                    correct2 += predicted2.eq(y_train.data).cuda().sum()
                    loss_sum2 = torch.mul(lossreg2, 1 / 10) + loss_cl2
                    loss_sum2.backward()
                    torch.nn.utils.clip_grad_norm_(n.parameters(), 5.0)
                    optimizer.step()
                    torch.cuda.empty_cache()
            if train_globa_step% 20==0:

                n.eval()
                checkpoint = {
                'state_dict': n.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'opt_state_dict2':opt.state_dict(),
                'epoch': epoch
            }

                torch.save(checkpoint, 'model_params.pkl')
            fmt = '{:.4f}'.format
            loadertrain.set_postfix(loss=fmt(loss.data.item()),

                                    acc=fmt(correct.item() / total * 100))
        
        if (epoch) % 1 ==0:
            test_loss = 0.0
            correct = 0.0
            total = 0.0
            n.eval()
            with torch.no_grad():
                for x_test, y_test in loadertest:


                    x_test, y_test = Variable(x_test.cuda()), Variable(y_test.cuda())
                  
                    y_pre, c_pre = n(x_test)

                    y_pre = y_pre.cuda()

                    loss_cl = loss2(c_pre, y_test)
                    loss = loss1(torch.mul(y_pre,1.0), torch.mul( x_test,1.0))

                    loss_sum = torch.mul(loss,1/1)
                    test_loss += loss_sum.data.item()
                    _, predicted = torch.max(c_pre.data, 1)
                    total += y_test.size(0)
                    correct += predicted.eq(y_test.data).cuda().sum()
                    val_globa_step+=1
                    fmt = '{:.4f}'.format
                    loadertest.set_postfix(loss=fmt(loss_sum.data.item()),

                                            acc=fmt(correct.item() / total * 100))
                sch.step(test_loss)
                fl=es.step(correct.item()/total*100, n,optimizer,opt,epoch)
                if fl:
                   torch.cuda.empty_cache()
                   sys.exit(0) 
                torch.cuda.empty_cache()