import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from art.data_generators import *
from art.utils import *
from art.classifiers import *
from art.attacks import *
import numpy as np
import resnet as resnet
import argparse
import models
from tqdm import tqdm
parser = argparse.ArgumentParser()
import warnings
warnings.filterwarnings("ignore")
parser.add_argument('--s', type=str, default='mnist',help='attack on which dataset:mnist, svhn, caltech101 or caltech256')
parser.add_argument('--d', type=str, default='l2',help='attack based on which distance met:inf,l1,l2')
parser.add_argument('--m', type=str, default='fgsm',help='attack based on which method:fgsm,pgd,cw,boundary,deepfool,jsma,bim,zoo(using --a w)')
parser.add_argument('--e', type=float, default=1.5,help='max distance between adv example and the ori:inf--0.3 for mnist 8/255.0 for others,l2--1.5 for mnist')
parser.add_argument('--a',type=str,default='w',help='attack method including whitebox(w) and blackbox(b)')
parser.add_argument('--at',type=str,default=None,help='model under attack with which method to train:None, fgsm ,pgd')
parser.add_argument('--atw',type=str,default=None,help='only in blackbox attack, which method helping model used:None, fgsm, pgd')
args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Hyper-parameters
param = {
    'test_batch_size': 100,
}


transform_test = transforms.Compose([torchvision.transforms.Resize((64, 64)),
									 transforms.ToTensor(),
									 ])
if args.s=='mnist':
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
elif args.s=='svhn':
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
else:
    print('Wrong dataset')
    exit(0)
loader_test = torch.utils.data.DataLoader(testset, batch_size=param['test_batch_size'],
										 shuffle=True, num_workers=8)


class res_m(nn.Module):
    def __init__(self, model1):
        super(res_m, self).__init__()
        self.m1 = model1

    def forward(self, input):
        _,y = self.m1(input)
        return y


pydataloader=PyTorchDataGenerator(loader_test,10000,param['test_batch_size'])

pylist=[]
for i in range(10000//param['test_batch_size']):
    (x,y)=pydataloader.get_batch()
    pylist.append((x,y))
# Setup model to be attacked
if args.a=='w':
    net = resnet.resnet101().cuda()
    if args.at is None:
        checkpoint = torch.load('bestmodel_params.pkl')
        net.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint = torch.load('bestmodel_params_adv_train_%s.pkl'%args.at)
        net.load_state_dict(checkpoint['state_dict'])
    
    net.eval()
    res = res_m(net).eval()
    loss = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adamax(res.parameters())
    fmodel = PyTorchClassifier(
        res,loss=loss,optimizer=optimizer,input_shape=(3,64,64),nb_classes=10,clip_values=(0.0, 1.0))
    
    evalmodel = PyTorchClassifier(
        res,loss=loss,optimizer=optimizer,input_shape=(3,64,64),nb_classes=10,clip_values=(0.0, 1.0))
elif args.a=='b':
    netblack = resnet.resnet101().cuda()
    net=models.resnet101().cuda()
    if args.atw is None:
        checkpoint = torch.load('bestmodel_params_resnet.pkl')
        net.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint = torch.load('bestmodel_params_resnet_adv_train_%s.pkl'%args.atw)
        net.load_state_dict(checkpoint['state_dict'])
    
    net.eval()
    
    if args.at is None:
        checkpoint = torch.load('bestmodel_params.pkl')
        netblack.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint = torch.load('bestmodel_params_adv_train_%s.pkl'%args.at)
        netblack.load_state_dict(checkpoint['state_dict'])
    res_black = res_m(netblack).eval()
    loss1 = nn.CrossEntropyLoss().cuda()
    optimizer1 = torch.optim.Adamax(net.parameters())
    loss2 = nn.CrossEntropyLoss().cuda()
    optimizer2 = torch.optim.Adamax(res_black.parameters())
    fmodel = PyTorchClassifier(
        net, loss=loss1,optimizer=optimizer1,input_shape=(3,64,64),nb_classes=10,clip_values=(0.0, 1.0))
    
    evalmodel = PyTorchClassifier(
        res_black, loss=loss2,optimizer=optimizer2,input_shape=(3,64,64),nb_classes=10,clip_values=(0.0, 1.0))
else:
    print('wrong attack type')
    exit(0)

ori_acc=0
adv_acc=0
loadertrain = tqdm(pylist, desc='{}'.format('attack'), ncols=0)
for x_train, y_train in loadertrain:
    if args.s=='mnist':
        x_train=np.stack((x_train,x_train,x_train),1)
    x_train.shape=(param['test_batch_size'],3,64,64)

    preds = np.argmax(fmodel.predict(x_train,batch_size=param['test_batch_size']), axis=1)
    preds.shape = (param['test_batch_size'])
    y=y_train.copy()
    y.shape = (param['test_batch_size'])
    y_train=to_categorical(y_train,10)
    acc_o = np.sum(preds == y)
    ori_acc+=acc_o
    
# Craft adversarial samples with FGSM
    epsilon = args.e # Maximum perturbation
    if args.m=='fgsm':
        if args.d=='inf':
            adv_crafter = FastGradientMethod(fmodel,norm=np.inf,eps=epsilon,batch_size=param['test_batch_size'])
        elif args.d=='l2':
            adv_crafter = FastGradientMethod(fmodel,norm=2,eps=epsilon,batch_size=param['test_batch_size'])
        elif args.d=='l1':
            adv_crafter = FastGradientMethod(fmodel,norm=1,eps=epsilon,batch_size=param['test_batch_size'])
        else:
            print('wrong distance')
            exit(0)
        x_test_adv = adv_crafter.generate(x_train,y_train)
    elif args.m=='pgd':
        if args.d=='inf':
            if args.s=='mnist':
                adv_crafter = ProjectedGradientDescent(fmodel,norm=np.inf,eps=epsilon,batch_size=param['test_batch_size'])
            elif args.s=='svhn':
                adv_crafter = ProjectedGradientDescent(fmodel, norm=np.inf, eps=epsilon, eps_step=2 / 255.0,
                                                       max_iter=40, batch_size=param['test_batch_size'])
        elif args.d=='l2':
            adv_crafter = ProjectedGradientDescent(fmodel,norm=2,eps=epsilon,batch_size=param['test_batch_size'])
        elif args.d=='l1':
            adv_crafter = ProjectedGradientDescent(fmodel,norm=1,eps=epsilon,batch_size=param['test_batch_size'])
        else:
            print('wrong distance')
            exit(0)
        x_test_adv = adv_crafter.generate(x_train,y_train)
    elif args.m=='boundary':
        if  args.d=='inf':
            adv_crafter = HopSkipJump(fmodel,targeted=False,norm=np.inf,max_eval=100)
        elif args.d=='l2':
            adv_crafter = HopSkipJump(fmodel, targeted=False,norm=2,max_eval=100)
        else:
            print('wrong distance')
            exit(0)
        x_test_adv = adv_crafter.generate(x_train)
    elif args.m=='cw':
        if args.d=='l2':
            adv_crafter = CarliniL2Method(fmodel,batch_size=param['test_batch_size'])
        elif args.d=='inf':
            adv_crafter = CarliniLInfMethod(fmodel,eps=epsilon,batch_size=param['test_batch_size'])
        else:
            print('wrong distance')
            exit(0)
        x_test_adv = adv_crafter.generate(x_train,y_train)
    elif args.m=='deepfool':
        adv_crafter = DeepFool(fmodel,batch_size=param['test_batch_size'])
        x_test_adv = adv_crafter.generate(x_train,y_train)
	
    elif args.m=='jsma':
        adv_crafter = SaliencyMapMethod(fmodel, batch_size=param['test_batch_size'])
        x_test_adv = adv_crafter.generate(x_train, y_train)
    elif args.m=='bim':
        adv_crafter = BasicIterativeMethod(fmodel, batch_size=param['test_batch_size'])
        x_test_adv = adv_crafter.generate(x_train, y_train)
    elif args.m == 'zoo' and args.a == 'w':
        adv_crafter = ZooAttack(fmodel,nb_parallel=1024, batch_size=param['test_batch_size'])
        x_test_adv = adv_crafter.generate(x_train, y_train)
    elif args.m == 'zoo' and args.a == 'b':
        print('zoo used in --a w condition')
        exit(0)
    else:
        print('wrong method')
        exit(0)
    if x_test_adv is not None:

        preds = np.argmax(evalmodel.predict(x_test_adv), axis=1)
        preds.shape = (param['test_batch_size'])
        acc_a = np.sum(preds == y)
        adv_acc += acc_a
    else:
        preds = np.argmax(evalmodel.predict(x_train), axis=1)
        preds.shape = (param['test_batch_size'])
        acc_a = np.sum(preds == y)
        adv_acc += acc_a
    loadertrain.set_postfix(oriacc=ori_acc,

                            advacc=adv_acc)

print("\nTest accuracy: %.2f%%" % (ori_acc/10000 * 100))
print("\nTest accuracy on adversarial sample: %.2f%%" % (adv_acc/10000 * 100))
