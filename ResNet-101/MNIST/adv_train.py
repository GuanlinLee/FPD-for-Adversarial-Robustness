import torchvision
import torch.utils.data
import resnet
from torch.autograd import Variable
from torch import nn
import early_stop_adv_train
import os, sys
from torchvision import  transforms

import argparse
parser = argparse.ArgumentParser()
import warnings
warnings.filterwarnings("ignore")

parser.add_argument('--m', type=str, default='fgsm')
parser.add_argument('--g', type=str, default='0')


args = parser.parse_args()


warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = args.g
torch.backends.cudnn.benchmark = True
from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
train_globa_step=0
val_globa_step=0
from tqdm import tqdm
wd=1e-5
learning_rate=1e-3
epochs=100
batch_size=200
torch.backends.cudnn.benchmark = True
transform=transforms.Compose([
                              torchvision.transforms.RandomRotation(15,  expand=True, center=None),
                              torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                              torchvision.transforms.Resize((64,64)),
                              torchvision.transforms.ToTensor(),
                             ])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)


transform_test=transforms.Compose([torchvision.transforms.Resize((64,64)),
                              transforms.ToTensor(),
                             ])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=8)

n = resnet.resnet101().cuda()


weight_p, bias_p = [], []
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
param = {
	'delay': 10,
}
if args.m=='fgsm':
	adversary = FGSMAttack()
elif args.m=='pgd':
	adversary = LinfPGDAttack()
else:
	print('wrong method')
	exit(0)
loss1 = nn.MSELoss()
loss1.cuda()
loss2 = nn.CrossEntropyLoss()
loss2.cuda()
optimizer = torch.optim.Adamax([{'params': base_params},
                              {'params':n.trans_conv.parameters(),'lr':learning_rate},
                              {'params':n.group2.parameters(),'lr':learning_rate}],
                      lr=learning_rate,weight_decay=wd)

opt = torch.optim.Adamax([{'params': base_params},
                          {'params':n.trans_conv.parameters(),'lr':learning_rate}],
                      lr=learning_rate,weight_decay=wd)

if os.path.exists('bestmodel_params_adv_train_%s.pkl'%(args.m)):
    checkpoint = torch.load('bestmodel_params_adv_train_%s.pkl'%(args.m))
    n.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    opt.load_state_dict(checkpoint['opt_state_dict2'])

sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=15)

es = early_stop_adv_train.EarlyStopping('max', patience=10)
for epoch in range(epochs):
	loadertrain = tqdm(trainloader, desc='{} E{:03d}'.format('train', epoch), ncols=0)
	loadertest = tqdm(testloader, desc='{} E{:03d}'.format('test', epoch), ncols=0)
	epoch_loss = 0.0
	correct = 0.0
	total = 0.0
	total2 = 0.0
	correct2 = 0.0
	for x_train, y_train in loadertrain:
		n.train()
		
		x_train, y_train = Variable(x_train.cuda()), Variable(y_train.cuda())
		
		y_pre, c_pre = n(torch.cat((x_train, x_train, x_train), 1))
		
		y_pre = y_pre.cuda()
		
		n.zero_grad()
		optimizer.zero_grad()
		n.zero_grad()
		optimizer.zero_grad()
		loss = loss1(torch.mul(y_pre, 255.0), torch.mul(torch.cat((x_train, x_train, x_train), 1), 1.0))
		if loss.item() > 3:
			loss.backward(retain_graph=True)
			torch.nn.utils.clip_grad_norm(n.parameters(), 10.0)
			opt.step()
			epoch_loss += loss.data.item()
			_, predicted = torch.max(c_pre.data, 1)
			total += y_train.size(0)
			correct += predicted.eq(y_train.data).cuda().sum()
			torch.cuda.empty_cache()
		else:
			loss_cl = loss2(c_pre, y_train)
			
			loss_sum = torch.mul(loss, 1 / 1) + loss_cl
			if epoch + 1 > param['delay']:
				# use predicted label to prevent label leaking
				y_pred = pred_batch(torch.cat((x_train, x_train, x_train), 1), n)
				x_adv = adv_train(torch.cat((x_train, x_train, x_train), 1), y_pred, n, loss2, adversary)
				n.zero_grad()
				optimizer.zero_grad()
				x_adv_var = to_var(x_adv)
				y_pre, c_pre = n(x_adv_var)
				loss_adv = loss2(c_pre, y_train)+loss1(torch.mul(y_pre, 1.0), torch.mul(torch.cat((x_train, x_train, x_train), 1), 1.0))/ 1
				loss_sum = (loss_sum + loss_adv) / 2
			loss_sum.backward(retain_graph=True)
			torch.nn.utils.clip_grad_norm(n.parameters(), 10.0)
			optimizer.step()
			epoch_loss += loss_sum.data.item()
			_, predicted = torch.max(c_pre.data, 1)
			total += y_train.size(0)
			correct += predicted.eq(y_train.data).cuda().sum()
			
			train_globa_step += 1
			torch.cuda.empty_cache()
			if loss.item() < 3 and epoch + 1 > param['delay']:
				y_pre2, c_pre2 = n(y_pre)
				y_pre2 = y_pre2.cuda()
				
				n.zero_grad()
				optimizer.zero_grad()
				lossreg2 = loss1(torch.mul(y_pre2, 1.0), torch.mul(torch.cat((x_train, x_train, x_train), 1), 1.0))
				loss_cl2 = loss2(c_pre2, y_train)
				_, predicted2 = torch.max(c_pre2.data, 1)
				total2 += y_train.size(0)
				correct2 += predicted2.eq(y_train.data).cuda().sum()
				loss_sum2 = torch.mul(lossreg2, 1 / 1) + loss_cl2
				loss_sum2.backward()
				torch.nn.utils.clip_grad_norm(n.parameters(), 10.0)
				optimizer.step()
				torch.cuda.empty_cache()


		

		if train_globa_step % 100 == 0:
			n.eval()
			checkpoint = {
				'state_dict': n.state_dict(),
				'opt_state_dict': optimizer.state_dict(),
				'opt_state_dict2': opt.state_dict(),
				'epoch': epoch
			}
			
			torch.save(checkpoint, 'model_params_adv_train%s.pkl'%(args.m))
		fmt = '{:.4f}'.format
		loadertrain.set_postfix(loss=fmt(loss.data.item()),
		
		                        acc=fmt(correct.item() / total * 100))

	if (epoch) % 1 == 0:
		test_loss = 0.0
		correct = 0.0
		total = 0.0
		n.eval()
		with torch.no_grad():
			for x_test, y_test in loadertest:
				
				x_test, y_test = Variable(x_test.cuda()), Variable(y_test.cuda())
				
				y_pre, c_pre = n(torch.cat((x_test, x_test, x_test), 1))
				
				y_pre = y_pre.cuda()
				
				loss_cl = loss2(c_pre, y_test)
				loss = loss1(torch.mul(y_pre, 1.0), torch.mul(torch.cat((x_test, x_test, x_test), 1), 1.0))
				
				loss_sum = torch.mul(loss, 1 / 1) + loss_cl
				test_loss += loss_sum.data.item()
				_, predicted = torch.max(c_pre.data, 1)
				total += y_test.size(0)
				correct += predicted.eq(y_test.data).cuda().sum()
				val_globa_step += 1
				fmt = '{:.4f}'.format
				loadertest.set_postfix(loss=fmt(loss_sum.data.item()),
				
				                        acc=fmt(correct.item() / total * 100))
			sch.step(test_loss)
			fl = es.step(correct.item() / total * 100, n, optimizer, opt, epoch,args.m)
			if fl:
				torch.cuda.empty_cache()
				sys.exit(0)
			torch.cuda.empty_cache()

