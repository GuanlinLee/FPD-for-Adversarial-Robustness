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

parser.add_argument('--m', type=str, default='pgd')
parser.add_argument('--g', type=str, default='1')


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
wd=0.0
learning_rate=1e-5
epochs=100
batch_size=64
torch.backends.cudnn.benchmark = True
transform=transforms.Compose([
                              torchvision.transforms.Resize((64,64)),
                              torchvision.transforms.ToTensor(),
                             ])

trainset = torchvision.datasets.SVHN(root='./data',split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=16)

transform_test=transforms.Compose([torchvision.transforms.Resize((64,64)),
                              transforms.ToTensor(),
                             ])

testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=16)



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
	'delay': 0,
}
if args.m=='fgsm':
	adversary = FGSMAttack(epsilon=8.0/255.0)
elif args.m=='pgd':
	adversary = LinfPGDAttack( epsilon=8.0/255.0, a=2.0/255.0,k=40)
else:
	print('wrong method')
	exit(0)
loss1 = nn.MSELoss()
loss1.cuda()
loss2 = nn.CrossEntropyLoss()
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
		
		y_pre,c_pre = n( x_train)
		
		n.zero_grad()
		optimizer.zero_grad()
		_, predicted = torch.max(c_pre.data, 1)
		total += y_train.size(0)
		correct += predicted.eq(y_train.data).cuda().sum()
		loss = loss2(c_pre, y_train)+loss1(torch.mul(y_pre, 1.0), torch.mul( x_train, 1.0))/ 1

		if epoch + 1 > param['delay']:
			# use predicted label to prevent label leaking
			y_pred = pred_batch(x_train, n)
			x_adv = adv_train(x_train, y_pred, n, loss2, adversary)
			x_adv_var = to_var(x_adv)
			y_pre, c_pre = n(x_adv_var)
			loss_adv = loss2( c_pre , y_train)+loss1(torch.mul(y_pre, 1.0), torch.mul(x_adv_var, 1.0))/ 1

			loss =  (loss_adv +loss)/2
		
		loss.backward(retain_graph=True)
		torch.nn.utils.clip_grad_norm(n.parameters(), 5.0)
		optimizer.step()
		epoch_loss += loss.data.item()

		torch.cuda.empty_cache()
		if epoch + 1 > param['delay']:
                    y_pre2, c_pre2 = n(y_pre)
                    y_pre2 = y_pre2.cuda()

                    n.zero_grad()
                    optimizer.zero_grad()
                    lossreg2 = loss1(torch.mul(y_pre2, 1.0), torch.mul(x_train, 1.0))
                    loss_cl2 = loss2(c_pre2, y_train)
                    _, predicted2 = torch.max(c_pre2.data, 1)
                    total2 += y_train.size(0)
                    correct2 += predicted2.eq(y_train.data).cuda().sum()
                    loss_sum2 = torch.mul(lossreg2, 1 / 1) + loss_cl2
                    loss_sum2.backward()
                    torch.nn.utils.clip_grad_norm(n.parameters(), 5.0)
                    optimizer.step()
		train_globa_step += 1
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
				
				y_pre,c_pre = n(x_test)
				
				loss_cl = loss2(c_pre, y_test)
				
				loss_sum =  loss_cl
				test_loss += loss_sum.data.item()
				_, predicted = torch.max(c_pre.data, 1)
				total += y_test.size(0)
				correct += predicted.eq(y_test.data).cuda().sum()
				val_globa_step += 1
				fmt = '{:.4f}'.format
				loadertest.set_postfix(loss=fmt(loss_sum.data.item()),
				
				                        acc=fmt(correct.item() / total * 100))
			sch.step(test_loss)
			fl = es.step(correct.item() / total * 100, n, optimizer,opt, epoch,args.m)
			if fl:
				torch.cuda.empty_cache()
				sys.exit(0)
			torch.cuda.empty_cache()

