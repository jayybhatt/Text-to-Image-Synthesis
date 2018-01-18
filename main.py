from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from datetime import datetime
import model
from dataset import TextDataset

import pdb
pdb.set_trace()

parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--dataset',
#     required=True,
#     default='folder',
#     help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument(
	'--dataroot', required=True, default='./data/coco', help='path to dataset')
parser.add_argument(
	'--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument(
	'--batchSize', type=int, default=64, help='input batch size')
parser.add_argument(
	'--imageSize',
	type=int,
	default=64,
	help='the height / width of the input image to network')
parser.add_argument(
	'--nte',
	type=int,
	default=1024,
	help='the size of the text embedding vector')
parser.add_argument(
	'--nt',
	type=int,
	default=256,
	help='the reduced size of the text embedding vector')
parser.add_argument(
	'--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument(
	'--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument(
	'--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument(
	'--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument(
	'--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument(
	'--netG', default='', help="path to netG (to continue training)")
parser.add_argument(
	'--netD', default='', help="path to netD (to continue training)")
parser.add_argument(
	'--outf',
	default='./output/',
	help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument(
	'--eval', 
	action='store_true', 
	help="choose whether to train the model or show demo")
opt = parser.parse_args()
print(opt)

try:
	output_dir = os.path.join(opt.outf,
							  datetime.strftime(datetime.now(), "%Y%m%d_%H%M"))
	os.makedirs(output_dir)
except OSError:
	pass

if opt.manualSeed is None:
	opt.manualSeed = random.randint(
		1, 10000
	)  #use random.randint(1, 10000) for randomness, shouldnt be done when we want to continue training from a checkpoint
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
	torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
	print(
		"WARNING: You have a CUDA device, so you should probably run with --cuda"
	)

image_transform = transforms.Compose([
	transforms.RandomCrop(opt.imageSize),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0, 0, 0), (1, 1, 1))
])

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
nt = int(opt.nt)
nte = int(opt.nte)


# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
		# m.bias.data.fill_(0)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


netG = model._netG(ngpu, nz, ngf, nc, nte, nt)
netG.apply(weights_init)
if opt.netG != '':
	netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model._netD(ngpu, nc, ndf, nte, nt)
netD.apply(weights_init)
if opt.netD != '':
	netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
	netD.cuda()
	netG.cuda()
	criterion.cuda()
	input, label = input.cuda(), label.cuda()
	noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

if not opt.eval:

	train_dataset = TextDataset(opt.dataroot, transform=image_transform)
	
	## Completed - TODO: Make a new DataLoader and Dataset to include embeddings
	train_dataloader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=opt.batchSize,
		shuffle=True,
		num_workers=int(opt.workers))

	# setup optimizer
	optimizerD = optim.Adam(
		netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
	optimizerG = optim.Adam(
		netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

	## Completed TODO: Change the error loss function to include embeddings [refer main_cls.lua on the original paper repo]

	for epoch in range(1, opt.niter + 1):
		if epoch % 75 == 0:
			optimizerG.param_groups[0]['lr'] /= 2
			optimizerD.param_groups[0]['lr'] /= 2
		for i, data in enumerate(train_dataloader, 0):
			############################
			# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
			###########################
			# train with real
			netD.zero_grad()
			real_cpu, text_embedding, _ = data
			batch_size = real_cpu.size(0)
			text_embedding = Variable(text_embedding)

			if opt.cuda:
				real_cpu = real_cpu.cuda()
				text_embedding = text_embedding.cuda()

			input.resize_as_(real_cpu).copy_(real_cpu)
			label.resize_(batch_size).fill_(real_label)
			inputv = Variable(input)
			labelv = Variable(label)

			output = netD(inputv, text_embedding)
			errD_real = criterion(output, labelv)  ##
			errD_real.backward()
			D_x = output.data.mean()

			### calculate errD_wrong
			inputv = torch.cat((inputv[1:], inputv[:1]), 0)
			output = netD(inputv, text_embedding)
			errD_wrong = criterion(output, labelv) * 0.5
			errD_wrong.backward()

			# train with fake
			noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
			noisev = Variable(noise)
			fake = netG(noisev, text_embedding)
			labelv = Variable(label.fill_(fake_label))
			output = netD(fake.detach(), text_embedding)
			errD_fake = criterion(output, labelv) * 0.5
			errD_fake.backward()
			D_G_z1 = output.data.mean()

			errD = errD_real + errD_fake + errD_wrong
			# errD.backward()
			optimizerD.step()

			############################
			# (2) Update G network: maximize log(D(G(z)))
			###########################
			netG.zero_grad()
			labelv = Variable(label.fill_(
				real_label))  # fake labels are real for generator cost
			output = netD(fake, text_embedding)
			errG = criterion(output, labelv)  ##
			errG.backward()
			D_G_z2 = output.data.mean()
			optimizerG.step()

			print(
				'[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
				% (epoch, opt.niter, i, len(train_dataloader), errD.data[0],
				errG.data[0], D_x, D_G_z1, D_G_z2))
			if i % 100 == 0:
				vutils.save_image(
					real_cpu, '%s/real_samples.png' % output_dir, normalize=True)
				fake = netG(fixed_noise, text_embedding)
				vutils.save_image(
					fake.data,
					'%s/fake_samples_epoch_%03d.png' % (output_dir, epoch),
					normalize=True)

		# do checkpointing
		torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (output_dir,
																epoch))
		torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (output_dir,
																epoch))

else:
	test_dataset = TextDataset(opt.dataroot, transform=image_transform,split='test')

	## Completed - TODO: Make a new DataLoader and Dataset to include embeddings
	test_dataloader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=opt.batchSize,
		shuffle=True,
		num_workers=int(opt.workers))

	for i, data in enumerate(test_dataloader, 0):
		real_image, text_embedding,caption = data
		batch_size = real_image.size(0)
		text_embedding = Variable(text_embedding)

		if opt.cuda:
			real_image = real_image.cuda()
			text_embedding = text_embedding.cuda()

		input.resize_as_(real_image).copy_(real_image)
		inputv = Variable(input)

		noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
		noisev = Variable(noise)
		num_test_outputs = 10
		

		# for count in range(num_test_outputs):
		# 	print (count)
		count =0
		print (i)
		synthetic_image = netG(noisev, text_embedding)
		synthetic_image = synthetic_image.detach()
		for i in range(synthetic_image.size()[0]):
			cap = caption[i].strip(".")
			cap = cap.replace("/"," or ")
			cap = cap.replace(" ","_")
			if len(cap) > 95:
				cap = cap[:95]
			file_path = './eval_results/'+cap
			# if not os.path.exists(file_path):
			# 	os.makedirs(file_path)
			try:
				vutils.save_image(synthetic_image[i].data,file_path+'_'+str(count)+'.jpg')
				# vutils.save_image(synthetic_image[i].data,os.path.join(file_path,str(count)+'.jpg'))
			except e:
				print (e)
