#coding:utf-8
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import pdb


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
	def forward(self, x):
		return x


def get_norm_layer(norm_type='instance'):
	"""Return a normalization layer

	Parameters:
		norm_type (str) -- the name of the normalization layer: batch | instance | none

	For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
	For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
	"""
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
	elif norm_type == 'none':
		norm_layer = lambda x: Identity()
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


def get_scheduler(optimizer, opt):
	"""Return a learning rate scheduler

	Parameters:
		optimizer          -- the optimizer of the network
		opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
							  opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

	For 'linear', we keep the same learning rate for the first <opt.niter> epochs
	and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
	For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
	See https://pytorch.org/docs/stable/optim.html for more details.
	"""
	if opt.lr_policy == 'linear':
		def lambda_rule(epoch):
			lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
			return lr_l
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
	elif opt.lr_policy == 'cosine':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
	return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
	"""Initialize network weights.

	Parameters:
		net (network)   -- network to be initialized
		init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
		init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

	We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
	work better for some applications. Feel free to try yourself.
	"""
	def init_func(m):  # define the initialization function
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
			init.normal_(m.weight.data, 1.0, init_gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
	"""Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
	Parameters:
		net (network)      -- the network to be initialized
		init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
		gain (float)       -- scaling factor for normal, xavier and orthogonal.
		gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

	Return an initialized network.
	"""
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
	init_weights(net, init_type, init_gain=init_gain)
	return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], model0_res=0, model1_res=0, extra_channel=3):
	"""Create a generator

	Parameters:
		input_nc (int) -- the number of channels in input images
		output_nc (int) -- the number of channels in output images
		ngf (int) -- the number of filters in the last conv layer
		netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
		norm (str) -- the name of normalization layers used in the network: batch | instance | none
		use_dropout (bool) -- if use dropout layers.
		init_type (str)    -- the name of our initialization method.
		init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
		gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

	Returns a generator

	Our current implementation provides two types of generators:
		U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
		The original U-Net paper: https://arxiv.org/abs/1505.04597

		Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
		Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
		We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


	The generator has been initialized by <init_net>. It uses RELU for non-linearity.
	"""
	net = None
	norm_layer = get_norm_layer(norm_type=norm)

	if netG == 'resnet_9blocks':
		net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
	elif netG == 'resnet_8blocks':
		net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=8)
	elif netG == 'resnet_style_9blocks':
		net = ResnetStyleGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, extra_channel=extra_channel)
	elif netG == 'resnet_style2_9blocks':
		net = ResnetStyle2Generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, model0_res=model0_res, extra_channel=extra_channel)
	elif netG == 'resnet_style2_8blocks':
		net = ResnetStyle2Generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=8, model0_res=model0_res, extra_channel=extra_channel)
	elif netG == 'resnet_style2_10blocks':
		net = ResnetStyle2Generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=10, model0_res=model0_res, extra_channel=extra_channel)
	elif netG == 'resnet_style3decoder_9blocks':
		net = ResnetStyle3DecoderGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, model0_res=model0_res)
	elif netG == 'resnet_style2mc_9blocks':
		net = ResnetStyle2MCGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, model0_res=model0_res, extra_channel=extra_channel)
	elif netG == 'resnet_style2mc2_9blocks':
		net = ResnetStyle2MC2Generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, model0_res=model0_res, model1_res=model1_res, extra_channel=extra_channel)
	elif netG == 'resnet_6blocks':
		net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
	elif netG == 'unet_128':
		net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
	elif netG == 'unet_256':
		net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
	return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], n_class=3):
	"""Create a discriminator

	Parameters:
		input_nc (int)     -- the number of channels in input images
		ndf (int)          -- the number of filters in the first conv layer
		netD (str)         -- the architecture's name: basic | n_layers | pixel
		n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
		norm (str)         -- the type of normalization layers used in the network.
		init_type (str)    -- the name of the initialization method.
		init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
		gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

	Returns a discriminator

	Our current implementation provides three types of discriminators:
		[basic]: 'PatchGAN' classifier described in the original pix2pix paper.
		It can classify whether 70×70 overlapping patches are real or fake.
		Such a patch-level discriminator architecture has fewer parameters
		than a full-image discriminator and can work on arbitrarily-sized images
		in a fully convolutional fashion.

		[n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
		with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

		[pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
		It encourages greater color diversity but has no effect on spatial statistics.

	The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
	"""
	net = None
	norm_layer = get_norm_layer(norm_type=norm)

	if netD == 'basic':  # default PatchGAN classifier
		net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
	elif netD == 'basic_cls':
		net = NLayerDiscriminatorCls(input_nc, ndf, n_layers=3, n_class=3, norm_layer=norm_layer)
	elif netD == 'n_layers':  # more options
		net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
	elif netD == 'pixel':     # classify if each pixel is real or fake
		net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
	else:
		raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
	return init_net(net, init_type, init_gain, gpu_ids)


def define_HED(init_weights_, gpu_ids_=[]):
	net = HED()

	if len(gpu_ids_) > 0:
		assert(torch.cuda.is_available())
		net.to(gpu_ids_[0])
		net = torch.nn.DataParallel(net, gpu_ids_)  # multi-GPUs

	if not init_weights_ == None:
		device = torch.device('cuda:{}'.format(gpu_ids_[0])) if gpu_ids_ else torch.device('cpu')
		print('Loading model from: %s'%init_weights_)
		state_dict = torch.load(init_weights_, map_location=str(device))
		if isinstance(net, torch.nn.DataParallel):
			net.module.load_state_dict(state_dict)
		else:
			net.load_state_dict(state_dict)
		print('load the weights successfully')

	return net

def define_VGG(init_weights_, feature_mode_, batch_norm_=False, num_classes_=1000, gpu_ids_=[]):
	net = VGG19(init_weights=init_weights_, feature_mode=feature_mode_, batch_norm=batch_norm_, num_classes=num_classes_)
	# set the GPU
	if len(gpu_ids_) > 0:
		assert(torch.cuda.is_available())
		net.cuda(gpu_ids_[0])
		net = torch.nn.DataParallel(net, gpu_ids_)  # multi-GPUs

	if not init_weights_ == None:
		device = torch.device('cuda:{}'.format(gpu_ids_[0])) if gpu_ids_ else torch.device('cpu')
		print('Loading model from: %s'%init_weights_)
		state_dict = torch.load(init_weights_, map_location=str(device))
		if isinstance(net, torch.nn.DataParallel):
			net.module.load_state_dict(state_dict)
		else:
			net.load_state_dict(state_dict)
		print('load the weights successfully')
	return net

###################################################################################################################
from torchvision.models import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
def define_vgg11_bn(gpu_ids_=[],vec=0):
	net = vgg11_bn(pretrained=True)
	net.classifier[6] = nn.Linear(4096, 1) #LSGAN needs no sigmoid, LSGAN-nn.MSELoss()
	if len(gpu_ids_) > 0:
		assert(torch.cuda.is_available())
		net.cuda(gpu_ids_[0])
		net = torch.nn.DataParallel(net, gpu_ids_)
	return net
def define_vgg19_bn(gpu_ids_=[],vec=0):
	net = vgg19_bn(pretrained=True)
	net.classifier[6] = nn.Linear(4096, 1) #LSGAN needs no sigmoid, LSGAN-nn.MSELoss()
	if len(gpu_ids_) > 0:
		assert(torch.cuda.is_available())
		net.cuda(gpu_ids_[0])
		net = torch.nn.DataParallel(net, gpu_ids_)
	return net
def define_vgg19(gpu_ids_=[],vec=0):
	net = vgg19(pretrained=True)
	net.classifier[6] = nn.Linear(4096, 1) #LSGAN needs no sigmoid, LSGAN-nn.MSELoss()
	if len(gpu_ids_) > 0:
		assert(torch.cuda.is_available())
		net.cuda(gpu_ids_[0])
		net = torch.nn.DataParallel(net, gpu_ids_)
	return net
###################################################################################################################
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
def define_resnet101(gpu_ids_=[],vec=0):
	net = resnet101(pretrained=True)
	num_ftrs = net.fc.in_features
	net.fc = nn.Linear(num_ftrs, 1) #LSGAN needs no sigmoid, LSGAN-nn.MSELoss()
	if len(gpu_ids_) > 0:
		assert(torch.cuda.is_available())
		net.cuda(gpu_ids_[0])
		net = torch.nn.DataParallel(net, gpu_ids_)
	return net
def define_resnet101a(init_weights_,gpu_ids_=[],vec=0):
	net = resnet101(pretrained=True)
	num_ftrs = net.fc.in_features
	net.fc = nn.Linear(num_ftrs, 1) #LSGAN needs no sigmoid, LSGAN-nn.MSELoss()
	if not init_weights_ == None:
		print('Loading model from: %s'%init_weights_)
		state_dict = torch.load(init_weights_, map_location=str(torch.device('cpu')))
		if isinstance(net, torch.nn.DataParallel):
			net.module.load_state_dict(state_dict)
		else:
			net.load_state_dict(state_dict)
		print('load the weights successfully')
	if len(gpu_ids_) > 0:
		assert(torch.cuda.is_available())
		net.cuda(gpu_ids_[0])
		net = torch.nn.DataParallel(net, gpu_ids_)
	return net
###################################################################################################################
import pretrainedmodels.models.resnext as resnext
def define_resnext101(gpu_ids_=[],vec=0):
	net = resnext.resnext101_64x4d(num_classes=1000,pretrained='imagenet')
	net.last_linear = nn.Linear(2048, 1) #LSGAN needs no sigmoid, LSGAN-nn.MSELoss()
	if len(gpu_ids_) > 0:
		assert(torch.cuda.is_available())
		net.cuda(gpu_ids_[0])
		net = torch.nn.DataParallel(net, gpu_ids_)
	return net
def define_resnext101a(init_weights_,gpu_ids_=[],vec=0):
	net = resnext.resnext101_64x4d(num_classes=1000,pretrained='imagenet')
	net.last_linear = nn.Linear(2048, 1) #LSGAN needs no sigmoid, LSGAN-nn.MSELoss()
	if not init_weights_ == None:
		print('Loading model from: %s'%init_weights_)
		state_dict = torch.load(init_weights_, map_location=str(torch.device('cpu')))
		if isinstance(net, torch.nn.DataParallel):
			net.module.load_state_dict(state_dict)
		else:
			net.load_state_dict(state_dict)
		print('load the weights successfully')
	if len(gpu_ids_) > 0:
		assert(torch.cuda.is_available())
		net.cuda(gpu_ids_[0])
		net = torch.nn.DataParallel(net, gpu_ids_)
	return net
###################################################################################################################
from torchvision.models import Inception3, inception_v3
def define_inception3(gpu_ids_=[],vec=0):
    net = inception_v3(pretrained=True)
    net.transform_input = False # assume [-1,1] input
    net.fc = nn.Linear(2048, 1)
    net.aux_logits = False
    if len(gpu_ids_) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids_[0])
        net = torch.nn.DataParallel(net, gpu_ids_)
    return net
def define_inception3a(init_weights_,gpu_ids_=[],vec=0):
    net = inception_v3(pretrained=True)
    net.transform_input = False # assume [-1,1] input
    net.fc = nn.Linear(2048, 1)
    net.aux_logits = False
    if not init_weights_ == None:
        print('Loading model from: ', init_weights_)
        state_dict = torch.load(init_weights_, map_location=str(torch.device('cpu')))
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(state_dict)
        else:
            net.load_state_dict(state_dict)
        print('load the weights successfully')
    if len(gpu_ids_) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids_[0])
        net = torch.nn.DataParallel(net, gpu_ids_)
    return net
###################################################################################################################
from torchvision.models.inception import BasicConv2d
def define_inception_v3(init_weights_,gpu_ids_=[],vec=0):

	## pretrained = True
	kwargs = {}
	if 'transform_input' not in kwargs:
		kwargs['transform_input'] = True
	if 'aux_logits' in kwargs:
		original_aux_logits = kwargs['aux_logits']
		kwargs['aux_logits'] = True
	else:
		original_aux_logits = True
	print(kwargs)
	net = Inception3(**kwargs)

	if not init_weights_ == None:
		print('Loading model from: %s'%init_weights_)
		state_dict = torch.load(init_weights_, map_location=str(torch.device('cpu')))
		if isinstance(net, torch.nn.DataParallel):
			net.module.load_state_dict(state_dict)
		else:
			net.load_state_dict(state_dict)
		print('load the weights successfully')

	if not original_aux_logits:
		net.aux_logits = False
		del net.AuxLogits

	net.fc = nn.Linear(2048, 1)
	if vec == 1:
		net.Conv2d_1a_3x3 = BasicConv2d(6, 32, kernel_size=3, stride=2)
	net.aux_logits = False

	if len(gpu_ids_) > 0:
		assert(torch.cuda.is_available())
		net.cuda(gpu_ids_[0])
		net = torch.nn.DataParallel(net, gpu_ids_)

	return net

def define_inception_v3a(init_weights_,gpu_ids_=[],vec=0):

	kwargs = {}
	if 'transform_input' not in kwargs:
		kwargs['transform_input'] = True
	if 'aux_logits' in kwargs:
		original_aux_logits = kwargs['aux_logits']
		kwargs['aux_logits'] = True
	else:
		original_aux_logits = True
	print(kwargs)
	net = Inception3(**kwargs)

	if not original_aux_logits:
		net.aux_logits = False
		del net.AuxLogits

	net.fc = nn.Linear(2048, 1)
	if vec == 1:
		net.Conv2d_1a_3x3 = BasicConv2d(6, 32, kernel_size=3, stride=2)
	net.aux_logits = False

	if not init_weights_ == None:
		print('Loading model from: %s'%init_weights_)
		state_dict = torch.load(init_weights_, map_location=str(torch.device('cpu')))
		if isinstance(net, torch.nn.DataParallel):
			net.module.load_state_dict(state_dict)
		else:
			net.load_state_dict(state_dict)
		print('load the weights successfully')

	if len(gpu_ids_) > 0:
		assert(torch.cuda.is_available())
		net.cuda(gpu_ids_[0])
		net = torch.nn.DataParallel(net, gpu_ids_)

	return net

def define_inception_ori(init_weights_,transform_input=False,gpu_ids_=[]):

	## pretrained = True
	kwargs = {}
	kwargs['transform_input'] = transform_input

	if 'aux_logits' in kwargs:
		original_aux_logits = kwargs['aux_logits']
		kwargs['aux_logits'] = True
	else:
		original_aux_logits = True
	print(kwargs)
	net = Inception3(**kwargs)


	if not init_weights_ == None:
		print('Loading model from: %s'%init_weights_)
		state_dict = torch.load(init_weights_, map_location=str(torch.device('cpu')))
		if isinstance(net, torch.nn.DataParallel):
			net.module.load_state_dict(state_dict)
		else:
			net.load_state_dict(state_dict)
		print('load the weights successfully')
		#for e in list(net.modules()):
		#	print(e)

	if not original_aux_logits:
		net.aux_logits = False
		del net.AuxLogits


	if len(gpu_ids_) > 0:
		assert(torch.cuda.is_available())
		net.cuda(gpu_ids_[0])

	return net
###################################################################################################################

def define_DT(init_weights_, input_nc_, output_nc_, ngf_, netG_, norm_, use_dropout_, init_type_, init_gain_, gpu_ids_):
	net = define_G(input_nc_, output_nc_, ngf_, netG_, norm_, use_dropout_, init_type_, init_gain_, gpu_ids_)

	if not init_weights_ == None:
		device = torch.device('cuda:{}'.format(gpu_ids_[0])) if gpu_ids_ else torch.device('cpu')
		print('Loading model from: %s'%init_weights_)
		state_dict = torch.load(init_weights_, map_location=str(device))
		if isinstance(net, torch.nn.DataParallel):
			net.module.load_state_dict(state_dict)
		else:
			net.load_state_dict(state_dict)
		print('load the weights successfully')
	return net

def define_C(input_nc, classes, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], h=512, w=512, nnG=3, dim=4096):
	net = None
	norm_layer = get_norm_layer(norm_type=norm)
	if netG == 'classifier':
		net = Classifier(input_nc, classes, ngf, num_downs=nnG, norm_layer=norm_layer, use_dropout=use_dropout, h=h, w=w, dim=dim)
	elif netG == 'vgg':
		net = VGG19(init_weights=None, feature_mode=False, batch_norm=True, num_classes=classes)
	return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
	"""Define different GAN objectives.

	The GANLoss class abstracts away the need to create the target label tensor
	that has the same size as the input.
	"""

	def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
		""" Initialize the GANLoss class.

		Parameters:
			gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
			target_real_label (bool) - - label for a real image
			target_fake_label (bool) - - label of a fake image

		Note: Do not use sigmoid as the last layer of Discriminator.
		LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
		"""
		super(GANLoss, self).__init__()
		self.register_buffer('real_label', torch.tensor(target_real_label))
		self.register_buffer('fake_label', torch.tensor(target_fake_label))
		self.gan_mode = gan_mode
		if gan_mode == 'lsgan':#cyclegan
			self.loss = nn.MSELoss()
		elif gan_mode == 'vanilla':
			self.loss = nn.BCEWithLogitsLoss()
		elif gan_mode in ['wgangp']:
			self.loss = None
		else:
			raise NotImplementedError('gan mode %s not implemented' % gan_mode)

	def get_target_tensor(self, prediction, target_is_real):
		"""Create label tensors with the same size as the input.

		Parameters:
			prediction (tensor) - - tpyically the prediction from a discriminator
			target_is_real (bool) - - if the ground truth label is for real images or fake images

		Returns:
			A label tensor filled with ground truth label, and with the size of the input
		"""

		if target_is_real:
			target_tensor = self.real_label
		else:
			target_tensor = self.fake_label
		return target_tensor.expand_as(prediction)

	def __call__(self, prediction, target_is_real):
		"""Calculate loss given Discriminator's output and grount truth labels.

		Parameters:
			prediction (tensor) - - tpyically the prediction output from a discriminator
			target_is_real (bool) - - if the ground truth label is for real images or fake images

		Returns:
			the calculated loss.
		"""
		if self.gan_mode in ['lsgan', 'vanilla']:
			target_tensor = self.get_target_tensor(prediction, target_is_real)
			loss = self.loss(prediction, target_tensor)
		elif self.gan_mode == 'wgangp':
			if target_is_real:
				loss = -prediction.mean()
			else:
				loss = prediction.mean()
		return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
	"""Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

	Arguments:
		netD (network)              -- discriminator network
		real_data (tensor array)    -- real images
		fake_data (tensor array)    -- generated images from the generator
		device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
		type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
		constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
		lambda_gp (float)           -- weight for this loss

	Returns the gradient penalty loss
	"""
	if lambda_gp > 0.0:
		if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
			interpolatesv = real_data
		elif type == 'fake':
			interpolatesv = fake_data
		elif type == 'mixed':
			alpha = torch.rand(real_data.shape[0], 1, device=device)
			alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
			interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
		else:
			raise NotImplementedError('{} not implemented'.format(type))
		interpolatesv.requires_grad_(True)
		disc_interpolates = netD(interpolatesv)
		gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
										grad_outputs=torch.ones(disc_interpolates.size()).to(device),
										create_graph=True, retain_graph=True, only_inputs=True)
		gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
		gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
		return gradient_penalty, gradients
	else:
		return 0.0, None


class ResnetGenerator(nn.Module):
	"""Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

	We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
	"""

	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
		"""Construct a Resnet-based generator

		Parameters:
			input_nc (int)      -- the number of channels in input images
			output_nc (int)     -- the number of channels in output images
			ngf (int)           -- the number of filters in the last conv layer
			norm_layer          -- normalization layer
			use_dropout (bool)  -- if use dropout layers
			n_blocks (int)      -- the number of ResNet blocks
			padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
		"""
		assert(n_blocks >= 0)
		super(ResnetGenerator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		model = [nn.ReflectionPad2d(3),
				 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
				 norm_layer(ngf),
				 nn.ReLU(True)]

		n_downsampling = 2
		for i in range(n_downsampling):  # add downsampling layers
			mult = 2 ** i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
					  norm_layer(ngf * mult * 2),
					  nn.ReLU(True)]

		mult = 2 ** n_downsampling
		for i in range(n_blocks):       # add ResNet blocks

			model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		for i in range(n_downsampling):  # add upsampling layers
			mult = 2 ** (n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
										 kernel_size=3, stride=2,
										 padding=1, output_padding=1,
										 bias=use_bias),
					  norm_layer(int(ngf * mult / 2)),
					  nn.ReLU(True)]
		model += [nn.ReflectionPad2d(3)]
		model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		model += [nn.Tanh()]

		self.model = nn.Sequential(*model)

	def forward(self, input, feature_mode = False):
		"""Standard forward"""
		if not feature_mode:
			return self.model(input)
		else:
			module_list = list(self.model.modules())
			x = input.clone()
			indexes = list(range(1,11))+[11,20,29,38,47,56,65,74,83]+list(range(92,101))
			for i in indexes:
				x = module_list[i](x)
				if i == 3:
					x1 = x.clone()
				elif i == 6:
					x2 = x.clone()
				elif i == 9:
					x3 = x.clone()
				elif i == 47:
					y7 = x.clone()
				elif i == 83:
					y4 = x.clone()
				elif i == 93:
					y3 = x.clone()
				elif i == 96:
					y2 = x.clone()
			#y = self.model(input)
			#pdb.set_trace()
			return x,x1,x2,x3,y4,y3,y2,y7

class ResnetStyleGenerator(nn.Module):
	"""Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

	We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
	"""

	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
		"""Construct a Resnet-based generator

		Parameters:
			input_nc (int)      -- the number of channels in input images
			output_nc (int)     -- the number of channels in output images
			ngf (int)           -- the number of filters in the last conv layer
			norm_layer          -- normalization layer
			use_dropout (bool)  -- if use dropout layers
			n_blocks (int)      -- the number of ResNet blocks
			padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
		"""
		assert(n_blocks >= 0)
		super(ResnetStyleGenerator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		model0 = [nn.ReflectionPad2d(3),
				 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
				 norm_layer(ngf),
				 nn.ReLU(True)]

		n_downsampling = 2
		for i in range(n_downsampling):  # add downsampling layers
			mult = 2 ** i
			model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
					  norm_layer(ngf * mult * 2),
					  nn.ReLU(True)]

		mult = 2 ** n_downsampling
		model1 = [nn.Conv2d(3, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
					  norm_layer(ngf * mult),
					  nn.ReLU(True)]

		model = []
		model += [nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
					  norm_layer(ngf * mult),
					  nn.ReLU(True)]
		for i in range(n_blocks):       # add ResNet blocks

			model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		for i in range(n_downsampling):  # add upsampling layers
			mult = 2 ** (n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
										 kernel_size=3, stride=2,
										 padding=1, output_padding=1,
										 bias=use_bias),
					  norm_layer(int(ngf * mult / 2)),
					  nn.ReLU(True)]
		model += [nn.ReflectionPad2d(3)]
		model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		model += [nn.Tanh()]

		self.model0 = nn.Sequential(*model0)
		self.model1 = nn.Sequential(*model1)
		self.model = nn.Sequential(*model)

	def forward(self, input1, input2):
		"""Standard forward"""
		f1 = self.model0(input1)
		f2 = self.model1(input2)
		#pdb.set_trace()
		f1 = torch.cat((f1,f2), 1)
		return self.model(f1)


class ResnetStyle2Generator(nn.Module):
	"""Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

	We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
	"""

	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', extra_channel=3, model0_res=0):
		"""Construct a Resnet-based generator

		Parameters:
			input_nc (int)      -- the number of channels in input images
			output_nc (int)     -- the number of channels in output images
			ngf (int)           -- the number of filters in the last conv layer
			norm_layer          -- normalization layer
			use_dropout (bool)  -- if use dropout layers
			n_blocks (int)      -- the number of ResNet blocks
			padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
		"""
		assert(n_blocks >= 0)
		super(ResnetStyle2Generator, self).__init__()
		self.n_blocks = n_blocks
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		model0 = [nn.ReflectionPad2d(3),
				 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
				 norm_layer(ngf),
				 nn.ReLU(True)]

		n_downsampling = 2
		for i in range(n_downsampling):  # add downsampling layers
			mult = 2 ** i
			model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
					  norm_layer(ngf * mult * 2),
					  nn.ReLU(True)]

		mult = 2 ** n_downsampling
		for i in range(model0_res):       # add ResNet blocks
			model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		model = []
		model += [nn.Conv2d(ngf * mult + extra_channel, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
					  norm_layer(ngf * mult),
					  nn.ReLU(True)]

		for i in range(n_blocks-model0_res):       # add ResNet blocks
			model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		for i in range(n_downsampling):  # add upsampling layers
			mult = 2 ** (n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
										 kernel_size=3, stride=2,
										 padding=1, output_padding=1,
										 bias=use_bias),
					  norm_layer(int(ngf * mult / 2)),
					  nn.ReLU(True)]
		model += [nn.ReflectionPad2d(3)]
		model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		model += [nn.Tanh()]

		self.model0 = nn.Sequential(*model0)
		self.model = nn.Sequential(*model)
		#print(list(self.modules()))

	def forward(self, input1, input2, feature_mode=False, ablate_res=-1):
		"""Standard forward"""
		if not feature_mode:
			if ablate_res == -1:
				f1 = self.model0(input1)
				y1 = torch.cat([f1, input2], 1)
				return self.model(y1)
			else:
				f1 = self.model0(input1)
				y = torch.cat([f1, input2], 1)
				module_list = list(self.model.modules())
				for i in range(1, 4):#merge module
					y = module_list[i](y)
				for k in range(self.n_blocks):#resblocks
					if k+1 == ablate_res:
						print('skip resblock'+str(k+1))
						continue
					y1 = y.clone()
					for i in range(6+9*k,13+9*k):
						y = module_list[i](y)
					y = y1 + y
				for i in range(4+9*self.n_blocks,13+9*self.n_blocks):#up convs
					y = module_list[i](y)
				return y
		else:
			module_list0 = list(self.model0.modules())
			x = input1.clone()
			for i in range(1,11):
				x = module_list0[i](x)
				if i == 3:
					x1 = x.clone()#[1,64,512,512]
				elif i == 6:
					x2 = x.clone()#[1,128,256,256]
				elif i == 9:
					x3 = x.clone()#[1,256,128,128]
			#f1 = self.model0(input1)#[1,256,128,128]
			#pdb.set_trace()
			y1 = torch.cat([x, input2], 1)#[1,259,128,128]
			module_list = list(self.model.modules())
			indexes = list(range(1,4))+[4,13,22,31,40,49,58,67,76]+list(range(85,94))
			y = y1.clone()
			for i in indexes:
				y = module_list[i](y)
				if i == 76:
					y4 = y.clone()#[1,256,128,128]
				elif i == 86:
					y3 = y.clone()#[1,128,256,256]
				elif i == 89:
					y2 = y.clone()#[1,64,512,512]
				elif i == 40:
					y7 = y.clone()
			#out = self.model(y1)
			#pdb.set_trace()
			return y,x1,x2,x3,y4,y3,y2,y7

class ResnetStyle3DecoderGenerator(nn.Module):
	"""Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

	We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
	"""

	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', model0_res=0):
		"""Construct a Resnet-based generator

		Parameters:
			input_nc (int)      -- the number of channels in input images
			output_nc (int)     -- the number of channels in output images
			ngf (int)           -- the number of filters in the last conv layer
			norm_layer          -- normalization layer
			use_dropout (bool)  -- if use dropout layers
			n_blocks (int)      -- the number of ResNet blocks
			padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
		"""
		assert(n_blocks >= 0)
		super(ResnetStyle3DecoderGenerator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		model0 = [nn.ReflectionPad2d(3),
				 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
				 norm_layer(ngf),
				 nn.ReLU(True)]

		n_downsampling = 2
		for i in range(n_downsampling):  # add downsampling layers
			mult = 2 ** i
			model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
					  norm_layer(ngf * mult * 2),
					  nn.ReLU(True)]

		mult = 2 ** n_downsampling
		for i in range(model0_res):       # add ResNet blocks
			model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		model1 = []
		model2 = []
		model3 = []
		for i in range(n_blocks-model0_res):       # add ResNet blocks
			model1 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
			model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
			model3 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		for i in range(n_downsampling):  # add upsampling layers
			mult = 2 ** (n_downsampling - i)
			model1 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
										 kernel_size=3, stride=2,
										 padding=1, output_padding=1,
										 bias=use_bias),
					  norm_layer(int(ngf * mult / 2)),
					  nn.ReLU(True)]
			model2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
										 kernel_size=3, stride=2,
										 padding=1, output_padding=1,
										 bias=use_bias),
					  norm_layer(int(ngf * mult / 2)),
					  nn.ReLU(True)]
			model3 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
										 kernel_size=3, stride=2,
										 padding=1, output_padding=1,
										 bias=use_bias),
					  norm_layer(int(ngf * mult / 2)),
					  nn.ReLU(True)]
		model1 += [nn.ReflectionPad2d(3)]
		model1 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		model1 += [nn.Tanh()]
		model2 += [nn.ReflectionPad2d(3)]
		model2 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		model2 += [nn.Tanh()]
		model3 += [nn.ReflectionPad2d(3)]
		model3 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		model3 += [nn.Tanh()]

		self.model0 = nn.Sequential(*model0)
		self.model1 = nn.Sequential(*model1)
		self.model2 = nn.Sequential(*model2)
		self.model3 = nn.Sequential(*model3)
		print(list(self.modules()))

	def forward(self, input, domain):
		"""Standard forward"""
		f1 = self.model0(input)
		if domain == 0:
			y = self.model1(f1)
		elif domain == 1:
			y = self.model2(f1)
		elif domain == 2:
			y = self.model3(f1)
		return y

class ResnetStyle2MCGenerator(nn.Module):
	# multi-column

	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', extra_channel=3, model0_res=0):
		"""Construct a Resnet-based generator

		Parameters:
			input_nc (int)      -- the number of channels in input images
			output_nc (int)     -- the number of channels in output images
			ngf (int)           -- the number of filters in the last conv layer
			norm_layer          -- normalization layer
			use_dropout (bool)  -- if use dropout layers
			n_blocks (int)      -- the number of ResNet blocks
			padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
		"""
		assert(n_blocks >= 0)
		super(ResnetStyle2MCGenerator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		model0 = [nn.ReflectionPad2d(3),
				 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
				 norm_layer(ngf),
				 nn.ReLU(True)]

		n_downsampling = 2
		model1_3 = []
		model1_5 = []
		for i in range(n_downsampling):  # add downsampling layers
			mult = 2 ** i
			model1_3 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
					  norm_layer(ngf * mult * 2),
					  nn.ReLU(True)]
			model1_5 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=5, stride=2, padding=2, bias=use_bias),
					  norm_layer(ngf * mult * 2),
					  nn.ReLU(True)]

		mult = 2 ** n_downsampling
		for i in range(model0_res):       # add ResNet blocks
			model1_3 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
			model1_5 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, kernel=5)]

		model = []
		model += [nn.Conv2d(ngf * mult * 2 + extra_channel, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
					  norm_layer(ngf * mult),
					  nn.ReLU(True)]

		for i in range(n_blocks-model0_res):       # add ResNet blocks
			model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		for i in range(n_downsampling):  # add upsampling layers
			mult = 2 ** (n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
										 kernel_size=3, stride=2,
										 padding=1, output_padding=1,
										 bias=use_bias),
					  norm_layer(int(ngf * mult / 2)),
					  nn.ReLU(True)]
		model += [nn.ReflectionPad2d(3)]
		model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		model += [nn.Tanh()]

		self.model0 = nn.Sequential(*model0)
		self.model1_3 = nn.Sequential(*model1_3)
		self.model1_5 = nn.Sequential(*model1_5)
		self.model = nn.Sequential(*model)
		print(list(self.modules()))

	def forward(self, input1, input2):
		"""Standard forward"""
		f0 = self.model0(input1)
		f1 = self.model1_3(f0)
		f2 = self.model1_5(f0)
		y1 = torch.cat([f1, f2, input2], 1)
		return self.model(y1)

class ResnetStyle2MC2Generator(nn.Module):
	# multi-column, need to insert style early

	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', extra_channel=3, model0_res=0, model1_res=0):
		"""Construct a Resnet-based generator

		Parameters:
			input_nc (int)      -- the number of channels in input images
			output_nc (int)     -- the number of channels in output images
			ngf (int)           -- the number of filters in the last conv layer
			norm_layer          -- normalization layer
			use_dropout (bool)  -- if use dropout layers
			n_blocks (int)      -- the number of ResNet blocks
			padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
		"""
		assert(n_blocks >= 0)
		super(ResnetStyle2MC2Generator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		model0 = [nn.ReflectionPad2d(3),
				 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
				 norm_layer(ngf),
				 nn.ReLU(True)]

		n_downsampling = 2
		model1_3 = []
		model1_5 = []
		for i in range(n_downsampling):  # add downsampling layers
			mult = 2 ** i
			model1_3 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
					  norm_layer(ngf * mult * 2),
					  nn.ReLU(True)]
			model1_5 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=5, stride=2, padding=2, bias=use_bias),
					  norm_layer(ngf * mult * 2),
					  nn.ReLU(True)]

		mult = 2 ** n_downsampling
		for i in range(model0_res):       # add ResNet blocks
			model1_3 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
			model1_5 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, kernel=5)]

		model2_3 = []
		model2_5 = []
		model2_3 += [nn.Conv2d(ngf * mult + extra_channel, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
					  norm_layer(ngf * mult),
					  nn.ReLU(True)]
		model2_5 += [nn.Conv2d(ngf * mult + extra_channel, ngf * mult, kernel_size=5, stride=1, padding=2, bias=use_bias),
					  norm_layer(ngf * mult),
					  nn.ReLU(True)]

		for i in range(model1_res):       # add ResNet blocks
			model2_3 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
			model2_5 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, kernel=5)]

		model = []
		model += [nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
					  norm_layer(ngf * mult),
					  nn.ReLU(True)]
		for i in range(n_blocks-model0_res-model1_res):       # add ResNet blocks
			model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		for i in range(n_downsampling):  # add upsampling layers
			mult = 2 ** (n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
										 kernel_size=3, stride=2,
										 padding=1, output_padding=1,
										 bias=use_bias),
					  norm_layer(int(ngf * mult / 2)),
					  nn.ReLU(True)]
		model += [nn.ReflectionPad2d(3)]
		model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		model += [nn.Tanh()]

		self.model0 = nn.Sequential(*model0)
		self.model1_3 = nn.Sequential(*model1_3)
		self.model1_5 = nn.Sequential(*model1_5)
		self.model2_3 = nn.Sequential(*model2_3)
		self.model2_5 = nn.Sequential(*model2_5)
		self.model = nn.Sequential(*model)
		print(list(self.modules()))

	def forward(self, input1, input2):
		"""Standard forward"""
		f0 = self.model0(input1)
		f1 = self.model1_3(f0)
		f2 = self.model1_5(f0)
		f3 = self.model2_3(torch.cat([f1,input2],1))
		f4 = self.model2_5(torch.cat([f2,input2],1))
		#pdb.set_trace()
		return self.model(torch.cat([f3,f4],1))

class ResnetBlock(nn.Module):
	"""Define a Resnet block"""

	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, kernel=3):
		"""Initialize the Resnet block

		A resnet block is a conv block with skip connections
		We construct a conv block with build_conv_block function,
		and implement skip connections in <forward> function.
		Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
		"""
		super(ResnetBlock, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, kernel)

	def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, kernel=3):
		"""Construct a convolutional block.

		Parameters:
			dim (int)           -- the number of channels in the conv layer.
			padding_type (str)  -- the name of padding layer: reflect | replicate | zero
			norm_layer          -- normalization layer
			use_dropout (bool)  -- if use dropout layers.
			use_bias (bool)     -- if the conv layer uses bias or not

		Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
		"""
		conv_block = []
		p = 0
		pad = int((kernel-1)/2)
		if padding_type == 'reflect':#by default
			conv_block += [nn.ReflectionPad2d(pad)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(pad)]
		elif padding_type == 'zero':
			p = pad
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		conv_block += [nn.Conv2d(dim, dim, kernel_size=kernel, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
		if use_dropout:
			conv_block += [nn.Dropout(0.5)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(pad)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(pad)]
		elif padding_type == 'zero':
			p = pad
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=kernel, padding=p, bias=use_bias), norm_layer(dim)]

		return nn.Sequential(*conv_block)

	def forward(self, x):
		"""Forward function (with skip connections)"""
		out = x + self.conv_block(x)  # add skip connections
		return out


class UnetGenerator(nn.Module):
	"""Create a Unet-based generator"""

	def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
		"""Construct a Unet generator
		Parameters:
			input_nc (int)  -- the number of channels in input images
			output_nc (int) -- the number of channels in output images
			num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
								image of size 128x128 will become of size 1x1 # at the bottleneck
			ngf (int)       -- the number of filters in the last conv layer
			norm_layer      -- normalization layer

		We construct the U-Net from the innermost layer to the outermost layer.
		It is a recursive process.
		"""
		super(UnetGenerator, self).__init__()
		# construct unet structure
		unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
		for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
			unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
		# gradually reduce the number of filters from ngf * 8 to ngf
		unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

	def forward(self, input):
		"""Standard forward"""
		return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
	"""Defines the Unet submodule with skip connection.
		X -------------------identity----------------------
		|-- downsampling -- |submodule| -- upsampling --|
	"""

	def __init__(self, outer_nc, inner_nc, input_nc=None,
				 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
		"""Construct a Unet submodule with skip connections.

		Parameters:
			outer_nc (int) -- the number of filters in the outer conv layer
			inner_nc (int) -- the number of filters in the inner conv layer
			input_nc (int) -- the number of channels in input images/features
			submodule (UnetSkipConnectionBlock) -- previously defined submodules
			outermost (bool)    -- if this module is the outermost module
			innermost (bool)    -- if this module is the innermost module
			norm_layer          -- normalization layer
			user_dropout (bool) -- if use dropout layers.
		"""
		super(UnetSkipConnectionBlock, self).__init__()
		self.outermost = outermost
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		if input_nc is None:
			input_nc = outer_nc
		downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
							 stride=2, padding=1, bias=use_bias)
		downrelu = nn.LeakyReLU(0.2, True)
		downnorm = norm_layer(inner_nc)
		uprelu = nn.ReLU(True)
		upnorm = norm_layer(outer_nc)

		if outermost:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
										kernel_size=4, stride=2,
										padding=1)
			down = [downconv]
			up = [uprelu, upconv, nn.Tanh()]
			model = down + [submodule] + up
		elif innermost:
			upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
										kernel_size=4, stride=2,
										padding=1, bias=use_bias)
			down = [downrelu, downconv]
			up = [uprelu, upconv, upnorm]
			model = down + up
		else:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
										kernel_size=4, stride=2,
										padding=1, bias=use_bias)
			down = [downrelu, downconv, downnorm]
			up = [uprelu, upconv, upnorm]

			if use_dropout:
				model = down + [submodule] + up + [nn.Dropout(0.5)]
			else:
				model = down + [submodule] + up

		self.model = nn.Sequential(*model)

	def forward(self, x):
		if self.outermost:
			return self.model(x)
		else:   # add skip connections
			return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
	"""Defines a PatchGAN discriminator"""

	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
		"""Construct a PatchGAN discriminator

		Parameters:
			input_nc (int)  -- the number of channels in input images
			ndf (int)       -- the number of filters in the last conv layer
			n_layers (int)  -- the number of conv layers in the discriminator
			norm_layer      -- normalization layer
		"""
		super(NLayerDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
			use_bias = norm_layer.func != nn.BatchNorm2d
		else:
			use_bias = norm_layer != nn.BatchNorm2d

		kw = 4
		padw = 1
		sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):  # gradually increase the number of filters
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]

		nf_mult_prev = nf_mult
		nf_mult = min(2 ** n_layers, 8)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]

		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
		self.model = nn.Sequential(*sequence)

	def forward(self, input):
		"""Standard forward."""
		return self.model(input)


class NLayerDiscriminatorCls(nn.Module):
	"""Defines a PatchGAN discriminator"""

	def __init__(self, input_nc, ndf=64, n_layers=3, n_class=3, norm_layer=nn.BatchNorm2d):
		"""Construct a PatchGAN discriminator

		Parameters:
			input_nc (int)  -- the number of channels in input images
			ndf (int)       -- the number of filters in the last conv layer
			n_layers (int)  -- the number of conv layers in the discriminator
			norm_layer      -- normalization layer
		"""
		super(NLayerDiscriminatorCls, self).__init__()
		if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
			use_bias = norm_layer.func != nn.BatchNorm2d
		else:
			use_bias = norm_layer != nn.BatchNorm2d

		kw = 4
		padw = 1
		sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):  # gradually increase the number of filters
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]

		nf_mult_prev = nf_mult
		nf_mult = min(2 ** n_layers, 8)
		sequence1 = [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]
		sequence1 += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

		sequence2 = [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]
		sequence2 += [
			nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]
		sequence2 += [
			nn.Conv2d(ndf * nf_mult, n_class, kernel_size=16, stride=1, padding=0, bias=use_bias)]


		self.model0 = nn.Sequential(*sequence)
		self.model1 = nn.Sequential(*sequence1)
		self.model2 = nn.Sequential(*sequence2)
		print(list(self.modules()))

	def forward(self, input):
		"""Standard forward."""
		feat = self.model0(input)
		# patchGAN output (1 * 62 * 62)
		patch = self.model1(feat)
		# class output (3 * 1 * 1)
		classl = self.model2(feat)
		return patch, classl.view(classl.size(0), -1)


class PixelDiscriminator(nn.Module):
	"""Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

	def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
		"""Construct a 1x1 PatchGAN discriminator

		Parameters:
			input_nc (int)  -- the number of channels in input images
			ndf (int)       -- the number of filters in the last conv layer
			norm_layer      -- normalization layer
		"""
		super(PixelDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
			use_bias = norm_layer.func != nn.InstanceNorm2d
		else:
			use_bias = norm_layer != nn.InstanceNorm2d

		self.net = [
			nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
			norm_layer(ndf * 2),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

		self.net = nn.Sequential(*self.net)

	def forward(self, input):
		"""Standard forward."""
		return self.net(input)


class HED(nn.Module):
	def __init__(self):
		super(HED, self).__init__()

		self.moduleVggOne = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.moduleVggTwo = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.moduleVggThr = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.moduleVggFou = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.moduleVggFiv = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=False)
		)

		self.moduleScoreOne = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.moduleScoreTwo = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.moduleScoreThr = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.moduleScoreFou = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.moduleScoreFiv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

		self.moduleCombine = nn.Sequential(
			nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid()
		)

	def forward(self, tensorInput):
		tensorBlue = (tensorInput[:, 2:3, :, :] * 255.0) - 104.00698793
		tensorGreen = (tensorInput[:, 1:2, :, :] * 255.0) - 116.66876762
		tensorRed = (tensorInput[:, 0:1, :, :] * 255.0) - 122.67891434

		tensorInput = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 1)

		tensorVggOne = self.moduleVggOne(tensorInput)
		tensorVggTwo = self.moduleVggTwo(tensorVggOne)
		tensorVggThr = self.moduleVggThr(tensorVggTwo)
		tensorVggFou = self.moduleVggFou(tensorVggThr)
		tensorVggFiv = self.moduleVggFiv(tensorVggFou)

		tensorScoreOne = self.moduleScoreOne(tensorVggOne)
		tensorScoreTwo = self.moduleScoreTwo(tensorVggTwo)
		tensorScoreThr = self.moduleScoreThr(tensorVggThr)
		tensorScoreFou = self.moduleScoreFou(tensorVggFou)
		tensorScoreFiv = self.moduleScoreFiv(tensorVggFiv)

		tensorScoreOne = nn.functional.interpolate(input=tensorScoreOne, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
		tensorScoreTwo = nn.functional.interpolate(input=tensorScoreTwo, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
		tensorScoreThr = nn.functional.interpolate(input=tensorScoreThr, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
		tensorScoreFou = nn.functional.interpolate(input=tensorScoreFou, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
		tensorScoreFiv = nn.functional.interpolate(input=tensorScoreFiv, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)

		return self.moduleCombine(torch.cat([ tensorScoreOne, tensorScoreTwo, tensorScoreThr, tensorScoreFou, tensorScoreFiv ], 1))

# class for VGG19 modle
# borrows largely from torchvision vgg
class VGG19(nn.Module):
	def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000):
		super(VGG19, self).__init__()
		self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
		self.init_weights = init_weights
		self.feature_mode = feature_mode
		self.batch_norm = batch_norm
		self.num_clases = num_classes
		self.features = self.make_layers(self.cfg, batch_norm)
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, num_classes),
		)
	# print('----------load the pretrained vgg net---------')
		# if not init_weights == None:
		# print('load the weights')
			# self.load_state_dict(torch.load(init_weights))


	def make_layers(self, cfg, batch_norm=False):
		layers = []
		in_channels = 3
		for v in cfg:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v
		return nn.Sequential(*layers)

	def forward(self, x):
		if self.feature_mode:
			module_list = list(self.features.modules())
			for l in module_list[1:27]:                 # conv4_4
				x = l(x)
		if not self.feature_mode:
			x = self.features(x)
			x = x.view(x.size(0), -1)
			x = self.classifier(x)

		return x

class Classifier(nn.Module):
	def __init__(self, input_nc, classes, ngf=64, num_downs=3, norm_layer=nn.BatchNorm2d, use_dropout=False, h=512, w=512, dim=4096):
		super(Classifier, self).__init__()
		self.input_nc = input_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		model = [nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias), nn.LeakyReLU(0.2, True)]
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, num_downs):
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			model += [
				nn.Conv2d(int(ngf * nf_mult_prev), int(ngf * nf_mult), kernel_size=4, stride=2, padding=1, bias=use_bias),
				norm_layer(int(ngf * nf_mult)),
				nn.LeakyReLU(0.2, True)
			]
		nf_mult_prev = nf_mult
		nf_mult = min(2 ** num_downs, 8)
		model += [
			nn.Conv2d(ngf * nf_mult_prev, ngf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
			norm_layer(ngf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]
		self.encoder = nn.Sequential(*model)

		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, dim),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(dim, dim),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(dim, classes),
		)

	def forward(self, x):
		ax = self.encoder(x)
		#print('ax',ax.shape) # (8, 512, 7, 7)
		ax = ax.view(ax.size(0), -1) # view -- reshape
		return self.classifier(ax)
