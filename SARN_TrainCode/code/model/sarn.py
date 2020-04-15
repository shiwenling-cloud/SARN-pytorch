from model import common
import torch

import torch.nn as nn


def make_model(args, parent=False):
	return SARN(args)

class SAB(nn.Module):
	def __init__(self, channel):
		super(SAB, self).__init__()

		modules_body = []
		# feature channel downscale and upscale --> channel weight
		modules_body.append(nn.Conv2d(channel, 1, 1)),
		modules_body.append(nn.ReLU(inplace=True))

		modules_body.append(nn.Conv2d(1, 1, 8, stride=4, padding=2, bias=True))
		modules_body.append(nn.ReLU(inplace=True))
		# self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
		modules_body.append((common.patch_NonLocalBlock2D(1, 1)))

		modules_body.append(nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=True))
		modules_body.append(nn.Sigmoid())

		self.body = nn.Sequential(*modules_body)

	def forward(self, x):
		res = self.body(x)

		return res * x


## Bottleneck Spatial Attention Module (BSAM)
class BSAM(nn.Module):
	def __init__(
			self, conv, n_feat, kernel_size, d, reduction,
			bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
		super(BSAM, self).__init__()
		modules_body = []
		modules_body.append(conv(n_feat, n_feat * d, kernel_size, bias=bias))
		if bn: modules_body.append(nn.BatchNorm2d(n_feat))
		modules_body.append(act)
		modules_body.append(conv(n_feat * d, n_feat, kernel_size, bias=bias))
		modules_body.append(SAB(n_feat))
		self.body = nn.Sequential(*modules_body)
		self.res_scale = res_scale

	def forward(self, x):
		res = self.body(x)
		# res = self.body(x).mul(self.res_scale)
		res += x
		return res


## Residual Block (RB)
class ResidualBlock(nn.Module):
	def __init__(self, conv, n_feat, kernel_size, d, reduction, act, res_scale, n_BSAMs):
		super(ResidualBlock, self).__init__()
		modules_body = []
		for i in range(n_BSAMs):
			modules_body.append(
				BSAM(conv, n_feat, kernel_size, d, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

		modules_body.append(conv(n_feat, n_feat, kernel_size))
		self.body = nn.Sequential(*modules_body)

	def forward(self, x):
		res = self.body(x)
		res += x
		return res


## Spatial Attention Residual Network (SARN)
class SARN(nn.Module):
	def __init__(self, args, conv=common.default_conv):
		super(SARN, self).__init__()

		n_resblocks = args.n_resblocks
		n_BSAMs = args.n_BSAMs
		n_feats = args.n_feats
		kernel_size = 3
		reduction = args.reduction
		scale = args.scale[0]
		act = nn.ReLU(True)

		# RGB mean for DIV2K
		rgb_mean = (0.4488, 0.4371, 0.4040)
		rgb_std = (1.0, 1.0, 1.0)
		self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

		# define head module
		modules_head = [conv(args.n_colors, n_feats, kernel_size)]
		# define body module
		modules_body = []
		for _ in range(3):
			modules_body.append(ResidualBlock(
				conv, n_feats, kernel_size, 1, reduction, act=act, res_scale=args.res_scale, n_BSAMs=n_BSAMs))
		for _ in range(n_resblocks - 3):
			modules_body.append(
				ResidualBlock(
					conv, n_feats, kernel_size, 4, reduction, act=act, res_scale=args.res_scale,
					n_BSAMs=n_BSAMs))

		modules_body.append(conv(n_feats, n_feats, kernel_size))

		# define tail module
		modules_tail = [
			common.Upsampler(conv, scale, n_feats, act=False),
			conv(n_feats, args.n_colors, kernel_size)]

		self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

		self.head = nn.Sequential(*modules_head)
		self.body = nn.Sequential(*modules_body)
		self.tail = nn.Sequential(*modules_tail)

	def forward(self, x):
		x = self.sub_mean(x)
		x = self.head(x)

		res = self.body(x)
		res += x

		x = self.tail(res)
		x = self.add_mean(x)

		return x

	def load_state_dict(self, state_dict, strict=False):
		own_state = self.state_dict()
		for name, param in state_dict.items():
			if name in own_state:
				if isinstance(param, nn.Parameter):
					param = param.data
				try:
					own_state[name].copy_(param)
				except Exception:
					if name.find('tail') >= 0:
						print('Replace pre-trained upsampler to new one...')
					else:
						raise RuntimeError('While copying the parameter named {}, '
										   'whose dimensions in the model are {} and '
										   'whose dimensions in the checkpoint are {}.'
										   .format(name, own_state[name].size(), param.size()))
			elif strict:
				if name.find('tail') == -1:
					raise KeyError('unexpected key "{}" in state_dict'
								   .format(name))

		if strict:
			missing = set(own_state.keys()) - set(state_dict.keys())
			if len(missing) > 0:
				raise KeyError('missing keys in state_dict: "{}"'.format(missing))
