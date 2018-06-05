
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import pdb
import math
import numpy as np

def conv1x1(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=(1,1), 
		stride=stride, padding=0, bias=False);
def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), 
		stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
	def __init__(self, in_planes, out_planes, downsample=False):
		super(BasicBlock, self).__init__()

		self.in_planes = in_planes
		self.out_planes = out_planes

		self.conv1 = conv3x3(in_planes, out_planes)
		self.bn1 = nn.BatchNorm2d(out_planes)

		self.conv2 = conv3x3(out_planes, out_planes)
		self.bn2 = nn.BatchNorm2d(out_planes)

		self.relu = nn.ReLU(inplace=True)

		self.conv_skip = conv1x1(in_planes, out_planes)
		self.bn_skip = nn.BatchNorm2d(out_planes)
	def forward(self, x):
		residual = x.clone()

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)

		# Include skip connection
		if self.in_planes==self.out_planes:
			x = x + residual
		else:
			residual = self.conv_skip(residual)
			residual = self.bn_skip(residual)
			x = x + residual

		x = self.relu(x)

		return x

class LayerBlock(nn.Module):
	def __init__(self, in_planes, out_planes, num_basic_blocks):
		super(LayerBlock, self).__init__()

		temp = []
		for i in range(num_basic_blocks):
			if i==0:
				temp.append(BasicBlock(in_planes, out_planes))
			else:
				temp.append(BasicBlock(out_planes, out_planes))

		self.blocks = nn.ModuleList(temp)

		self.maxpool = nn.MaxPool2d((2,2), stride=2)

	def forward(self, x):
		for block in self.blocks:
			x = block(x)

		x = self.maxpool(x)

		return x

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		# img size 32x32

		self.conv = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
		# img size = (38-7)/2+1 = 31/2+1 = 16x16

		self.layer_blocks = nn.ModuleList([
				LayerBlock(32, 64, 10),
				# img size 8x8

				LayerBlock(64, 128, 10),
				# img size 4x4
			
				LayerBlock(128, 256, 10),
				# img size 2x2
			
				# self.layer_block4 = LayerBlock(128, 256, 7),
				# img size 1x1
		])

		self.avgpool = nn.AvgPool2d((2,2), stride=1)
		# img size = (2-2)/1 + 1 = 1x1

		fc_in_size = 256 * 1 * 1

		self.fc = nn.Linear(fc_in_size, 10)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal(m.weight.data)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def forward(self, x):
		x = self.conv(x)

		for layer_block in self.layer_blocks:
			x = layer_block(x)

		x = self.avgpool(x)

		x = x.view(x.size(0), -1)

		x = self.fc(x)

		return x
if __name__ == '__main__':
    model = MemNet()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable Params: " + str(params))
    # print(model)

