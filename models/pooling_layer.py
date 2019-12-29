import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPooling(nn.Module):
	def forward(self, x, x_mask=None):
		if x_mask is None or x_mask.data.sum() == 0:
			return torch.max(x, 1)[0]
		else:
			lengths = (1 - x_mask).sum(1)
			return torch.cat([torch.max(i[:l], dim=0)[0].view(1, -1) for i, l in zip(x, lengths)], dim=0)

class AvgPooling(nn.Module):
	def forward(self, x, x_mask=None):
		if x_mask is None or x_mask.data.sum() == 0:
			return torch.mean(x, 1)
		else:
			lengths = (1 - x_mask).sum(1)
			return torch.cat([torch.mean(i[:l], dim=0)[0].view(1, -1) for i, l in zip(x, lengths)], dim=0)

class LastPooling(nn.Module):
	def __init__(self):
		super(LastPooling, self).__init__()

	def forward(self, x, x_mask=None):
		if x_mask is None or x_mask.data.sum() == 0:
			return x[:, -1, :]
		else:
			lengths = (1 - x_mask).sum(1)
			return torch.cat([i[l - 1, :] for i, l in zip(x, lengths)], dim=0).view(x.size(0), -1)

class LinearSeqAttnPooling(nn.Module):
	"""Self attention over a sequence:

	* o_i = softmax(Wx_i) for x_i in X.
	"""

	def __init__(self, input_size,bias=False):
		super(LinearSeqAttnPooling, self).__init__()
		self.linear = nn.Linear(input_size, 1,bias=bias)

	def forward(self, x, x_mask=None):
		"""
		Args:
			x: batch * len * hdim
			x_mask: batch * len (1 for padding, 0 for true)
		Output:
			alpha: batch * len
		"""
		# TODO why need contiguous
		x = x.contiguous()
		x_flat = x.view(-1, x.size(-1))
		scores = self.linear(x_flat).view(x.size(0), x.size(1))
		if x_mask is not None:
			scores.data.masked_fill_(x_mask.data, -float('inf'))
		alpha = F.softmax(scores, dim=-1)
		self.alpha = alpha
		return alpha.unsqueeze(1).bmm(x).squeeze(1)

class NoPooling(nn.Module):
	# placeholder for identity mapping
	def forward(self, x, x_mask=None):
		return x

