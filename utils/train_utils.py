import torch


class CrossEntropyLoss(torch.nn.Module):
	""" Cross entropy that accepts label smoothing"""

	def __init__(self, class_num=12, label_smooth=0, size_average=True):
		super(CrossEntropyLoss, self).__init__()
		self.class_num = class_num
		self.size_average = size_average
		self.label_smooth = label_smooth

	def forward(self, input, target):
		logsoftmax = torch.nn.LogSoftmax()
		one_hot_target = torch.zeros(target.size()[0], self.class_num,device=target.device)
		one_hot_target = one_hot_target.scatter_(1, target.unsqueeze(1), 1)
		if self.label_smooth > 0:
			one_hot_target = (1 - self.label_smooth) * one_hot_target + self.label_smooth * (1 - one_hot_target)
		if self.size_average:
			return torch.mean(torch.sum(-one_hot_target * logsoftmax(input), dim=1))
		else:
			return torch.sum(torch.sum(-one_hot_target * logsoftmax(input), dim=1))


def get_lr_schedule(lr_schedule, params, optimizer):
	if lr_schedule == 'multistep':
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['lr_decay_epochs'],
		                                                 gamma=params['lr_decay'], )
	elif lr_schedule == 'cyclic':
		scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, params['lr'] / 5, params['lr'],
		                                              step_size_up=params['period'],
		                                              mode='triangular', gamma=1.0, )
	else:
		scheduler = None

	return scheduler
