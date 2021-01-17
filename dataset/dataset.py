import os
from collections import Counter

import joblib
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import IPython

def data_time_split(data_list,params):
	input_time_step = params['input_time_step']
	output_time_step = params['output_time_step']
	trajs = data_list['traj']
	speeds = data_list['speed']
	features = data_list['feature']
	actions = data_list['action']
	x_traj,y_traj,x_traj_len=[],[],[]
	y_intent = []
	x_speed, y_speed = [],[]
	x_feature, y_feature = [],[]
	data_ids = []
	inds=np.arange(0,len(trajs))

	for ind in inds:
		traj = trajs[ind]
		speed = speeds[ind]
		feature = features[ind]
		action = actions[ind]
		begin=0
		end=input_time_step+output_time_step
		steps=len(traj)
		src_len = steps - output_time_step
		mask_now=False
		if src_len < input_time_step/4:
			continue

		if steps< end:
			mask_now = True
			pad_traj = np.array([traj[0]*0]*(end-steps))
			traj = np.concatenate([pad_traj,traj])
			pad_speed = np.array([speed[0] * 0] * (end - steps))
			speed = np.concatenate([pad_speed, speed])
			pad_feature= np.array([feature[0] * 0] * (end - steps))
			feature = np.concatenate([pad_feature, speed])
			pad_actions= np.array([action[0] * 0] * (end - steps))
			action = np.concatenate([pad_actions, action])
			steps = len(traj)

		while end<=steps:
			# input
			inp_traj = traj[begin:begin+input_time_step].reshape((input_time_step, -1))
			x_traj.append(inp_traj)
			data_ids.append(ind)
			if mask_now:
				x_traj_len.append(src_len)
			else:
				x_traj_len.append(len(inp_traj))

			inp_sp= speed[begin:begin+input_time_step].reshape((input_time_step, -1))
			x_speed.append(inp_sp)

			inp_feat= feature[begin:begin+input_time_step].reshape((input_time_step, -1))
			x_feature.append(inp_feat)

			# output
			out_traj = traj[begin+input_time_step:end].reshape((output_time_step, -1))
			y_traj.append(out_traj)

			out_sp= speed[begin+input_time_step:end].reshape((output_time_step, -1))
			y_speed.append(out_sp)

			out_feat= feature[begin+input_time_step:end].reshape((output_time_step, -1))
			y_feature.append(out_feat)

			y_intent.append(action[begin+input_time_step-1])

			begin += 1
			end += 1

	x_traj=np.array(x_traj)
	x_speed = np.array(x_speed)
	x_feature = np.array(x_feature)
	y_traj=np.array(y_traj)
	y_speed = np.array(y_speed)
	y_feature = np.array(y_feature)
	y_intent=np.array(y_intent)
	x_traj_len = np.array(x_traj_len)
	data_ids=np.array(data_ids) # for each window, describes the rollout number that it came from
	pred_start_pos = x_traj[:,-1]
	data ={'x_traj':x_traj,'x_speed':x_speed,'x_feature':x_feature,
	       'y_traj':y_traj,'y_speed':y_speed,'y_feature':y_feature,
	       'y_intent':y_intent,'pred_start_pos':pred_start_pos,
	       'x_traj_len':x_traj_len,'data_ids':data_ids}
	return data

def normalize_data(data, data_stats):
	new_data={}
	for k,v in data.items():
		if k in ['x_traj','x_speed','y_traj','y_speed','x_feature','y_feature']:
			mark = k.split('_')[-1]
			data_mean,data_std=data_stats[mark+'_mean'],data_stats[mark+'_std']
			new_data[k] = (v-data_mean)/data_std
		else:
			new_data[k] = v
	return new_data

class Trajectory_Data(Dataset):
	def __init__(self, params, mode='train',data_stats={}):
		self.mode = mode
		print(mode,'data preprocessing')
		cache_dir = params['log_dir']+mode+'.cache'
		if os.path.exists(cache_dir):
			print('loading data from cache',cache_dir)
			self.data = joblib.load(cache_dir)
		else:
			raw_data = joblib.load(params['data_path'])[mode]
			self.data = data_time_split(raw_data,params) # This just does windowing

			if mode=='train':
				data_stats['traj_mean'] = np.mean(self.data['x_traj'],axis=(0,1))
				data_stats['traj_std'] = np.std(self.data['x_traj'], axis=(0, 1))
				data_stats['speed_mean'] = np.mean(self.data['x_speed'],axis=(0,1))
				data_stats['speed_std'] = np.std(self.data['x_speed'], axis=(0, 1))
				data_stats['feature_mean'] = np.mean(self.data['x_feature'],axis=(0,1))
				data_stats['feature_std'] = np.std(self.data['x_feature'], axis=(0, 1))
			self.data['data_stats'] = data_stats
			if params['normalize_data']:
				if mode=='train':
					print('data statistics:')
					print(data_stats)
				self.data = normalize_data(self.data, data_stats)
			joblib.dump(self.data,cache_dir)

		enc_inp= None
		for feat in params['inp_feat']:
			dat = self.data['x_'+feat]
			if enc_inp is None:
				enc_inp = dat
			else:
				enc_inp = np.concatenate([enc_inp,dat],axis=-1)


		self.data['x_encoder'] = enc_inp
		self.data['y_decoder'] = self.data['y_speed']
		self.data['start_decode'] = self.data['x_speed'][:,-1]


		self.input_time_step = params['input_time_step']
		self.input_feat_dim = self.data['x_encoder'].shape[2]

		print(mode + '_data size:', len(self.data['x_encoder']))
		print('each category counts:')
		print(Counter(self.data['y_intent']))
		# print("In dataset.py")
		# IPython.embed()

	def __getitem__(self, index):
		x = self.data['x_encoder'][index]
		y_traj = self.data['y_decoder'][index]
		y_inten = self.data['y_intent'][index]
		start_decode = self.data['start_decode'][index] # this is for incremental decoding
		pred_start_pos = self.data['pred_start_pos'][index]
		x_len = self.data['x_traj_len'][index]
		x_mask = np.zeros(shape=x.shape[0],dtype=np.int)
		bias = self.input_time_step - x_len
		# left pad
		if bias>0:
			x_mask[:bias] = 1
			x[:bias] = 0

		return (x, y_traj, y_inten, start_decode, pred_start_pos, x_mask)

	def __len__(self):
		return len(self.data['x_encoder'])

def get_data_loader(params, mode='train',pin_memory=False):
	if mode == 'train':
		train_data = Trajectory_Data(params, mode='train')
		data_stats = train_data.data['data_stats']
		train_loader = torch.utils.data.DataLoader(
			train_data, batch_size=params['batch_size'], shuffle=True,
		 drop_last=True, pin_memory=pin_memory)

		valid_data = Trajectory_Data(params, mode='valid',data_stats=data_stats)
		valid_loader = torch.utils.data.DataLoader(
			valid_data, batch_size=params['batch_size'], shuffle=False,
			 drop_last=False, pin_memory=pin_memory)

		test_data = Trajectory_Data(params, mode='test',data_stats=data_stats)
		test_loader = torch.utils.data.DataLoader(
			test_data, batch_size=params['batch_size'], shuffle=False,
			drop_last=False, pin_memory=pin_memory)

		for k, v in data_stats.items():
			data_stats[k] = [float(x) for x in v]
		params['data_stats'] = data_stats
		params['print_step'] = max(1,len(train_loader) // 10)
		return train_loader, valid_loader, test_loader, params

	elif mode == 'test':
		data_stats = params['data_stats']
		for k,v in data_stats.items():
			data_stats[k] = np.array(v)
		test_data = Trajectory_Data(params, mode='test',data_stats=data_stats)
		test_loader = torch.utils.data.DataLoader(
			test_data, batch_size=params['batch_size'], shuffle=False,
			 num_workers=1,drop_last=False, pin_memory=pin_memory)
		return test_loader
	elif mode == 'valid':
		data_stats = params['data_stats']
		for k,v in data_stats.items():
			data_stats[k] = np.array(v)
		test_data = Trajectory_Data(params, mode='valid',data_stats=data_stats)
		test_loader = torch.utils.data.DataLoader(
			test_data, batch_size=params['batch_size'], shuffle=False,
			 num_workers=1,drop_last=False, pin_memory=pin_memory)
		return test_loader
	else:
		return None

