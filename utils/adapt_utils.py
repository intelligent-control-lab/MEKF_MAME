from time import time
import numpy as np
import torch
from .pred_utils import get_prediction_on_batch
import IPython, copy

# TODO: this selects how much data to use
# data_size=100
# test set size: 1986
data_size=3000
# data_size = 400
def batch2iter_data(dataloader, device='cpu',data_size=data_size):
	traj_hist, traj_labels, intent_labels, start_decodes, pred_start_pos, x_mask, rollout_start_inds = None, None, None, None, None, None, [0]

	for i, data in enumerate(dataloader, 0):
		x, y_traj, y_intent, start_decode, start_pos, mask = data
		if traj_hist is None:
			traj_hist = x
			traj_labels = y_traj
			intent_labels = y_intent
			start_decodes = start_decode
			pred_start_pos = start_pos
			x_mask = mask
		else:
			traj_hist = torch.cat([traj_hist, x], dim=0)
			traj_labels = torch.cat([traj_labels, y_traj], dim=0)
			intent_labels = torch.cat([intent_labels, y_intent], dim=0)
			start_decodes = torch.cat([start_decodes, start_decode], dim=0)
			pred_start_pos = torch.cat([pred_start_pos, start_pos], dim=0)
			x_mask = torch.cat([x_mask, mask], dim=0)

		# rollout_start_inds.append(traj_hist.shape[0])
		if data_size>0 and 	traj_hist.size(0)>data_size:
			break

	# print(traj_hist.shape)
	traj_hist = traj_hist.float().to(device)
	traj_labels = traj_labels.float().to(device)
	intent_labels = intent_labels.float().to(device)
	start_decodes = start_decodes.float().to(device)
	pred_start_pos = pred_start_pos.float().to(device)
	x_mask = x_mask.byte().to(device)
	data = [traj_hist, traj_labels, intent_labels, start_decodes, pred_start_pos, x_mask]

	# print("at the end of batch2iterdata")
	# IPython.embed()
	return data


def online_adaptation(dataloader, model, optimizer, params, device,
                          adapt_step=1, use_multi_epoch=False,multiepoch_thresh=(0, 0), reset_after_rollout=True):
	optim_name = optimizer.__class__.__name__
	if optim_name == 'Lookahead':
		optim_name = optim_name + '_' + optimizer.optimizer.__class__.__name__
	print('optimizer:', optim_name)
	print('adapt_step:', adapt_step, ', use_multi_epoch:', use_multi_epoch,', multiepoch_thresh:', multiepoch_thresh),

	data = batch2iter_data(dataloader, device)
	traj_hist, traj_labels, intent_labels, start_decodes, pred_start_pos, x_mask = data
	batches = []
	for ii in range(len(pred_start_pos)):
		temp_batch=[]
		for item in data:
			temp_batch.append(item[[ii]])
		batches.append(temp_batch)

	# IPython.embed()
	windows_per_rollout = 400 - (params["output_time_step"] + params["input_time_step"]) + 1
	if reset_after_rollout:
		for i in range(6): # TODO: 10
			rollout_batch = batches[i*windows_per_rollout: (i+1)*windows_per_rollout]
			rollout_traj_preds, rollout_intent_preds = online_adaptation_single_rollout(model, rollout_batch, optimizer, optim_name, adapt_step, multiepoch_thresh, device)

			if i == 0:
				traj_preds = rollout_traj_preds
				intent_preds = rollout_intent_preds
			else:
				traj_preds = torch.cat((traj_preds, rollout_traj_preds), axis=0)
				intent_preds = torch.cat((intent_preds, rollout_intent_preds), axis=0)
	else:
		traj_preds, intent_preds = online_adaptation_single_rollout(model, batches, optimizer,
																					optim_name, adapt_step,
																					multiepoch_thresh, device)

	return traj_hist, traj_preds, traj_labels, intent_preds, intent_labels, pred_start_pos


def online_adaptation_single_rollout(model, batches, optimizer, optim_name, adapt_step, multiepoch_thresh, device):
	"""
	Returns a 3D Tensor
	"""
	traj_preds = []
	intent_preds = []

	temp_pred_list = []
	temp_label_list = []
	temp_data_list = []
	cnt = [0, 0, 0]
	cost_list = []
	post_cost_list=[]
	cost_diff_list = []

	t1 = time()

	for t in range(len(batches)):
		batch_data = batches[t]
		_, pred_traj, y_traj, pred_intent, _, _ = get_prediction_on_batch(batch_data, model, device)

		traj_preds.append(pred_traj[0].detach()[None])
		intent_preds.append(pred_intent[0].detach()[None])

		temp_pred_list += [pred_traj]
		temp_label_list += [y_traj]
		temp_data_list += [batch_data]
		if len(temp_pred_list) > adapt_step:
			# maintains a buffer of length adapt_step containing the past adapt_step measurements for supervised adaptation
			temp_pred_list = temp_pred_list[1:]
			temp_label_list = temp_label_list[1:]
			temp_data_list = temp_data_list[1:]

		if t < adapt_step - 1:
			continue

		Y = temp_label_list[0]
		Y_hat = temp_pred_list[0]
		full_loss =(Y - Y_hat).detach().pow(2).mean().cpu().numpy().round(6)
		cost_list.append(full_loss)

		Y_tau = Y[:, :adapt_step].contiguous().view((-1, 1))
		Y_hat_tau = Y_hat[:, :adapt_step].contiguous().view((-1, 1))
		err = (Y_tau - Y_hat_tau).detach()
		curr_cost = err.pow(2).mean().cpu().numpy()
		update_epoch = 1

		# IPython.embed()
		if 0 <= multiepoch_thresh[0] <= multiepoch_thresh[1]:
			if curr_cost< multiepoch_thresh[0]:
				update_epoch=1
			elif curr_cost< multiepoch_thresh[1]:
				update_epoch = 2
			else:
				update_epoch = 0
		cnt[update_epoch] += 1
		for cycle in range(update_epoch):
			def mekf_closure(index=0):
				optimizer.zero_grad()
				dim_out = optimizer.optimizer.state['dim_out'] if 'Lookahead' in optim_name else optimizer.state['dim_out']
				retain = index < dim_out - 1
				Y_hat_tau[index].backward(retain_graph=retain)
				return err

			def lbfgs_closure():
				optimizer.zero_grad()
				temp_data = temp_data_list[0]
				_, temp_pred_traj, temp_y_traj, _, _, _ = get_prediction_on_batch(temp_data, model, device)
				y_tau = temp_y_traj[:, :adapt_step].contiguous().view((-1, 1))
				y_hat_tau = temp_pred_traj[:, :adapt_step].contiguous().view((-1, 1))
				loss = (y_tau - y_hat_tau).pow(2).mean()
				loss.backward()
				return loss

			if 'MEKF' in optim_name:
				optimizer.step(mekf_closure)
			elif 'LBFGS' in optim_name:
				optimizer.step(lbfgs_closure)
			else:
				loss = (Y_tau - Y_hat_tau).pow(2).mean()
				loss.backward()
				optimizer.step()

		temp_data = temp_data_list[0]
		_, post_pred_traj, post_y_traj, _, _, _ = get_prediction_on_batch(temp_data, model, device)
		post_loss = (post_pred_traj - post_y_traj).detach().pow(2).mean().cpu().numpy().round(6)
		post_cost_list.append(post_loss)

		cost_diff_list.append(full_loss-post_loss)

		if t % 10 == 0:
			print('finished pred {}, time:{},  partial cost before adapt:{}, partial cost after adapt:{}'.format(t, time() - t1, full_loss,post_loss))
			t1 = time()

	# IPython.embed()
	print("avg cost improvement (should be +): %f +/- %f" % (np.mean(cost_diff_list), np.std(cost_diff_list)))
	print('avg_cost:', np.mean(cost_list))
	print('number of update epoch', cnt)
	traj_preds = torch.cat(traj_preds, axis=0)
	intent_preds = torch.cat(intent_preds, axis=0)
	return traj_preds, intent_preds
