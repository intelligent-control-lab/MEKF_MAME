
import torch

def get_prediction_on_batch(data, model, device='cpu'):
	x, y_traj, y_intent, start_decode, pred_start_pos,x_mask = data
	x = x.float().to(device)
	y_traj = y_traj.float().to(device)
	y_intent = y_intent.long().to(device)
	start_decode = start_decode.float().to(device)
	x_mask = x_mask.byte().to(device)
	pred_start_pos = pred_start_pos.float().to(device)

	pred_traj, pred_intent = model(src_seq=x, start_decode=start_decode, encoder_mask=x_mask)
	return x, pred_traj, y_traj, pred_intent, y_intent, pred_start_pos

def get_predictions(dataloader, model, device,data_size=-1):
	traj_hist, traj_preds, traj_labels, intent_preds, intent_labels, pred_start_pos = None,None,None,None,None,None

	for i, data in enumerate(dataloader, 0):
		x, pred_traj, y_traj, pred_intent, y_intent, start_pos = get_prediction_on_batch(data, model, device)
		if traj_hist is None:
			traj_hist = x
			traj_preds = pred_traj
			traj_labels = y_traj
			intent_preds = pred_intent
			intent_labels = y_intent
			pred_start_pos = start_pos
		else:
			traj_hist = torch.cat([traj_hist,x],dim=0)
			traj_labels = torch.cat([traj_labels, y_traj], dim=0)
			intent_labels = torch.cat([intent_labels, y_intent], dim=0)
			pred_start_pos = torch.cat([pred_start_pos, start_pos], dim=0)
			traj_preds = torch.cat([traj_preds, pred_traj], dim=0)
			intent_preds = torch.cat([intent_preds, pred_intent], dim=0)
		if data_size>0 and traj_hist.size(0)>data_size:
			break
	return traj_hist, traj_preds, traj_labels, intent_preds, intent_labels, pred_start_pos

def get_position(speed, start_pose=None, data_stats=None):
	if data_stats is None or start_pose is None:
		return speed
	speed = speed * data_stats['data_std'] + data_stats['data_mean']
	displacement = torch.cumsum(speed, dim=1)
	start_pose = torch.unsqueeze(start_pose, dim=1)
	position = displacement + start_pose
	return position