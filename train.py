# coding=utf-8

import os
import warnings

import torch
import IPython
from parameters import hyper_parameters
from dataset.dataset import get_data_loader
from models.model_factory import create_model
from utils.pred_utils import get_prediction_on_batch, get_predictions,get_position
from utils.train_utils import CrossEntropyLoss, get_lr_schedule

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('training with device:', device)


def evaluate(model, data_loader, criterion_traj, criterion_intend, params, epoch=0, mark='valid'):
	data_stats = {'data_mean': params['data_mean'], 'data_std': params['data_std']}
	print('[Evaluation %s Set] -------------------------------' % mark)
	out_str = "epoch: %d, " % (epoch)
	with torch.no_grad():
		dat = get_predictions(data_loader, model,  device)
		traj_hist, traj_preds, traj_labels, intent_preds, intent_labels, pred_start_pos = dat

		# TODO: replacing loss traj with MSE on de-normalized output
		loss_traj = criterion_traj(traj_preds, traj_labels)
		loss_traj = loss_traj.cpu().detach().numpy()

		# loss_traj = criterion_traj(traj_preds*data_stats["data_std"] + data_stats["data_mean"], traj_labels*data_stats["data_std"] + data_stats["data_mean"])
		# loss_traj = loss_traj.cpu().detach().numpy()

		traj_preds = get_position(traj_preds, pred_start_pos, data_stats)
		traj_labels = get_position(traj_labels, pred_start_pos, data_stats)
		mse = (traj_preds - traj_labels).pow(2).sum().float() / (traj_preds.size(0) * traj_preds.size(1))
		mse = mse.cpu().detach().numpy()
		# IPython.embed()

		# TODO: swapping what Abu calls mse and trajectory_loss
		# temp = mse
		# mse = loss_traj
		# loss_traj = temp

		out_str += "trajectory_loss: %.6f, trajectory_mse: %.6f, " % (loss_traj, mse)

		loss_intent = criterion_intend(intent_preds, intent_labels)
		loss_intent = loss_intent.cpu().detach().numpy()
		_, pred_intent_cls = intent_preds.max(1)
		label_cls = intent_labels
		acc = (pred_intent_cls == label_cls).sum().float() / label_cls.size(0)
		acc = acc.cpu().detach().numpy()
		out_str += "intent_loss: %.4f, intent_acc: %.4f, " % (loss_intent, acc)

	print(out_str)
	print('-------------------------------')

	log_dir = params['log_dir']
	if not os.path.exists(log_dir + '%s.tsv' % mark):
		# with open(log_dir + 'test.tsv', 'a') as f: # TODO: bug or intentional?
		with open(log_dir + '%s.tsv' % mark, 'a') as f:
			f.write('epoch\ttraj_loss\tintent_loss\tmse\tacc\n')

	with open(log_dir + '%s.tsv' % mark, 'a') as f:
		f.write('%05d\t%f\t%f\t%f\t%f\n' % (epoch, loss_traj, loss_intent, mse, acc))
	return acc, mse


def train_on_batch(data, model, optimizer, criterion_traj, criterion_intend, params, print_result=False, epoch=0,
                   iter=0):
	optimizer.zero_grad()
	x, pred_traj, y_traj, pred_intent, y_intent, pred_start_pos = get_prediction_on_batch(data, model, device)

	loss_traj = criterion_traj(pred_traj, y_traj)
	loss_intent = criterion_intend(pred_intent, y_intent)
	loss = params['traj_intent_loss_ratio'][0] * loss_traj + params['traj_intent_loss_ratio'][1] * loss_intent

	loss.backward()
	_ = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
	optimizer.step()

	if print_result:
		data_stats = {'data_mean': params['data_mean'], 'data_std': params['data_std']}
		out_str = "epoch: %d, iter: %d, loss:  %.4f " % (epoch, iter,loss.detach().cpu().numpy())

		pred_traj = get_position(pred_traj, pred_start_pos, data_stats)
		y_traj = get_position(y_traj, pred_start_pos, data_stats)
		mse = (pred_traj - y_traj).pow(2).sum().float() / (pred_traj.size(0) * pred_traj.size(1))
		mse = mse.cpu().detach().numpy()
		loss_traj_val = loss_traj.cpu().detach().numpy()
		# IPython.embed()
		out_str += "trajectory_loss: %.4f, trajectory_mse: %.4f, " % (loss_traj_val, mse)

		_, pred_intent_cls = pred_intent.max(1)
		label_cls = y_intent
		acc = (pred_intent_cls == label_cls).sum().float() / label_cls.size(0)
		acc = acc.cpu().detach().numpy()
		loss_intent_val = loss_intent.cpu().detach().numpy()
		out_str += "intent_loss: %.4f, intent_acc: %.4f, " % (loss_intent_val, acc)

		print(out_str)
		log_path = params['log_dir'] + 'train.tsv'
		if not os.path.exists(log_path):
			with open(log_path, 'a') as f:
				f.write('epoch\titer\ttraj_loss\tintent_loss\tmse\tacc\n')

		with open(log_path, 'a') as f:
			f.write('%05d\t%05d\t%f\t%f\t%f\t%f\n' % (epoch, iter, loss_traj_val, loss_intent_val, mse, acc))

	return loss


def train(params):
	train_params = params.train_param()

	train_loader, valid_loader, test_loader, train_params = get_data_loader(train_params, mode='train')
	# IPython.embed()
	params._save_overwrite_parameters(params_key='train_param', params_value=train_params)

	train_params['data_mean'] = torch.tensor(train_params['data_stats']['speed_mean'], dtype=torch.float).unsqueeze(
		0).to(device)
	train_params['data_std'] = torch.tensor(train_params['data_stats']['speed_std'], dtype=torch.float).unsqueeze(0).to(
		device)

	model = create_model(params)
	model = model.to(device)

	criterion_traj = torch.nn.MSELoss(reduction='mean').to(device)
	criterion_intend = CrossEntropyLoss(class_num=train_params['class_num'],
	                                    label_smooth=train_params['label_smooth']).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])

	scheduler = get_lr_schedule(train_params['lr_schedule'], train_params, optimizer)

	best_result = {'valid_acc': 0, 'valid_mse': 99999, 'test_acc': 0, 'test_mse': 99999, 'epoch': 0}
	print('begin to train')
	for epoch in range(1, train_params['epochs'] + 1):
		for i, data in enumerate(train_loader, 0):
			# IPython.embed()
			print_result = True if i % train_params['print_step'] == 0 else False
			train_on_batch(data, model, optimizer, criterion_traj, criterion_intend, params=train_params,
			               print_result=print_result, epoch=epoch, iter=i)


		save_model_path = os.path.join(train_params['save_dir'], 'model_%d.pkl' % (epoch))
		torch.save(model, save_model_path)
		print('save model to', save_model_path)


		model.eval()
		valid_acc, valid_mse = evaluate(model, valid_loader, criterion_traj, criterion_intend, params=train_params,
		                                epoch=epoch,
		                                mark='valid')
		test_acc, test_mse = evaluate(model, test_loader, criterion_traj, criterion_intend, params=train_params,
		                              epoch=epoch,
		                              mark='test')
		model.train()
		if valid_mse < best_result['valid_mse'] or valid_acc > best_result['valid_acc']:
			best_result['valid_mse'] = valid_mse
			best_result['valid_acc'] = valid_acc
			best_result['test_mse'] = test_mse
			best_result['test_acc'] = test_acc
			best_result['epoch'] = epoch

		if scheduler is not None:
			scheduler.step(epoch)

	print('Best Results (epoch %d):' % best_result['epoch'])
	print('validation_acc = %f, validation_mse = %f, test_acc = %f, test_mse = %f'
	      % (best_result['valid_acc'], best_result['valid_mse'], best_result['test_acc'], best_result['test_mse']))
	return model


def main():
	# TODO: modify params here
	params = hyper_parameters(dataset='vehicle_ngsim', model_type='fc')
	# params = hyper_parameters(dataset='vehicle_ngsim', model_type='rnn')
	params._set_default_dataset_params()
	params.print_params()
	train(params)


if __name__ == '__main__':
	main()
