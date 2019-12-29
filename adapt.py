# coding=utf-8

import os
import warnings

import joblib
import torch
import numpy as np
from dataset.dataset import get_data_loader
from adaptation.lookahead import Lookahead
from adaptation.mekf import MEKF_MA
from parameters import hyper_parameters, adapt_hyper_parameters
from utils.adapt_utils import online_adaptation
from utils.pred_utils import get_predictions,get_position

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
warnings.filterwarnings("ignore")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device =torch.device("cpu")
print('testing with device:', device)

rnn_layer_name = ['encoder.rnn.weight_ih_l0', 'encoder.rnn.bias_ih_l0',
                  'encoder.rnn.weight_hh_l0','encoder.rnn.bias_hh_l0',
                  'decoder.rnn.weight_ih_l0',  'decoder.rnn.bias_ih_l0',
                  'decoder.rnn.weight_hh_l0','decoder.rnn.bias_hh_l0',
                  'decoder.output_projection.weight', 'decoder.output_projection.bias']
fc_layer_name = ['encoder.layers.0.weight', 'encoder.layers.0.bias',
                 'encoder.layers.3.weight', 'encoder.layers.3.bias',
                 'decoder.layers.0.weight', 'decoder.layers.0.bias',
                 'decoder.layers.3.weight', 'decoder.layers.3.bias',
                 'decoder.output_projection.weight', 'decoder.output_projection.bias', ]


def adaptable_prediction(data_loader, model, train_params, device, adaptor, adapt_step=1):
	'''adaptation hyper param'''
	adapt_params = adapt_hyper_parameters(adaptor=adaptor, adapt_step=adapt_step, log_dir=train_params['log_dir'])
	adapt_params._save_parameters()
	adapt_params.print_params()

	adapt_weights = []
	if train_params['encoder'] == 'rnn':
		adapt_layers = rnn_layer_name[8:]
	else:
		adapt_layers = fc_layer_name[8:]
	print('adapt_weights:')
	print(adapt_layers)
	for name, p in model.named_parameters():
		if name in adapt_layers:
			adapt_weights.append(p)
			print(name, p.size())

	optim_param = adapt_params.adapt_param()
	if adaptor == 'mekf' or adaptor=='mekf_ma':
		optimizer = MEKF_MA(adapt_weights, dim_out=adapt_step * train_params['coordinate_dim'],
		                 p0=optim_param['p0'], lbd=optim_param['lbd'], sigma_r=optim_param['sigma_r'],
		                 sigma_q=optim_param['sigma_q'], lr=optim_param['lr'],
		                 miu_v=optim_param['miu_v'], miu_p=optim_param['miu_p'],
		                 k_p=optim_param['k_p'])
	elif adaptor == 'sgd':
		optimizer = torch.optim.SGD(adapt_weights, lr=optim_param['lr'], momentum=optim_param['momentum'],
		                            nesterov=optim_param['nesterov'])

	elif adaptor == 'adam':
		optimizer = torch.optim.Adam(adapt_weights, lr=optim_param['lr'], betas=optim_param['betas'],
		                             amsgrad=optim_param['amsgrad'])

	elif adaptor == 'lbfgs':
		optimizer = torch.optim.LBFGS(adapt_weights, lr=optim_param['lr'], max_iter=optim_param['max_iter'],
		                              history_size=optim_param['history_size'])
	else:
		raise NotImplementedError
	print('base optimizer configs:', optimizer.defaults)
	if optim_param['use_lookahead']:
		optimizer = Lookahead(optimizer, k=optim_param['la_k'], alpha=optim_param['la_alpha'])

	st_param = adapt_params.strategy_param()
	pred_result = online_adaptation(data_loader, model, optimizer, train_params, device,
                                    adapt_step=adapt_step,
                                    use_multi_epoch=st_param['use_multi_epoch'],
                                    multiepoch_thresh=st_param['multiepoch_thresh'])


	return pred_result


def test(params, adaptor='none', adapt_step=1):
	train_params = params.train_param()
	train_params['data_mean'] = torch.tensor(train_params['data_stats']['speed_mean'], dtype=torch.float).unsqueeze(
		0).to(device)
	train_params['data_std'] = torch.tensor(train_params['data_stats']['speed_std'], dtype=torch.float).unsqueeze(0).to(
		device)
	data_stats = {'data_mean': train_params['data_mean'], 'data_std': train_params['data_std']}

	model = torch.load(train_params['init_model'])
	model = model.to(device)
	print('load model', train_params['init_model'])

	data_loader = get_data_loader(train_params, mode='test')
	print('begin to test')
	if adaptor == 'none':
		with torch.no_grad():
			pred_result = get_predictions(data_loader, model,  device)
	else:
		pred_result = adaptable_prediction(data_loader, model, train_params, device, adaptor, adapt_step)

	traj_hist, traj_preds, traj_labels, intent_preds, intent_labels, pred_start_pos = pred_result
	traj_preds = get_position(traj_preds, pred_start_pos, data_stats)
	traj_labels = get_position(traj_labels, pred_start_pos, data_stats)
	intent_preds_prob = intent_preds.detach().clone()
	_, intent_preds = intent_preds.max(1)

	result = {'traj_hist': traj_hist, 'traj_preds': traj_preds, 'traj_labels': traj_labels,
	          'intent_preds': intent_preds,'intent_preds_prob':intent_preds_prob,
	          'intent_labels': intent_labels, 'pred_start_pos': pred_start_pos}

	for k, v in result.items():
		result[k] = v.cpu().detach().numpy()

	out_str = 'Evaluation Result: \n'

	num, time_step = result['traj_labels'].shape[:2]
	mse = np.power(result['traj_labels'] - result['traj_preds'], 2).sum() / (num * time_step)
	out_str += "trajectory_mse: %.4f, \n" % (mse)

	acc = (result['intent_labels'] == result['intent_preds']).sum() / len(result['intent_labels'])
	out_str += "action_acc: %.4f, \n" % (acc)

	print(out_str)
	save_path = train_params['log_dir'] + adaptor + str(adapt_step) + '_pred.pkl'
	joblib.dump(result, save_path)
	print('save result to', save_path)
	return result


def main(dataset='vehicle_ngsim', model_type='rnn', adaptor='mekf',adapt_step=1):
	save_dir = 'output/' + dataset + '/' + model_type + '/'
	model_path = save_dir + 'model_1.pkl'
	params = hyper_parameters()
	params._load_parameters(save_dir + 'log/')
	params.params_dict['train_param']['init_model'] = model_path
	params.print_params()
	test(params, adaptor=adaptor, adapt_step=adapt_step)


if __name__ == '__main__':
	main()

