import json
import os

class hyper_parameters(object):
	def __init__(self, input_time_step=20, output_time_step=50,
	             data_dir='data/',dataset='vehicle_ngsim',model_type='rnn',
	             num_class=3,coordinate_dim=2,inp_feat=('traj', 'speed')): #32

		self.input_time_step = input_time_step
		self.output_time_step = output_time_step
		self.coordinate_dim = coordinate_dim
		self.inten_num_class = num_class

		self.inp_feat = inp_feat
		self.out_feat = ('speed',)
		self.encoder_feat_dim = self.coordinate_dim * len(self.inp_feat) # x, v
		self.decoder_feat_dim = self.coordinate_dim * 1 # v

		self.data_dir = data_dir
		self.dataset = dataset
		self.model_type = model_type
		self.params_dict = {}


	def _set_default_dataset_params(self):
		if self.dataset=='human_kinect':
			self.inp_feat = ('traj', 'speed')
			self.input_time_step = 20
			self.output_time_step = 10
			self.inten_num_class = 12
			self.coordinate_dim = 3
			self.encoder_feat_dim = self.coordinate_dim * 2 # x,v
			self.decoder_feat_dim =self.coordinate_dim

		if self.dataset=='human_mocap':
			self.inp_feat = ('traj', 'speed')
			self.input_time_step = 20
			self.output_time_step = 10
			self.inten_num_class = 3
			self.coordinate_dim = 3
			self.encoder_feat_dim = self.coordinate_dim * 2
			self.decoder_feat_dim =self.coordinate_dim

		if self.dataset=='vehicle_holomatic':
			self.inp_feat = ('feature', 'speed')
			self.input_time_step = 20
			self.output_time_step = 50
			self.inten_num_class = 5
			self.coordinate_dim = 2
			self.traj_feature_dim = 8
			self.encoder_feat_dim =self.coordinate_dim + self.traj_feature_dim
			self.decoder_feat_dim = self.coordinate_dim

		if self.dataset=='vehicle_ngsim':
			self.inp_feat = ('feature', 'speed')
			self.input_time_step = 20
			self.output_time_step = 50
			self.inten_num_class = 3
			self.coordinate_dim = 2
			self.traj_feature_dim = 4
			self.encoder_feat_dim =self.coordinate_dim + self.traj_feature_dim
			self.decoder_feat_dim = self.coordinate_dim

	def train_param(self, param_dict=None):
		default_train_params = dict(
			dataset=self.dataset,
			data_path=self.data_dir + self.dataset + '.pkl',
			save_dir='output/'+self.dataset+'/'+self.model_type +'/',
			init_model=None,
			normalize_data=True,
			input_time_step=self.input_time_step,
			output_time_step=self.output_time_step,
			inp_feat = self.inp_feat,

			traj_intent_loss_ratio=[1, 0.0],  #TODO: originally [1, 0.1], traj loss : intent loss
			lr=0.01,
			lr_schedule='multistep',  # multistep
			lr_decay_epochs=[7, 14],
			lr_decay=0.1,
			epochs=20,
			batch_size=128,

			coordinate_dim=self.coordinate_dim,
			encoder=self.model_type,
			encoder_feat_dim=self.encoder_feat_dim,

			decoder=self.model_type,
			decoder_feat_dim = self.decoder_feat_dim,


			class_num=self.inten_num_class,
			pool_type='linear_attn',
			label_smooth=0.1,
			traj_attn_intent_dim=64,
		)
		if param_dict is None and 'train_param' in self.params_dict:
			param_dict = self.params_dict['train_param']
		params = self._overwrite_params(default_train_params, param_dict)
		params['log_dir'] = params['save_dir'] + 'log/'
		dir_split = params['log_dir'].replace('\\','/').split('/')
		base_dir=''
		for _path in dir_split:
			base_dir =os.path.join(base_dir,_path)
			if not os.path.exists(base_dir):
				os.mkdir(base_dir)

		if params['encoder'] == 'fc':
			params['pool_type'] = 'none'

		return params

	def encode_rnn_param(self, param_dict=None):
		default_rnn_params = dict(
			cell_type='gru',
			feat_dim=self.encoder_feat_dim,
			max_seq_len=self.input_time_step,
			hidden_size=64,
			num_layers=1,
			dropout_fc=0.,
			dropout_rnn=0.,
			bidirectional=False,
		)
		if param_dict is None and 'encode_rnn_param' in self.params_dict:
			param_dict = self.params_dict['encode_rnn_param']
		param = self._overwrite_params(default_rnn_params, param_dict)
		return param

	def encode_fc_param(self, param_dict=None):
		default_fc_params = dict(
			feat_dim=self.encoder_feat_dim,
			max_seq_len=self.input_time_step,
			hidden_dims=[64,64],#[128,128],#[128,128,64]
			dropout=0.,
		)
		if param_dict is None and 'encode_fc_param' in self.params_dict:
			param_dict = self.params_dict['encode_fc_param']
		param = self._overwrite_params(default_fc_params, param_dict)
		return param

	def decode_rnn_param(self, param_dict=None):
		default_rnn_params = dict(
			cell_type='gru',
			feat_dim = self.decoder_feat_dim,
			max_seq_len=self.output_time_step,
			hidden_size=64,
			num_layers=1,
			dropout_fc=0.,
			dropout_rnn=0.,
			attention=True,
		)
		if param_dict is None and 'decode_rnn_param' in self.params_dict:
			param_dict = self.params_dict['decode_rnn_param']
		param = self._overwrite_params(default_rnn_params, param_dict)
		return param

	def decode_fc_param(self, param_dict=None):
		default_fc_params = dict(
			feat_dim=self.decoder_feat_dim,
			max_seq_len=self.output_time_step,
			hidden_dims=[64,64],
			dropout=0.,
		)
		if param_dict is None and 'decode_fc_param' in self.params_dict:
			param_dict = self.params_dict['decode_fc_param']
		param = self._overwrite_params(default_fc_params, param_dict)
		return param

	def classifier_fc_param(self, param_dict=None):
		default_fc_params = dict(
			hidden_dims=[64],
			dropout=0.,
			num_class=self.inten_num_class,
		)
		if param_dict is None and 'classifier_fc_param' in self.params_dict:
			param_dict = self.params_dict['classifier_fc_param']
		param = self._overwrite_params(default_fc_params, param_dict)
		return param

	def print_params(self):
		print('train parameters:')
		t_param = self.train_param()
		print(t_param)
		print('encode_param:')
		encode_param = self.encode_fc_param() if t_param['encoder'] == 'fc' else self.encode_rnn_param()
		print(encode_param)
		print('decode_param:')
		decode_param = self.decode_fc_param() if t_param['decoder'] == 'fc' else self.decode_rnn_param()
		print(decode_param)
		print('classifier_fc_param:')
		print(self.classifier_fc_param())

	def _overwrite_params(self, old_param, new_param):
		if new_param is None:
			return old_param
		for k, v in new_param.items():
			old_param[k] = v
		return old_param

	def _save_parameters(self, log_dir=None):
		params_dict = {}
		params_dict['train_param'] = self.train_param()
		params_dict['encode_rnn_param'] = self.encode_rnn_param()
		params_dict['encode_fc_param'] = self.encode_fc_param()
		params_dict['decode_rnn_param'] = self.decode_rnn_param()
		params_dict['decode_fc_param'] = self.decode_fc_param()
		params_dict['classifier_fc_param'] = self.classifier_fc_param()

		if log_dir is None:
			log_dir = params_dict['train_param']['log_dir']

		with open(log_dir + 'hyper_parameters.json', 'w') as f:
			json.dump(params_dict, f)

	def _save_overwrite_parameters(self, params_key, params_value, log_dir=None):
		params_dict = {}
		params_dict['train_param'] = self.train_param()
		params_dict['encode_rnn_param'] = self.encode_rnn_param()
		params_dict['encode_fc_param'] = self.encode_fc_param()
		params_dict['decode_rnn_param'] = self.decode_rnn_param()
		params_dict['decode_fc_param'] = self.decode_fc_param()
		params_dict['classifier_fc_param'] = self.classifier_fc_param()

		params_dict[params_key] = params_value

		if log_dir is None:
			log_dir = params_dict['train_param']['log_dir']

		with open(log_dir + 'hyper_parameters.json', 'w') as f:
			json.dump(params_dict, f)

	def _load_parameters(self, log_dir=None):
		if log_dir is None:
			log_dir = self.train_param()['log_dir']

		with open(log_dir + 'hyper_parameters.json', 'r') as f:
			self.params_dict = json.load(f)


class adapt_hyper_parameters(object):
	def __init__(self, adaptor='none',adapt_step=1,log_dir=None):
		self.adaptor = adaptor
		self.adapt_step = adapt_step
		self.log_dir = log_dir
		self.params_dict = {}

		adaptor=adaptor.lower()
		if adaptor=='nrls' or adaptor=='mekf' or adaptor=='mekf_ma':
			self.adapt_param=self.mekf_param
		elif adaptor=='sgd':
			self.adapt_param=self.sgd_param
		elif adaptor=='adam':
			self.adapt_param=self.adam_param
		elif adaptor=='lbfgs':
			self.adapt_param=self.lbfgs_param

	def strategy_param(self,param_dict=None):
		default_params = dict(
			adapt_step=self.adapt_step,
			use_multi_epoch=True,
			multiepoch_thresh=(-1, -1),
		)
		if param_dict is None and 'strategy_param' in self.params_dict:
			param_dict = self.params_dict['strategy_param']
		params = self._overwrite_params(default_params, param_dict)
		return params

	def mekf_param(self, param_dict=None):
		default_params = dict(
			p0=1e-2,  # 1e-2
			lbd=1-1e-6,  # 1
			sigma_r=1,
			sigma_q=0,
			lr=1, # 1

			miu_v=0,  #momentum
			miu_p=0, # EMA of P
			k_p=1, #look ahead of P

			use_lookahead=False, # outer lookahead
			la_k=1,  # outer lookahead
			la_alpha=1,
		)
		if param_dict is None and 'mekf_param' in self.params_dict:
			param_dict = self.params_dict['mekf_param']
		params = self._overwrite_params(default_params, param_dict)
		return params

	def sgd_param(self, param_dict=None):
		default_params = dict(
			lr=1e-6,
			momentum=0.7,
			nesterov=False,

			use_lookahead=False, 	#look ahead
			la_k=5,
			la_alpha = 0.8,
		)
		if param_dict is None and 'sgd_param' in self.params_dict:
			param_dict = self.params_dict['sgd_param']
		param = self._overwrite_params(default_params, param_dict)
		return param

	def adam_param(self, param_dict=None):
		default_params = dict(
			lr=1e-6,
			betas=(0.1, 0.99),
			amsgrad=True,

			use_lookahead=False,	#look ahead
			la_k=5,
			la_alpha=0.8,
		)
		if param_dict is None and 'adam_param' in self.params_dict:
			param_dict = self.params_dict['adam_param']
		param = self._overwrite_params(default_params, param_dict)
		return param

	def lbfgs_param(self, param_dict=None):
		default_params = dict(
			lr=0.002,
			max_iter=20,
			history_size=100,

			use_lookahead=False, #look ahead
			la_k=1,
			la_alpha=1.0,
		)
		if param_dict is None and 'lbfgs_param' in self.params_dict:
			param_dict = self.params_dict['lbfgs_param']
		param = self._overwrite_params(default_params, param_dict)
		return param

	def print_params(self):
		print('adaptation optimizer:',self.adaptor)
		print('adaptation strategy parameters:')
		print(self.strategy_param())
		if self.adaptor=='none':
			print('no adaptation')
		else:
			print('adaptation optimizer parameters:')
			print(self.adapt_param())

	def _overwrite_params(self, old_param, new_param):
		if new_param is None:
			return old_param
		for k, v in new_param.items():
			old_param[k] = v
		return old_param

	def _save_parameters(self, log_dir=None):
		params_dict = {}
		params_dict['strategy_param'] = self.strategy_param()
		params_dict['mekf_param'] = self.mekf_param()
		params_dict['sgd_param'] = self.sgd_param()
		params_dict['adam_param'] = self.adam_param()
		params_dict['lbfgs_param'] = self.lbfgs_param()

		if log_dir is None:
			log_dir = self.log_dir

		with open(log_dir + 'adapt_hyper_parameters.json', 'w') as f:
			json.dump(params_dict, f)

	def _load_parameters(self, log_dir=None):
		if log_dir is None:
			log_dir = self.log_dir

		with open(log_dir + 'adapt_hyper_parameters.json', 'r') as f:
			self.params_dict = json.load(f)