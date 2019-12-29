import torch
from torch import nn

from .fc_model import FCEncoder, FCDecoder, FCClassifier
from .pooling_layer import AvgPooling, LastPooling, LinearSeqAttnPooling, NoPooling
from .rnn_model import RNNEncoder, RNNDecoder


class MultiTask_Model(nn.Module):
	def __init__(self, encoder_type, decoder_type,pool_type, params):
		super(MultiTask_Model, self).__init__()

		self.encoder_type = encoder_type
		self.pool_type = pool_type
		self.decoder_type = decoder_type
		self.params = params
		self.train_param=self.params.train_param()
		self.traj_attn_intent_dim = self.train_param['traj_attn_intent_dim']

		self.encoder = self._create_encoder(self.encoder_type)
		self.enc_out_units = self.encoder.output_units

		self.decoder = self._create_decoder(self.decoder_type)

		self.clf_pool = self._create_pooling(self.pool_type)
		self.classifier = self._create_decoder(decoder_type='classifier')

		if self.traj_attn_intent_dim>0:
			self.attn_pool = self._create_pooling(self.pool_type,input_size=self.traj_attn_intent_dim)


	def _create_encoder(self, encoder_type):
		# create encoder
		if encoder_type == 'rnn':
			rnn_param = self.params.encode_rnn_param()
			encoder = RNNEncoder(**rnn_param)
		else:
			fc_param = self.params.encode_fc_param()
			encoder = FCEncoder(**fc_param)
		return encoder

	def _create_pooling(self, pool_type,input_size=None):
		if input_size is None:
			input_size=self.enc_out_units
		if pool_type == 'mean' or pool_type == 'avg':
			pool = AvgPooling()
		elif pool_type == 'last':
			pool = LastPooling()
		elif pool_type == 'linear_attn':
			pool = LinearSeqAttnPooling(input_size=input_size)
		else:
			pool = NoPooling()
		return pool

	def _create_decoder(self, decoder_type):
		if decoder_type == 'rnn':
			rnn_params = self.params.decode_rnn_param()
			decoder = RNNDecoder(encoder_output_units=self.enc_out_units,traj_attn_intent_dim=self.traj_attn_intent_dim,
			                     **rnn_params)
		elif decoder_type == 'classifier':
			clf_params = self.params.classifier_fc_param()
			decoder = FCClassifier(encoder_output_units=self.enc_out_units,traj_attn_intent_dim=self.traj_attn_intent_dim,
			                       **clf_params)
		else:
			fc_param = self.params.decode_fc_param()
			decoder = FCDecoder(encoder_output_units=self.enc_out_units,traj_attn_intent_dim=self.traj_attn_intent_dim,
			                    **fc_param)

		return decoder

	def forward(self, src_seq,start_decode=None,encoder_mask=None):

		enc = self.encoder(src_seq)
		encoder_out, encoder_state= enc

		if self.decoder_type == 'rnn':
			out_traj, hidden_out_traj = self.decoder(enc, start_decode,encoder_mask=encoder_mask)
		else:
			out_traj,hidden_out_traj = self.decoder(encoder_out)


		clf_inp = self.clf_pool(encoder_out,x_mask=encoder_mask)
		if self.traj_attn_intent_dim>0:
			hidden_out_traj = self.attn_pool(hidden_out_traj)
			clf_inp = torch.cat([clf_inp, hidden_out_traj], dim=1)
		out_intent,_ = self.classifier(clf_inp)

		return out_traj, out_intent

def create_model(params):
	train_params = params.train_param()
	if train_params['init_model'] is not None:
		model = torch.load(train_params['init_model'])
		print('load model', train_params['init_model'])
	else:
		model = MultiTask_Model(
			encoder_type=train_params['encoder'],
			pool_type=train_params['pool_type'],
			decoder_type=train_params['decoder'],
			params=params)

	param_num = sum([p.data.nelement() for p in model.parameters()])
	print("Number of model parameters: {} M".format(param_num / 1024. / 1024.))
	model.train()

	return model
