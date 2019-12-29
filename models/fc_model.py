import torch
import torch.nn as nn


class FCEncoder(nn.Module):
	"""Fully connected  encoder."""

	def __init__(self, feat_dim, max_seq_len, hidden_dims, dropout=0.0, act_fn=nn.ReLU, **kwargs):
		super(FCEncoder, self).__init__()
		self.feat_dim = feat_dim

		self.layers = nn.ModuleList([])
		input_dim = feat_dim * max_seq_len
		for hd in hidden_dims:
			self.layers.append(Linear(input_dim, hd))
			self.layers.append(nn.Dropout(dropout))
			self.layers.append(act_fn())
			input_dim = hd

		self.embed_inputs = None
		self.output_units = hidden_dims[-1]

	def forward(self, src_seq):
		batch_size = src_seq.size(0)
		x = src_seq.view(batch_size, -1)
		out_stack = []
		for ii, layer in enumerate(self.layers):
			x = layer(x)
			if (ii + 1) % 3 == 0:  # fc, drop out, relu
				out_stack.append(x)
		out_stack = torch.cat(out_stack, dim=1)
		return x, out_stack


class FCDecoder(nn.Module):
	"""FC decoder."""

	def __init__(self, feat_dim, max_seq_len, hidden_dims, dropout=0.0, act_fn=nn.ReLU,
	             encoder_output_units=10, traj_attn_intent_dim=0, **kwargs):
		super(FCDecoder, self).__init__()
		self.feat_dim = feat_dim
		self.max_seq_len = max_seq_len
		self.traj_attn_intent_dim = traj_attn_intent_dim

		self.layers = nn.ModuleList([])
		input_dim = encoder_output_units
		for hd in hidden_dims:
			self.layers.append(Linear(input_dim, hd))
			self.layers.append(nn.Dropout(dropout))
			self.layers.append(act_fn())
			input_dim = hd

		self.output_projection = Linear(input_dim, max_seq_len * feat_dim)

		if traj_attn_intent_dim > 0:
			self.traj_attn_fc = Linear(input_dim, traj_attn_intent_dim)

	def forward(self, encoder_outs):
		x = encoder_outs
		for layer in self.layers:
			x = layer(x)
		hidden_out = x
		if self.traj_attn_intent_dim > 0:
			hidden_out = self.traj_attn_fc(hidden_out)
		x = self.output_projection(x)
		x = x.view(-1, self.max_seq_len, self.feat_dim)
		return x, hidden_out


class FCClassifier(nn.Module):
	"""FC classifier."""

	def __init__(self, encoder_output_units, hidden_dims, dropout=0.0, act_fn=nn.ReLU,
	             num_class=11, traj_attn_intent_dim=0, **kwargs):
		super(FCClassifier, self).__init__()

		self.layers = nn.ModuleList([])
		input_dim = encoder_output_units + traj_attn_intent_dim
		for hd in hidden_dims:
			self.layers.append(Linear(input_dim, hd))
			self.layers.append(nn.Dropout(dropout))
			self.layers.append(act_fn())
			input_dim = hd

		self.output_projection = Linear(input_dim, num_class, bias=False)

	def forward(self, encoder_outs):
		x = encoder_outs
		for layer in self.layers:
			x = layer(x)
		hidden=x
		x = self.output_projection(x)
		return x,hidden


def Linear(in_features, out_features, bias=True):
	"""Linear layer (input: N x T x C)"""
	m = nn.Linear(in_features, out_features, bias=bias)
	m.weight.data.uniform_(-0.1, 0.1)
	if bias:
		m.bias.data.uniform_(-0.1, 0.1)
	return m
