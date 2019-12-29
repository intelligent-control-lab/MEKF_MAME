# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNEncoder(nn.Module):
	"""RNN encoder."""

	def __init__(
			self, cell_type='lstm', feat_dim=3, hidden_size=128, num_layers=1,
			dropout_fc=0.1, dropout_rnn=0.0, bidirectional=False,**kwargs):
		super(RNNEncoder, self).__init__()
		self.cell_type = cell_type.lower()
		self.feat_dim = feat_dim
		self.num_layers = num_layers
		self.dropout_fc = dropout_fc
		self.dropout_rnn = dropout_rnn
		self.bidirectional = bidirectional
		self.hidden_size = hidden_size
		input_size = feat_dim

		if self.cell_type == 'lstm':
			self.rnn = LSTM(input_size=input_size, hidden_size=hidden_size,
			                num_layers=num_layers, bidirectional=bidirectional,
			                dropout=self.dropout_rnn)
		elif self.cell_type == 'gru':
			self.rnn = GRU(input_size=input_size, hidden_size=hidden_size,
			               num_layers=num_layers, bidirectional=bidirectional,
			               dropout=self.dropout_rnn)
		else:
			self.rnn = RNN(input_size=input_size, hidden_size=hidden_size,
			               num_layers=num_layers, bidirectional=bidirectional,
			               dropout=self.dropout_rnn)

		self.output_units = hidden_size
		if bidirectional:
			self.output_units *= 2

	def forward(self, src_seq):
		x = src_seq
		# B x T x C -> T x B x C
		x = x.transpose(0, 1)

		encoder_out, h_t = self.rnn(x)

		if self.cell_type == 'lstm':
			final_hiddens, final_cells = h_t
		else:
			final_hiddens, final_cells = h_t, None

		if self.dropout_fc>0:
			encoder_out = F.dropout(encoder_out, p=self.dropout_fc, training=self.training)

		if self.bidirectional:
			batch_size = src_seq.size(0)

			def combine_bidir(outs):
				out = outs.view(self.num_layers, 2, batch_size, -1).transpose(1, 2).contiguous()
				return out.view(self.num_layers, batch_size, -1)

			final_hiddens = combine_bidir(final_hiddens)
			if self.cell_type == 'lstm':
				final_cells = combine_bidir(final_cells)

		#  T x B x C -> B x T x C
		encoder_out = encoder_out.transpose(0, 1)
		final_hiddens = final_hiddens.transpose(0, 1)
		if self.cell_type == 'lstm':
			final_cells = final_cells.transpose(0, 1)

		return encoder_out, (final_hiddens, final_cells)


class AttentionLayer(nn.Module):
	def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
		super(AttentionLayer, self).__init__()

		self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
		self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

	def forward(self, input, source_hids, encoder_padding_mask=None):
		# input: bsz x input_embed_dim
		# source_hids: srclen x bsz x output_embed_dim

		# x: bsz x output_embed_dim
		x = self.input_proj(input)

		# compute attention
		attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

		# don't attend over padding
		if encoder_padding_mask is not None:
			attn_scores = attn_scores.float().masked_fill_(
				encoder_padding_mask,
				float('-inf')
			).type_as(attn_scores)  # FP16 support: cast to float and back

		attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

		# sum weighted sources
		x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)
		out = torch.cat((x, input), dim=1)
		x = F.tanh(self.output_proj(out))
		return x, attn_scores


class RNNDecoder(nn.Module):
	"""RNN decoder."""

	def __init__(
			self, cell_type='lstm', feat_dim=3, hidden_size=128, num_layers=1,
			dropout_fc=0.1, dropout_rnn=0.0, encoder_output_units=128, max_seq_len=10,
			attention=True, traj_attn_intent_dim=0,**kwargs):
		super(RNNDecoder, self).__init__()
		self.cell_type = cell_type.lower()
		self.dropout_fc = dropout_fc
		self.dropout_rnn = dropout_rnn
		self.hidden_size = hidden_size
		self.encoder_output_units = encoder_output_units
		self.num_layers = num_layers
		self.max_seq_len = max_seq_len
		self.feat_dim = feat_dim
		self.traj_attn_intent_dim =traj_attn_intent_dim
		input_size = feat_dim

		if encoder_output_units != hidden_size:
			self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
			if self.cell_type == 'lstm':
				self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
			else:
				self.encoder_cell_proj = None
		else:
			self.encoder_hidden_proj = self.encoder_cell_proj = None

		if self.cell_type == 'lstm':
			self.cell = LSTM
		elif self.cell_type == 'gru':
			self.cell = GRU
		else:
			self.cell = RNN

		self.rnn = self.cell(input_size=input_size,
		                     hidden_size=hidden_size, bidirectional=False,
		                     dropout=self.dropout_rnn,
		                     num_layers=num_layers)

		if attention:
			self.attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
		else:
			self.attention = None

		self.output_projection = Linear(hidden_size, feat_dim)

		if traj_attn_intent_dim>0:
			self.traj_attn_fc = Linear(hidden_size, traj_attn_intent_dim)

	def forward(self, encoder_out_list, start_decode=None, encoder_mask=None):

		x = start_decode.unsqueeze(1)
		bsz = x.size(0)

		# get outputs from encoder
		encoder_outs, (encoder_hiddens, encoder_cells) = encoder_out_list
		# B x T x C -> T x B x C
		encoder_outs = encoder_outs.transpose(0, 1)
		encoder_hiddens = encoder_hiddens.transpose(0, 1)
		if encoder_mask is not None:
			encoder_mask = encoder_mask.transpose(0, 1)
		prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]
		if self.cell_type == 'lstm':
			encoder_cells = encoder_cells.transpose(0, 1)
			prev_cells = [encoder_cells[i] for i in range(self.num_layers)]

		x = x.transpose(0, 1)
		srclen = encoder_outs.size(0)

		# initialize previous states

		if self.encoder_hidden_proj is not None:
			prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
		prev_hiddens = torch.stack(prev_hiddens, dim=0)
		if self.encoder_cell_proj is not None:
			prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
		if self.cell_type == 'lstm':
			prev_cells = torch.stack(prev_cells, dim=0)

		attn_scores = x.new_zeros(srclen, self.max_seq_len, bsz)
		inp = x
		outs = []
		hidden_outs=[]
		for j in range(self.max_seq_len):
			if self.cell_type == 'lstm':
				output, (prev_hiddens, prev_cells) = self.rnn(inp, (prev_hiddens, prev_cells))
			else:
				output, prev_hiddens = self.rnn(inp, prev_hiddens)
			output = output.view(bsz, -1)
			# apply attention using the last layer's hidden state
			if self.attention is not None:
				out, attn_scores[:, j, :] = self.attention(output, encoder_outs, encoder_mask)
			else:
				out = output
			if self.dropout_fc>0:
				out = F.dropout(out, p=self.dropout_fc, training=self.training)
			hid_out = out
			if self.traj_attn_intent_dim > 0:
				hid_out= self.traj_attn_fc(hid_out)
				hid_out = F.selu(hid_out)
			hidden_outs.append(hid_out)

			out = self.output_projection(out)
			# save final output
			outs.append(out)

			inp = out.unsqueeze(0)

		# collect outputs across time steps
		x = torch.cat(outs, dim=0).view(self.max_seq_len, bsz, self.feat_dim)
		hidden_outs = torch.cat(hidden_outs, dim=0).view(self.max_seq_len, bsz, -1)
		# T x B x C -> B x T x C
		x = x.transpose(1, 0)
		hidden_outs=hidden_outs.transpose(1, 0)
		# srclen x tgtlen x bsz -> bsz x tgtlen x srclen
		attn_scores = attn_scores.transpose(0, 2)

		# project back to input space

		return x, hidden_outs


def LSTM(input_size, hidden_size, **kwargs):
	m = nn.LSTM(input_size, hidden_size, **kwargs)
	for name, param in m.named_parameters():
		if 'weight' in name or 'bias' in name:
			param.data.uniform_(-0.1, 0.1)
	return m


def GRU(input_size, hidden_size, **kwargs):
	m = nn.GRU(input_size, hidden_size, **kwargs)
	for name, param in m.named_parameters():
		if 'weight' in name or 'bias' in name:
			param.data.uniform_(-0.1, 0.1)
	return m


def RNN(input_size, hidden_size, **kwargs):
	m = nn.RNN(input_size, hidden_size, **kwargs)
	for name, param in m.named_parameters():
		if 'weight' in name or 'bias' in name:
			param.data.uniform_(-0.1, 0.1)
	return m


def Linear(in_features, out_features, bias=True):
	"""Linear layer (input: N x T x C)"""
	m = nn.Linear(in_features, out_features, bias=bias)
	m.weight.data.uniform_(-0.1, 0.1)
	if bias:
		m.bias.data.uniform_(-0.1, 0.1)
	return m

