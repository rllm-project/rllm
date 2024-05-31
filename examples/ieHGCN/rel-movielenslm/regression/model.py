import torch
import torch.nn as nn
import torch.nn.functional as F

class HeteGCNLayer(nn.Module):

	def __init__(self, net_schem, input_layer_shape, out_layer_shape, type_fusion, type_attention_size):
		super(HeteGCNLayer, self).__init__()
		
		self.net_schem = net_schem
		self.input_layer_shape = input_layer_shape
		self.out_layer_shape = out_layer_shape

		self.hete_agg = nn.ModuleDict()
		for k in net_schem:
			self.hete_agg[k] = HeteAggregateLayer(k, net_schem[k], input_layer_shape, out_layer_shape[k], type_fusion, type_attention_size)


	def forward(self, x_dict, adj_dict):
		
		ret_x_dict = {}
		for k in self.hete_agg.keys():
			ret_x_dict[k] = self.hete_agg[k](x_dict, adj_dict[k])

		return ret_x_dict



class HeteAggregateLayer(nn.Module):
	
	def __init__(self, curr_k, nb_list, input_layer_shape, output_shape, type_fusion, type_attention_size):
		super(HeteAggregateLayer, self).__init__()
		
		self.nb_list = nb_list
		self.curr_k = curr_k
		self.type_fusion = type_fusion
		
		self.W_rel = nn.ParameterDict()
		for k in nb_list:
			self.W_rel[k] = nn.Parameter(torch.FloatTensor(input_layer_shape[k], output_shape))
			nn.init.xavier_uniform_(self.W_rel[k].data, gain=1.414)
		
		self.w_self = nn.Parameter(torch.FloatTensor(input_layer_shape[curr_k], output_shape))
		nn.init.xavier_uniform_(self.w_self.data, gain=1.414)

		self.bias = nn.Parameter(torch.FloatTensor(1, output_shape))
		nn.init.xavier_uniform_(self.bias.data, gain=1.414)

		if type_fusion == 'att':
			self.w_query = nn.Parameter(torch.FloatTensor(output_shape, type_attention_size))
			nn.init.xavier_uniform_(self.w_query.data, gain=1.414)
			self.w_keys = nn.Parameter(torch.FloatTensor(output_shape, type_attention_size))
			nn.init.xavier_uniform_(self.w_keys.data, gain=1.414)
			self.w_att = nn.Parameter(torch.FloatTensor(2*type_attention_size, 1))
			nn.init.xavier_uniform_(self.w_att.data, gain=1.414)


	def forward(self, x_dict, adj_dict):

		self_ft = torch.mm(x_dict[self.curr_k], self.w_self)
		
		nb_ft_list = [self_ft]
		nb_name = [self.curr_k]
		for k in self.nb_list:
			nb_ft = torch.mm(x_dict[k], self.W_rel[k])
			nb_ft = torch.spmm(adj_dict[k], nb_ft)
			nb_ft_list.append(nb_ft)
			nb_name.append(k)
		
		if self.type_fusion == 'mean':
			agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mean(1)

		elif self.type_fusion == 'att':
			att_query = torch.mm(self_ft, self.w_query).repeat(len(nb_ft_list), 1)
			att_keys = torch.mm(torch.cat(nb_ft_list, 0), self.w_keys)
			att_input = torch.cat([att_keys, att_query], 1)
			att_input = F.dropout(att_input, 0.5, training=self.training)
			e = F.elu(torch.matmul(att_input, self.w_att))
			attention = F.softmax(e.view(len(nb_ft_list), -1).transpose(0,1), dim=1)
			agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mul(attention.unsqueeze(-1)).sum(1)
			# print('curr key: ', self.curr_k, 'nb att: ', nb_name, attention.mean(0).tolist())

		output = agg_nb_ft + self.bias

		return output



class HGCN(nn.Module):
	
	def __init__(self, net_schem, layer_shape, label_keys, type_fusion='att', type_attention_size=64):
		super(HGCN, self).__init__()
		
		self.hgc1 = HeteGCNLayer(net_schem, layer_shape[0], layer_shape[1], type_fusion, type_attention_size)
		self.hgc2 = HeteGCNLayer(net_schem, layer_shape[1], layer_shape[2], type_fusion, type_attention_size)
		self.hgc3 = HeteGCNLayer(net_schem, layer_shape[2], layer_shape[3], type_fusion, type_attention_size)
		self.hgc4 = HeteGCNLayer(net_schem, layer_shape[3], layer_shape[4], type_fusion, type_attention_size)
		

		self.embd2class = nn.ParameterDict()
		self.bias = nn.ParameterDict()
		self.label_keys = label_keys
		for k in label_keys:
			self.embd2class[k] = nn.Parameter(torch.FloatTensor(layer_shape[-2][k], layer_shape[-1][k]))
			nn.init.xavier_uniform_(self.embd2class[k].data, gain=1.414)
			self.bias[k] = nn.Parameter(torch.FloatTensor(1, layer_shape[-1][k]))
			nn.init.xavier_uniform_(self.bias[k].data, gain=1.414)


	def forward(self, ft_dict, adj_dict):

		x_dict = self.hgc1(ft_dict, adj_dict)
		x_dict = self.non_linear(x_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)
		
		x_dict = self.hgc2(x_dict, adj_dict)
		x_dict = self.non_linear(x_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)
		
		x_dict = self.hgc3(x_dict, adj_dict)
		x_dict = self.non_linear(x_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)
		
		x_dict = self.hgc4(x_dict, adj_dict)


		logits = {}
		embd = {}
		for k in self.label_keys:
			embd[k] = x_dict[k]
			logits[k] = torch.mm(x_dict[k], self.embd2class[k]) + self.bias[k]
		return logits, embd


	def non_linear(self, x_dict):
		y_dict = {}
		for k in x_dict:
			y_dict[k] = F.elu(x_dict[k])
		return y_dict


	def dropout_ft(self, x_dict, dropout):
		y_dict = {}
		for k in x_dict:
			y_dict[k] = F.dropout(x_dict[k], dropout, training=self.training)
		return y_dict