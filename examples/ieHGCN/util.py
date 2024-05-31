import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import pickle
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer



# current_path = os.path.dirname(__file__)
# sys.path.append(current_path + '/../data')

def load_dblp4area4057():
	hgcn_path = 'example/ieHGCN/DBLP/dblp4area4057_hgcn_0.2.pkl'
	# print('hgcn load: ', hgcn_path, '\n')
	with open(hgcn_path, 'rb') as in_file:
		(label, ft_dict, adj_dict) = pickle.load(in_file)
	
		adj_dict['p']['a'] = adj_dict['p']['a'].to_sparse()
		adj_dict['p']['c'] = adj_dict['p']['c'].to_sparse()
		adj_dict['p']['t'] = adj_dict['p']['t'].to_sparse()
		
		adj_dict['a']['p'] = adj_dict['a']['p'].to_sparse()
		adj_dict['c']['p'] = adj_dict['c']['p'].to_sparse()
		adj_dict['t']['p'] = adj_dict['t']['p'].to_sparse()

	# print("\nlabel", label)
	# print("\nft_dict", ft_dict)
	# print("\nadj_dict", adj_dict)
	return label, ft_dict, adj_dict

if __name__ == '__main__':
	load_dblp4area4057()