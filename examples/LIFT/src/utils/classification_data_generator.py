import sys
sys.path.append('./')
sys.path.append('./../')
import os, random, itertools
from attr import attributes
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from functools import partial
import openai, openml
from einops import rearrange, repeat

from utils import mnist
from utils import configs as cfgs


# # ################ Functions ################
# # 'Whether_of_not_the_TA_is_a_native_English_speaker','Course_instructor','Course','Summer_or_regular_semester','Class_size','Class_attribute'
# def data2text_feature_name(row, integer = False, label = True):
#     prompt = "Knowing a Teaching Assistant who's " 
#     if row['Whether_of_not_the_TA_is_a_native_English_speaker'] == 1:
#         prompt += 'an English speaker, '
#     elif row['Whether_of_not_the_TA_is_a_native_English_speaker'] == 2:
#         prompt += 'a non-English speaker, '
#     prompt += "who teaches the course %d with instructor %d during " % (row["Course"],row["Course_instructor"])
#     if row["Summer_or_regular_semester"] == 1:
#         prompt += "the summer session, "
#     else:
#         prompt += "the regular session, "
#     prompt += 'with %d students, what is the rating for this Teaching Assistant?' % row["Class_attribute"]
    
#     completion = "%d" % row['Class_attribute']
#     return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)

# def df2jsonl_feature_name(df, filename, integer = False):
#     jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name, integer = integer), axis = 1).tolist())
#     with open(os.path.join(filename), 'w') as f:
#         f.write(jsonl)

def data2text(row, integer = False, label = True, 
				  context = False, feature_names = None, target_names = None, init = '', end = ''):
	if context:
		# s0 = 'a non-English' if row[0] == 2 else 'an English'
		# s1 = 'summer' if row[3] == 1 else 'regular'
		# prompt = f"The teaching assistant is {s0} speaker. He teaches the course number {int(row[2])} with the instructor number {int(row[1])} during the {s1} semester, with {int(row[4])} students. What is his rating?"

		prompt = init
		for i in range(len(row)-label):
			v = row[i]
			if isinstance(v, int):
				prompt += "%s is %d, " % (feature_names[i], v)
			elif isinstance(v, str):
				prompt += "%s is %s, " % (feature_names[i], v)
			else:
				prompt += "%s is %.4f, " % (feature_names[i], v)
		prompt += end
		if label:
			v = row['y']
			if not isinstance(v, str):
				v = int(v)
			completion = "%s" % str(v) #target_names[int(row['y'])]
			final_prompt = "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
		else:
			final_prompt = f"{prompt}###"
	else:
		prompt = "When we have " 
		for i in range(len(row)-label):
			v = row[i]
			if isinstance(v, int):
				prompt += "x%d=%d, " % (i+1, v)
			elif isinstance(v, str):
				prompt += "x%d=%s, " % (i+1, v)
			else:
				prompt += "x%d=%.2f, " % (i+1, v)
		prompt += "What is this type? "

		if not label:
			final_prompt = f"{prompt}###"
		else:
			v = row['y']
			if not isinstance(v, str):
				v = int(v)
			completion = "%s" % str(v) #row['y']
			final_prompt = "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
	return final_prompt

def df2jsonl(df, filename, integer = False, 
			 context = False, feature_names = None, target_names = None, init = '', end = ''):
	jsonl = '\n'.join(df.apply(func = partial(data2text, 
											  integer = integer, 
											  context = context, 
											  feature_names = feature_names, 
											  target_names = target_names, 
											  init = init, 
											  end = end), axis = 1).tolist())
	fpath = os.path.join('data', filename)
	with open(fpath, 'w') as f:
		f.write(jsonl)
	return fpath

def array2prompts(X, integer = False,
				 context = False, feature_names = None, target_names = None, init = '', end = ''):
	return list(map(partial(data2text, 
							integer = integer, 
							label = False,
							context = context, 
							feature_names = feature_names, 
							target_names = target_names, 
							init = init, 
							end = end
						   ), X))

def data_split(X, y):
	if len(set(y)) == 2:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
	else:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
	# n = X.shape[0]
	# idx = np.arange(n)
	# random.shuffle(idx)
	# train_idx, valid_idx, test_idx = idx[:int(.6*n)], idx[int(.6*n):int(.8*n)], idx[int(.8*n):]
	# X_train, X_valid, X_test = X[train_idx], X[valid_idx], X[test_idx]
	# y_train, y_valid, y_test = y[train_idx], y[valid_idx], y[test_idx]
	return X_train, X_valid, X_test, y_train, y_valid, y_test

def gridX_generate(lb, ub, resolution = 50):
	# h = 0.02
#     lb = np.min(X, axis=0)[0]
#     ub = np.max(X, axis=0)[0]
	rang = ub - lb
	h = rang/resolution
	xx, yy = np.meshgrid(np.arange(lb, ub, h),
						np.arange(lb, ub, h))
	X_grid = np.c_[xx.ravel(), yy.ravel()]

	grid_prompts = array2prompts(X_grid)
	return pd.DataFrame(X_grid), grid_prompts


############### Load openML ####################
def load_openml(did=-1, ignore_cat=False, convert_cat=False):
	# fpath = f'data/openml/{did}_{normalize}.npy'
	# if os.path.isfile(fpath):
	# 	data = np.load(fpath, allow_pickle=True)
	# 	X, y = data.item()['X'], data.item()['y']
	# 	attribute_names = data.item()['attr']
	# else:	
		# dataset
	ds = openml.datasets.get_dataset(did)
	# values
	X, y, categorical_indicator, attribute_names = ds.get_data(target=ds.default_target_attribute)      
	# preprocess
	Xy = pd.concat([X,y], axis=1, ignore_index=True) # X & y concatenated together
	if ignore_cat:
		# non-cat
		non_categorial_indices = np.where(np.array(categorical_indicator) == False)[0] # find where categorical columns are  
		Xy = Xy.iloc[:, [*non_categorial_indices, -1]] # Slice columns -- ignore categorical X columns and add y column (-1)
		attribute_names = [attribute_names[i] for i in non_categorial_indices]   
	
	Xy.replace('?', np.NaN, inplace=True) # replace ? with NaNs    
	Xy = Xy[Xy.iloc[:, -1].notna()] # remove all the rows whose labels are NaN
	y_after_NaN_removal = Xy.iloc[:, -1]
	Xy.dropna(axis=1, inplace=True) # drop all the columns with missing entries
	Xy.dropna(inplace=True) # drop all the rows with missing entries
	assert((Xy.iloc[:, -1] == y_after_NaN_removal).all())
	X_raw, y = Xy.iloc[:, :-1], Xy.iloc[:, -1]

	# fine the categorical
	categorial_indices = np.where(np.array(categorical_indicator) == True)[0]
	scaler = StandardScaler()
	if len(categorial_indices) > 0:
		enc = OneHotEncoder(handle_unknown='ignore')     
		# Slice columns -- ignore categorical X columns and add y column (-1)
		X_cat = X.iloc[:, [*categorial_indices]] 
		X_cat_new = pd.DataFrame(enc.fit_transform(X_cat).toarray())
		X_cat_new = X_cat_new.values
		noncat_indices = np.where(np.array(categorical_indicator) == False)[0]
		
		if len(noncat_indices) > 0:
			X_noncat = X.iloc[:, [*noncat_indices]]
			X_noncat = scaler.fit_transform(X_noncat)
			X_norm = np.concatenate([X_noncat, X_cat_new], axis=1)
		else:
			X_norm = X_cat_new
		# X_norm = pd.concat([X_noncat, X_cat_new], axis=1, ignore_index=True)
		
		# attribute_names = [attribute_names[i] for i in non_categorial_indices]   
	else:
		X_norm =  scaler.fit_transform(X_raw)
	# if X.shape[0] == 0 or X.shape[1] == 0: # check if X is empty or not
	# 	print("Empty dataset")
	# else:
	# if normalize:
	# 	# else:
	# 	# X = X.to_numpy(dtype=np.float32)
	y = y.cat.codes.values
	#import pdb; pdb.set_trace()
	# assert(X.shape[1] == len(attribute_names))
	# np.save(fpath, {'X': X, 'y': y, 'attr': attribute_names})
	return y, X_raw.values, X_norm, attribute_names


# +
class DataGenerator(object):
	"""
	A class of functions for generating jsonl datasets for classification tasks.
	"""
	def __init__(self, did, seed = 123):
		self.seed = seed
		self.did = did
		self.fname = f'{did}'
		self.scaler = StandardScaler()

	def preprocess_data(self, data,  normalized=False, corruption_level=0, outliers=None):
		X, y = data['data'], data['target']
		if normalized:
			X = self.scaler.fit_transform(X)
		
		X_train, X_valid, X_test, y_train, y_valid, y_test = data_split(X, y)
		if outliers is not None:
			X_out, y_out = outliers
			X_train = np.concatenate([X_train, X_out], axis = 0)
			y_train = np.concatenate([y_train, y_out], axis = 0)
		if corruption_level > 0:
			# corrupt here
			n = len(y_train)
			m = int(n * corruption_level)
			inds = random.sample(range(1, n), m)
			for i in inds:
				y_train[i] = 1 - y_train[i] #binary
		
		train_df, val_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
		train_df['y'], val_df['y'], test_df['y'] = y_train, y_valid, y_test   

		return train_df, val_df, test_df

	def prepare_prompts(self, fnames, dfs, context=False, init=None, end=None, feature_names=None, target_names=None):
		X_test = dfs['test'].values[:, :-1]
		jsonl_files = {}
		for mode in ['train', 'val']:
			jsonl_files[mode] = df2jsonl(dfs[mode], fnames[mode],
						context = context, 
						feature_names = feature_names, 
						target_names = target_names, 
						init = init, 
						end = end)
		test_prompts = array2prompts(X_test,
			context = context, 
			feature_names = feature_names,
			target_names = target_names, 
			init = init, 
			end = end)
		
		return jsonl_files, test_prompts

	
	def load_synthetic_datatsets(self, did, with_prompt=False, **kwargs):
		# noise = kwargs['noise'] 
		# n = kwargs['num_samples']
		noise = 0.1
		n = 2000
		if did == cfgs.synthetic_data_ids['nnet']:
			# labeling_func, ranges=(-6, 6), corrupted =0.05
			lb, ub = kwargs['ranges']
			xy_min = [lb, lb]
			xy_max = [ub, ub]
			X = np.random.uniform(low=xy_min, high=xy_max, size=(n,2))
			y = kwargs['labeling_func'](X)
			# self.fname = '%s_n_%d_noise_%.2f'%(self.data_name, n, noise)
		elif did == cfgs.synthetic_data_ids['blobs']:
			# p, num_class = kwargs['p'], kwargs['num_class']
			p, num_class = 2, 4
#           # p, num_class
			X, y = datasets.make_blobs(n_samples = n, centers = num_class, n_features = p, random_state = self.seed)
			# self.fname =  '%s_n_%d_p_%d_class_%d'%(self.data_name, n, p, num_class)
		elif did in [cfgs.synthetic_data_ids['circles'], cfgs.synthetic_data_ids['twocircles']]:
			# factor = kwargs['factor']
			factor = 0.8
			if did == cfgs.synthetic_data_ids['circles']:
			# factors
				X, y = datasets.make_circles(n_samples = n, noise = noise, random_state = self.seed, factor = factor)
			else:
				X1, y1 = datasets.make_circles(n_samples = n//2, noise = noise, random_state = self.seed, factor = factor)
				X2, y2 = datasets.make_circles(n_samples = n//2, noise = noise, random_state = self.seed * 2, factor = factor)
				X2.T[0] += 3
				X, y = np.concatenate([X1, X2]), np.concatenate([y1, 1-y2])
			# self.fname = '%s_n_%d_noise_%.2f_factor_%.1f.jsonl'%(self.data_name, n, noise, factor)
		else:
			if did == cfgs.synthetic_data_ids['moons']:
				X, y = datasets.make_moons(n_samples = n, noise = noise, random_state = self.seed)
			elif did == cfgs.synthetic_data_ids['gaussian9cluster']:
				X, y = [], []
				label = 0
				for i in [-10, 0, 10]:
					for j in [-10, 0, 10]:
						label += 1
						mean = [i, j]
						cov = noise * np.diag(np.ones(2))
						X.append(np.random.multivariate_normal(mean, cov, n//9))
						y.append(np.ones(n//9) * label)
				X, y = np.concatenate(X), np.concatenate(y)
			# names
			# self.fname = '%s_n_%d_noise_%.2f'%(self.data_name, n, noise)
		
		return y, X, None, None
				
		data = {'data': X, 'target': y}
		train_df, val_df, test_df = self.preprocess_data(data, normalized=True)
		if with_prompt:
			dfs = {'train': train_df, 'val': val_df, 'test': test_df}
			jsonl_files, test_prompts = self.prepare_prompts(self.fnames, dfs)
			return jsonl_files, test_prompts, test_df
		else:
			return train_df, val_df, test_df

	
	def load_real_datasets(self, with_prompt=False):
		if self.data_name == 'iris':
			data = datasets.load_iris()
			init = 'Given a iris plant with '
			end = 'what is the type of it?'
		elif self.data_name == 'breastCancer':
			data = datasets.load_breast_cancer()
			init = 'Given a cell nuclei with '
			end = 'what is the condition?'
		elif self.data_name == 'wine':
			data = datasets.load_wine()
			init = 'Given the wine with '
			end = 'which class does it belong to?'
		
		train_df, val_df, test_df = self.preprocess_data(data, normalized=True)
		if with_prompt:
			dfs = {'train': train_df, 'val': val_df, 'test': test_df}
			fnames = {}
			for mode in ['train', 'val']:
				fnames[mode] = 'real/%d_context_%s_%s.jsonl' % (self.data_name, True, mode)
			jsonl_files, test_prompts = self.prepare_prompts(fnames, dfs, context=True, init=init, end=end)
			return jsonl_files, test_prompts, test_df
		return train_df, val_df, test_df

	def load_openml_datasets(self, did, with_prompt=False, use_name=False):
		if did < 10:
			# synthetic data
			train_df, val_df, test_df = self.load_synthetic_datatsets(did)
		else:
			y, X_raw, X_norm, att_names = load_openml(did)
			train_df, val_df, test_df = self.preprocess_data({'data': X, 'target': y}, normalized=False)

		if with_prompt:
			train_jsonl = 'openml/%d_context_%s_%s.jsonl' % (did, with_prompt, 'train')
			# if os.path.isfile(train_jsonl):
			# 	val_jsonl = 'openml/%d_context_%s_%s.jsonl' % (did, with_prompt, 'val') 
			# 	test_data = test_df.values
			# 	X_test, y_test = test_data[:, :-1], test_data[:, -1]
			# 	test_prompts = array2prompts(X_test, integer = False,
			# 	context = False, feature_names = None, target_names = None, init = '', end = '')
			# 	return train_jsonl, val_jsonl
			# else:
			dfs = {'train': train_df, 'val': val_df, 'test': test_df}
			# train_df.to_csv('./data/openml/%d_context_%s_train.csv' % (did, use_name))
			# val_df.to_csv('./data/openml/%d_context_%s_val.csv' % (did, use_name))
			# test_df.to_csv('./data/openml/%d_context_%s_test.csv' % (did, use_name))
			fnames = {}
			for mode in ['train', 'val']:
				fnames[mode] = 'openml/%d_context_%s_%s.jsonl' % (did, use_name, mode)
			if use_name:
				init = 'Given that'
				end = 'What is the category?'
				jsonl_files, test_prompts =  self.prepare_prompts(fnames, dfs, context=True, init=init, end=end, feature_names=att_names, target_names=att_names[-1])
			else:
				jsonl_files, test_prompts = self.prepare_prompts(fnames, dfs)
			# with open('./data/openml/%d_context_%s_test.txt' % (did, use_name),'w') as fpt:
			# 	for ele in test_prompts:
			# 		fpt.write(ele+'\n')
			return jsonl_files, test_prompts, test_df,val_df
		else:
			return train_df, val_df, test_df
	
	def img2seq(self, X, patch_size=16):
		p =patch_size 
		img = X.reshape(X.shape[0], 1, 28, 28)
		out = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
		out = out.max(axis=1)
		print(out.shape)
		return out
		
	
	def reshape_mnist(self, x):
		x = x.reshape(-1, 28, 28)
		x = x[:,5:-5,5:-5]
		x = x.reshape(-1, 18*18)
# 		x = x > 255//2
		return x
		# x_test = x_test.reshape(-1, 28, 28)
		# x_test = x_test[:,5:-5,5:-5]
		# x_test = x_train.reshape(-1, 18*18)

	def mnist(self, patch_size=4):
		random.seed(self.seed)
		X, y, X_test, y_test = mnist.load()
		
# 		X = self.img2seq(X, patch_size)
# 		X_test = self.img2seq(X_test, patch_size)
		X = self.reshape_mnist(X)
		X_test = self.reshape_mnist(X_test)

		idx = np.arange(60000)
		random.shuffle(idx)
		train_idx, valid_idx = idx[:50000], idx[50000:]
		X_train, X_valid = X[train_idx], X[valid_idx]
		y_train, y_valid = y[train_idx], y[valid_idx]
		train_df, val_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
		train_df['y'], val_df['y'], test_df['y'] = y_train, y_valid, y_test
		
		self.jsonl_files = {}
		self.jsonl_files['train'] = df2jsonl(train_df, f'mnist_train_{patch_size}.jsonl')
		self.jsonl_files['val'] = df2jsonl(val_df, f'mnist_valid_{patch_size}.jsonl')
		
		test_prompts = array2prompts(X_test)
		return train_df, val_df, test_df, test_prompts
	
	def permuted_mnist(self, patch_size=4):
		random.seed(self.seed)
		
		X, y, X_test, y_test = mnist.load()
		permutation = np.arange(X.shape[1])
		random.shuffle(permutation)
		X0 = np.apply_along_axis(lambda x: x[permutation], 1, X)
		X_test = np.apply_along_axis(lambda x: x[permutation], 1, X_test)
		
		X = self.img2seq(X, patch_size)
		X_test = self.img2seq(X_test, patch_size)

		idx = np.arange(60000)
		random.shuffle(idx)
		train_idx, valid_idx = idx[:50000], idx[50000:]
		X_train, X_valid = X[train_idx], X[valid_idx]
		y_train, y_valid = y[train_idx], y[valid_idx]
		train_df, val_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
		train_df['y'], val_df['y'], test_df['y'] = y_train, y_valid, y_test
		
		self.train_jsonl = df2jsonl(train_df, f'permuted_mnist_train_{patch_size}.jsonl')
		self.val_jsonl = df2jsonl(val_df, f'permuted_mnist_valid_{patch_size}.jsonl')
		
		test_prompts = array2prompts(X_test)
		return train_df, val_df, test_df, test_prompts
# -


