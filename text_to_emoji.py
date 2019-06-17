import os
import sys
import time
import logging
import json
import numpy as np
import pickle
import pdb
import pandas as pd
import requests
import re
import random
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation,SpatialDropout1D,Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import BatchNormalization
from keras.regularizers import L1L2
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.models import load_model
from sklearn.model_selection import train_test_split
np.random.seed(42)


def extract_text():
	file = 'raw.pickle'
	response = requests.get("https://raw.githubusercontent.com/bfelbo/DeepMoji/master/data/PsychExp/raw.pickle")
	open(file, 'wb').write(response.content)
	data = pickle.load(open(file,'rb'),encoding='latin1')
	if os.path.exists('data.txt'):
		os.remove('data.txt')
	try:
		texts = [str(x) for x in data['texts']]
		labels = [x['label'] for x in data['info']]
		with open("data.txt", 'a') as txtfile: 
			for i in range(len(texts)):
				txtfile.write(np.array2string(labels[i]))
				txtfile.write(str(texts[i])+'\n')

	except Exception as e:
		texts = [x for x in data['texts']]
		labels = [x['label'] for x in data['info']]



def read_text_file(file_name):
	data_list  = []
	with open(file_name,'r') as f:
		for line in f:
			line = line.strip()
			label = ' '.join(line[:line.find("]")].strip().split())
			text = line[line.find("]")+1:].strip()
			data_list.append([label, text])

	return data_list

def extract_labels(text_list):
	label_list = []
	text_list = [text_list[i][0].replace('[','') for i in range(len(text_list))]
	label_list = [list(np.fromstring(text_list[i], dtype=float, sep=' ')) for i in range(len(text_list))]
	return label_list

def extract_text_msgs(text_list):
	msg_list = []
	msg_list = [text_list[i][1] for i in range(len(text_list))]
	return msg_list

def read_glove_vector(glove_file):
	with open(glove_file,'r',encoding='UTF-8') as file:
		words = set()
		word_to_vec = {}
		for line in file:
			line = line.strip().split()
			line[0] = re.sub('[^a-zA-Z]', '', line[0])
			if len(line[0]) > 0:
				words.add(line[0])
				try:
					word_to_vec[line[0]] = np.array(line[1:],dtype=np.float64)
				except:
					print('Error has occured')
					print('-'*50)
					print(line[1:])

		i = 1
		word_to_index = {}
		index_to_word = {}
		for word in sorted(words):
			word_to_index[word] = i
			index_to_word[i] = word
			i = i+1
	return word_to_index,index_to_word,word_to_vec

def sentences_to_indices(text_arr,word_to_index,max_len):
	arr_len = text_arr.shape[0]
	arr_indices = np.zeros((arr_len,max_len))
	for i in range(arr_len):
		sentence = text_arr[i].lower().split()
		j = 0
		for word in sentence:
			if word in word_to_index:
				arr_indices[i,j] = word_to_index[word]
				j = j+1

	return arr_indices


def create_embedding_layer(word_to_index,word_to_vec):
	corpus_len = len(word_to_index) + 1
	embed_dim = word_to_vec['word'].shape[0]

	embed_matrix = np.zeros((corpus_len,embed_dim))

	for word, index in word_to_index.items():
		embed_matrix[index,:] = word_to_vec[word]

	embedding_layer = Embedding(corpus_len, embed_dim)
	embedding_layer.build((None,))
	embedding_layer.set_weights([embed_matrix])

	return embedding_layer

def create_lstm_model(input_shape,embedding_layer):
	sentence_indices = Input(shape=input_shape, dtype=np.int32)
	embedding_layer =  embedding_layer
	embeddings = embedding_layer(sentence_indices)
	reg = L1L2(0.01, 0.01)

	X = Bidirectional(LSTM(128, return_sequences=True,bias_regularizer=reg,kernel_initializer='he_uniform'))(embeddings)
	X = BatchNormalization()(X)
	X = Dropout(0.5)(X)
	X = LSTM(64)(X)
	X = Dropout(0.5)(X)
	X = Dense(7, activation='softmax')(X)
	X =  Activation('softmax')(X)
	model = Model(sentence_indices, X)

	return model



if __name__ == '__main__':
	extract_text()
	textlist = read_text_file("data.txt")
	label_list = extract_labels(textlist)
	msg_list = extract_text_msgs(textlist)
	word_to_index,index_to_word,word_to_vec = read_glove_vector('glove.6B.50d.txt')
	x_train, x_test, y_train, y_test = train_test_split(msg_list, label_list,stratify = label_list,\
		test_size = 0.2, random_state = 123)
	tk = Tokenizer(lower = True, filters='')
	tk.fit_on_texts(msg_list)
	train_tokenized = tk.texts_to_sequences(x_train)
	test_tokenized = tk.texts_to_sequences(x_test)
	maxlen = 50
	X_train = pad_sequences(train_tokenized, maxlen = maxlen)
	X_test = pad_sequences(test_tokenized, maxlen = maxlen)
	if os.path.exists('tokenizer.pickle'):
		os.remove('tokenizer.pickle')
		with open('tokenizer.pickle', 'wb') as tokenizer:
			pickle.dump(tk, tokenizer, protocol=pickle.HIGHEST_PROTOCOL)

	embedding_layer = create_embedding_layer(word_to_index,word_to_vec)
	model = create_lstm_model((maxlen,),embedding_layer)
	print(model.summary())
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, np.array(y_train), epochs = 30, batch_size = 32, shuffle=True)
	model.save('emoji_model.h5')
	model = load_model('emoji_model.h5')
	loss, acc = model.evaluate(X_test, np.array(y_test))
	test_sent = tk.texts_to_sequences(['Feeling sad that my favourite cricketer has retired'])
	test_sent = pad_sequences(test_sent, maxlen = maxlen)
	pred = model.predict(test_sent)
	print(np.argmax(pred))



