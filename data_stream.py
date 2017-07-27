from __future__ import print_function

import re
import numpy as np
import json
import re
import pickle

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

def find_all(a_str, sub):
	start = 0
	while True:
		start = a_str.find(sub, start)
		if start == -1: return
		yield start
		start += len(sub)  # use start += 1 to find overlapping matches


def get_batches(inpath=None,word_vocab = None, char_vocab = None, batch_size = 60,
	 isShuffle=False, isLoop=False, isSort=True, max_char_per_word=10, max_sent_length=200):

	with open('/media/amit/FE42C05442C01377/Ujjawal/edge/data/train-v1.1.json', 'rb') as f:
		dataset = json.load(f)

	for article in dataset['data']:
		for para in article['paragraphs']:
			para['context'] = para['context'].replace(u'\u000A', '')
			para['context'] = para['context'].replace(u'\u00A0', ' ')
			try:
				context = para['context'].encode('utf-8').lower()
				context_wp = re.sub(r'[^\w\s]','',context)
			except:
				continue

			questions = []
			passages = []
			starts = []
			stops = []

			for qa in para['qas']:
				for answer in qa['answers']:
					answer['text'] = answer['text'].replace(u'\u00A0', ' ')
					text = answer['text']

					try:
						text = text.encode('utf-8').lower()
						text_wp = re.sub(r'[^\w\s]','',text)
					except:
						continue

					if(text_wp in context_wp):
						a_start = context_wp.find(text_wp)
						a_stop = context_wp.find(text_wp) + len(text_wp)
					else:
						continue

					questions.append(get_vectors_q(qa['question']))
					passages.append(get_vectors(context_wp))
					starts.append(a_start)
					stops.append(a_stop)
			
			yield [questions,passages,starts,stops]



def get_vectors_q(string):
	words = tokenizer.tokenize(string)
	z = [0]*300
	if(len(words)>=15):
		return get_vectors(words[:15])
	else:
		vect = list(get_vectors(words))
		for i in range(15-len(words)):
			vect.append(np.array(z))
		return np.asarray(vect)

def get_vectors(words):
	embedding = load_embedding()
	z = [0]*300
	zeros = np.array(z)
	vect = []
	for word in words:
		try:
			vect.append(np.array(embedding[word]))
		except:
			vect.append(np.array(zeros))

	return np.asarray(vect)

def load_embedding():
	with open('embedding.pkl', 'r') as f:
		embedding = pickle.load(f)
	return embedding


"""
def get_batches(batch_size):
	questions , passages,starts,stops , vocab_size = prepare_data()

	length = len(questions)

	for i in range(0:int(length//batch_size)):
		yield questions[i*batch_size : (i+1)*batch_size] , passages[i*batch_size : (i+1)*batch_size],starts[i*batch_size : (i+1)*batch_size], stops[i*batch_size : (i+1)*batch_size]



					if context[answer_start : answer_start + len(text)] == text:
						if text.lstrip() == text:
							pass
						else:
							answer_start += len(text) - len(text.lstrip())
							answer['answer_start'] = answer_start
							text = text.lstrip()
							answer['text'] = text
					else:
						text = text.lstrip()
						answer['text'] = text
						starts = list(find_all(context, text))
						if len(starts) == 1:
							answer_start = starts[0]
						elif len(starts) > 1:
							new_answer_start = min(starts, key=lambda s: abs(s - answer_start))
							loc_diffs.append(abs(new_answer_start - answer_start))
							answer_start = new_answer_start
						else:
							continue # CHECK THIS
						answer['answer_start'] = answer_start

with open('', 'r') as f:
	data = json.load(f)
	
i=0


for article in data['data']:
	for para in article['paragraphs']:
		for k in para['qas']:
			questions.append(k['question'])
			answers.append(k['answers'][0]['text'])
			start_index.append(k['answers'][0]['answer_start'])
			passages.append(para['context'])
			print(h.split_doc(para['context']))



"""
