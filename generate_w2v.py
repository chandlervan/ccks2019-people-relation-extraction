from gensim.models import Word2Vec
import pandas as pd
from collections import Counter
import jieba
import os

def get_w2v():
    os.mkdir('./w2v/')
    sent_train = pd.read_csv('./dataset/sent_data_train.txt', delimiter='\t', header=None)
    sent_dev = pd.read_csv('./dataset/sent_data_dev.txt', delimiter='\t', header=None)
    sent_test = pd.read_csv('./dataset/sent_data_test.txt', delimiter='\t', header=None)
    '''加载与训练预测相关的数据'''
    sent_train.columns = ['sent_id', 'e1', 'e2', 'text']
    sent_dev.columns = ['sent_id', 'e1', 'e2', 'text']
    sent_test.columns = ['sent_id', 'e1', 'e2', 'text']
    '''加载语料，用来训练词向量'''
    text = []
    with open('./dataset/text.txt', 'r', encoding='utf8') as f:
        for line in f.readlines():
            text.append(line.strip('\n'))
    all_text = pd.concat([sent_train['text'], sent_dev['text'], sent_test['text']])
    all_text = [str(v).lower() for v in all_text]
    text_seg = [v.split(' ') for v in all_text]
    all_word = []
    for v in text_seg:
        all_word += v
    word_cnt = Counter(all_word)
    for word in word_cnt.keys():
        jieba.add_word(word)
    text_seg_corpus = [jieba.lcut(v) for v in text]
    text_seg_all = text_seg + text_seg_corpus
    w2v = Word2Vec(size=128)
    w2v.build_vocab(text_seg_all)
    w2v.train(text_seg_all, total_examples=w2v.corpus_count, epochs=5)
    w2v.save('./w2v/w2v_model.w2v')