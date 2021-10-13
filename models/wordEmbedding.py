# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np
import json
import os
class WordEmbedding(nn.Module):
    def __init__(self, args):
        super(WordEmbedding, self).__init__()
        self.gpu = args.ifgpu
        self.embedding_dim = args.embedding_dim
        self.drop = nn.Dropout(args.embeddingdropout)
        # get word_alphabet
        root_path = os.path.join(args.data_root_dir, args.data_name)
        word2id_dict_path = os.path.join(root_path, "vocabulary.json")
        f = open(word2id_dict_path, 'r', encoding='utf-8')
        self.word_alphabet = json.load(f)
        f.close()
    #word Embedding
        self.word_embedding = nn.Embedding(len(self.word_alphabet), self.embedding_dim)
        if args.pretrain_word_embedding :
            pretrained_embedding = self.load_glove_embeddings(self.word_alphabet,self.embedding_dim)
            self.word_embedding.weight.data.copy_(
                pretrained_embedding)


        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(len(self.word_alphabet), self.embedding_dim)))


    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def load_glove_embeddings(self, data_word2id_dic, embedding_dim):
        """Loading the glove embeddings"""
        glove_word2id_path = "../data/glove_word2id.json"
        glove_embedding_path = "../data/glove_numpy.npy"
        f = open(glove_word2id_path, 'r')
        glove_word2id = json.load(f)
        f.close()
        glove_vecs = np.load(glove_embedding_path)
        embeddings = np.zeros((len(data_word2id_dic), embedding_dim))
        data_id2word_dic = {}
        for key, value in data_word2id_dic.items():
            data_id2word_dic[value] = key

        # data_vecs = np.zeros((len(data_word2id_dic), embedding_dim), dtype=np.float32)
        scale = np.sqrt(3.0 / embedding_dim)
        for i in range(len(data_word2id_dic)):
            if data_id2word_dic[i] in glove_word2id:
                embeddings[i] = glove_vecs[glove_word2id[data_id2word_dic[i]]]
            else:
                embeddings[i] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return torch.from_numpy(embeddings).float()


    def forward(self, word_inputs):
        word_embs = self.word_embedding(word_inputs)

        return word_embs
