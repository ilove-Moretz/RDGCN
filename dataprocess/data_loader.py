
import os
import numpy as np
import torch.utils.data as data_utils
import torch
import re
from tqdm import tqdm
from collections import Counter
import itertools
import scipy.sparse as sp
import pickle
def clean_str(string):
    # remove stopwords
    # string = ' '.join([word for word in string.split() if word not in cachedStopWords])
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def pad_sentences(sentences, padding_word="<PAD/>", max_length=500):
    sequence_length_mx = min(max(len(x) for x in sentences), max_length)
    seq_every_length = []
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        seq_every_length.append(min(len(sentence),max_length))
        if len(sentence) < max_length:
            num_padding = sequence_length_mx - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:max_length]
        padded_sentences.append(new_sentence)
    return padded_sentences,np.array(seq_every_length)


def load_data_and_labels(data,num_nodes):
    x_text = [clean_str(doc['text']) for doc in data]
    x_text = [s.split(" ") for s in x_text]
    labels = [doc['catgy'] for doc in data]

    row_idx, col_idx, val_idx = [], [], []
    for i in tqdm(range(len(labels))):
        l_list = list(set(labels[i]))  # remove duplicate cateories to avoid double count
        for y in l_list:
            row_idx.append(i)
            col_idx.append(y)
            val_idx.append(1)
    m = max(row_idx) + 1
    # n = max(col_idx) + 1
    n = num_nodes  # category number
    Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
    return [x_text, Y]


def build_vocab(sentences, vocab_size=50000):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    # append <UNK/> symbol to the vocabulary
    vocabulary['<UNK/>'] = len(vocabulary)
    vocabulary_inv.append('<UNK/>')
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    x = np.array(
        [[vocabulary[word] if word in vocabulary else vocabulary['<UNK/>'] for word in sentence] for sentence in
         sentences])
    # x = np.array([[vocabulary[word] if word in vocabulary else len(vocabulary) for word in sentence] for sentence in sentences])
    return x




def process_ori_data(path):
    f = open(path, 'r')
    data_ori = f.readlines()
    f.close()
    data = []
    for item in data_ori:
        label_item = item.strip().split('\t')[0]
        labels = []
        # if len(label_item)!=53:
        #     print(label_item)
        #     print(len(label_item))
        for idx in range(len(label_item)):
            if label_item[idx] == "1":
                labels.append(idx)
        dic_item = {}
        dic_item['catgy'] = labels;
        dic_item['text'] = item.strip().split('\t')[1].strip()
        data.append(dic_item)
    return data


def load_data(data_path, max_length=500, vocab_size=50000,num_nodes=53):

    path_train = os.path.join(data_path, "train.txt")
    train = process_ori_data(path_train)
    path_test = os.path.join(data_path, "test.txt")
    test = process_ori_data(path_test)
    trn_sents, Y_trn = load_data_and_labels(train,num_nodes)
    tst_sents, Y_tst = load_data_and_labels(test,num_nodes)
    trn_sents_padded,trn_seq_lenth = pad_sentences(trn_sents, max_length=max_length)
    tst_sents_padded,tst_seq_lenth = pad_sentences(tst_sents, max_length=max_length)
    print("len:", len(trn_sents_padded), len(tst_sents_padded))

    vocabulary, vocabulary_inv = build_vocab(trn_sents_padded + tst_sents_padded, vocab_size=vocab_size)
    X_trn = build_input_data(trn_sents_padded, vocabulary)
    X_tst = build_input_data(tst_sents_padded, vocabulary)
    Y_trn = Y_trn[0:].toarray()
    Y_tst = Y_tst[0:].toarray()
    return X_trn, Y_trn, X_tst, Y_tst, trn_seq_lenth,tst_seq_lenth,vocabulary, vocabulary_inv

def load_test_data(args,data_path, max_length=500, vocab_size=50000,num_nodes=53):
    path_test = os.path.join(data_path, "test.txt")
    test = process_ori_data(path_test)
    tst_sents, Y_tst = load_data_and_labels(test,num_nodes)
    tst_sents_padded, tst_seq_lenth = pad_sentences(tst_sents, max_length=max_length)
    print("len:", len(tst_sents_padded))
    import json
    root_path =  os.path.join(args.data_root_dir,args.data_name)
    vocabulary_path = os.path.join(root_path,"vocabulary.json")
    vocabulary_inv_path = os.path.join(root_path,"vocabulary_inv.json")
    f = open(vocabulary_path, "r")
    vocabulary = json.load( f)
    f.close()
    f = open(vocabulary_inv_path, "r")
    vocabulary_inv = json.load(f)
    f.close()
    X_tst = build_input_data(tst_sents_padded, vocabulary)
    Y_tst = Y_tst[0:].toarray()
    _ = []
    return _, _, X_tst, Y_tst, _, tst_seq_lenth, vocabulary, vocabulary_inv




def make_data_loader(args, is_train=True):
    if is_train:
        ori_data_path = os.path.join(args.data_root_dir, args.data_ori_name)

        X_trn, Y_trn, X_tst, Y_tst, trn_seq_lenth, tst_seq_lenth, vocabulary, vocabulary_inv = load_data(ori_data_path,
                                                                                                         args.max_length,
                                                                                                         args.vocab_size,
                                                                                                         args.num_nodes)
        import json
        root_path = os.path.join(args.data_root_dir, args.data_name)
        vocabulary_path = os.path.join(root_path, "vocabulary.json")
        vocabulary_inv_path = os.path.join(root_path, "vocabulary_inv.json")
        f = open(vocabulary_path, "w")
        json.dump(vocabulary, f)
        f.close()
        f = open(vocabulary_inv_path, "w")
        json.dump(vocabulary_inv, f)
        f.close()

        val_data = X_tst
        val_label = Y_tst
        val_seq_lengh = tst_seq_lenth
        train_data = X_trn
        train_label = Y_trn
        train_seq_lengh = trn_seq_lenth
        val_dataset = data_utils.TensorDataset(torch.from_numpy(val_data).type(torch.LongTensor),
                                               torch.from_numpy(val_label).type(torch.LongTensor),
                                               torch.from_numpy(val_seq_lengh).type(torch.LongTensor))
        val_data_loader = data_utils.DataLoader(val_dataset, args.batch_size, shuffle=True, drop_last=False)

        train_dataset = data_utils.TensorDataset(torch.from_numpy(train_data).type(torch.LongTensor),
                                                 torch.from_numpy(train_label).type(torch.LongTensor),
                                                 torch.from_numpy(train_seq_lengh).type(torch.LongTensor))

        train_data_loader = data_utils.DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=False)
        return train_data_loader, val_data_loader
    else:

        ori_data_path = os.path.join(args.data_root_dir, args.data_ori_name)
        _, _, X_tst, Y_tst, _, tst_seq_lenth, vocabulary, vocabulary_inv = load_test_data(args,
                                                                                      ori_data_path,
                                                                                      args.max_length,
                                                                                      args.vocab_size,
                                                                                      args.num_nodes)

        val_data = X_tst
        val_label = Y_tst
        val_seq_lengh = tst_seq_lenth
        val_dataset = data_utils.TensorDataset(torch.from_numpy(val_data).type(torch.LongTensor),
                                               torch.from_numpy(val_label).type(torch.LongTensor),
                                               torch.from_numpy(val_seq_lengh).type(torch.LongTensor))
        val_data_loader = data_utils.DataLoader(val_dataset, args.batch_size, shuffle=True, drop_last=False)
        _=[]
        return _ ,val_data_loader









