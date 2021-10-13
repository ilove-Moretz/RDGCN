import os, sys, pdb

from models import get_model
from dataprocess.data_loader import make_data_loader
import warnings
from trainer import Trainer
import torch
import torch.backends.cudnn as cudnn
import random
import pickle
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='PyTorch Training for Multi-label Classification')

parser.add_argument('--seed', default=1000, type=int, help='seed for initializing training. ')

#fix config
dataname = "ENRON"
modelname = "RDGCN"
num_nodes_dict={"AAPD":54,"RCV1":103, "SLASHDOT": 22, "ENRON":53}
parser.add_argument('--encoderExtractor', default='LSTM', type=str)
parser.add_argument('--encoder_Bidirectional', default=True, type=bool, help='if use BiLSTM')
# data config
parser.add_argument('--data_root_dir', default='../data/', type=str, help='data path')
parser.add_argument('--data_name', default=dataname, type=str, help='data name')
parser.add_argument('--data_ori_name', default='{}_ori'.format(dataname), type=str, help='ori data file name')
parser.add_argument('--num_nodes', default=num_nodes_dict[dataname], type=int)
# model file config
parser.add_argument('--model_dir', default='../save_model/{}_{}'.format(modelname,dataname), type=str, help='save model path')
parser.add_argument('--continue_train', default=False, type=bool, help='if continue train')
parser.add_argument('--chechpoint_path', default='../save_model/testfile8/checkpoint.pth', type=str, help='checkpoint.pth')
parser.add_argument('--evaluate_chechpoint_path', default='../save_model/{}_{}/checkpoint_best.pth'.format(modelname,dataname), type=str, help='checkpoint.pth')
parser.add_argument('--modelname', default="{}_{}".format(modelname,dataname), type=str)
# train config
parser.add_argument('--evaluate', default=1, type=int, help='if evaluate')
parser.add_argument('--ifgpu', default=True, type=bool, help='if use gpu')
parser.add_argument('--epoch', default=500, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--pretrain_word_embedding', default=True, type=bool, help='if pretrain_word_embedding')
parser.add_argument('--if_inint_hidden', default=False, type=bool, help='if inint hidden')
# args config
parser.add_argument('--embedding_dim', default=300, type=int)
parser.add_argument('--vocab_size', default=50000, type=int)
parser.add_argument('--max_length', default=500, type=int)
parser.add_argument('--encoder_dim', default=200, type=int)
# parser.add_argument('--out_features', default=100, type=int)
parser.add_argument('--relationHeadDim', default=50, type=int)
parser.add_argument('--lr', default=0.001, type=float, help='if continue train')
parser.add_argument('--lstmdropout', default=0.5, type=float)
parser.add_argument('--embeddingdropout', default=0.5, type=float)
parser.add_argument('--cuda', default=0, type=int)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def main(args):
    args.data_ori_name = '{}_ori'.format(args.data_name)
    args.num_nodes = num_nodes_dict[args.data_name]
    args.modelname = "{}_{}_{}".format(modelname,args.data_name,str(args.seed))
    args.model_dir = '../save_model/{}'.format(args.modelname)
    args.evaluate_chechpoint_path = '../save_model/{}/checkpoint_best.pth'.format(args.modelname)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)


    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if args.evaluate==0:
        argsDict = args.__dict__
        with open(os.path.join(args.model_dir, 'setting.txt'), 'w') as f:
            f.writelines('------------------ start save args------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
        pickle_path = os.path.join(args.model_dir, 'setting.pickle')
        f = open(pickle_path,'wb')
        pickle.dump(args, f,0)
        f.close()
    else:
        pickle_path = os.path.join(args.model_dir, 'setting.pickle')
        f = open(pickle_path, 'rb')
        args = pickle.load(f)
        args.evaluate = 1





    if args.seed is not None:
        print('* absolute seed: {}'.format(args.seed))
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        np.random.seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    is_train = True if not args.evaluate else False
    train_loader, val_loader= make_data_loader(args, is_train=is_train)
    args.evaluate_chechpoint_path = '../save_model/{}/checkpoint_best.pth'.format(args.modelname)

    model, start_epoch, best_score = get_model(args)
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))





    trainer = Trainer(model, train_loader, val_loader, start_epoch,best_score,args)

    if is_train:
        trainer.train()
    else:
        trainer.evaluate()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
