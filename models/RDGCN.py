import torch
import torch.nn as nn
from .wordEmbedding import WordEmbedding
from .wordHiddenRep import WordHiddenRep
from . extractfeature import ExtractATENLA
import os


class DynamicGraphConvolution(nn.Module):
    def __init__(self,args):
        super(DynamicGraphConvolution, self).__init__()
        self.num_nodes = args.num_nodes
        self.in_features = args.encoder_dim
        self.out_features = args.encoder_dim
        self.Embedding = WordEmbedding(args)
        self.encoder = WordHiddenRep(args)
        self.extractATENLA = ExtractATENLA(args)
        self.static_weight = nn.Sequential(
            nn.Conv1d(self.num_nodes, self.num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(self.in_features, self.in_features, 1)
        self.bn_global = nn.BatchNorm1d(self.in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(self.in_features * 2, self.num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(self.in_features, self.out_features, 1)
        self.mask_mat = nn.Parameter(torch.eye(self.num_nodes).float())
        self.last_linear = nn.Conv1d(self.out_features, self.num_nodes, 1)



        self.tmp_linear = nn.Conv1d(800, self.in_features, 1)


        self.gpu = args.ifgpu
        if self.gpu:

            self.conv_global = self.conv_global.cuda()
            self.bn_global = self.bn_global.cuda()
            self.conv_create_co_mat = self.conv_create_co_mat.cuda()
            self.dynamic_weight = self.dynamic_weight.cuda()
            self.last_linear = self.last_linear.cuda()
            self.Embedding = self.Embedding.cuda()
            self.encoder = self.encoder.cuda()
            self.extractATENLA = self.extractATENLA.cuda()
            self.mask_mat = nn.Parameter(torch.eye(self.num_nodes).float().to("cuda"))

        initiation = nn.init.xavier_uniform_
        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                m.weight.data = initiation(m.weight.data)
                if m.bias is not None:
                    m.bias.data = nn.init.zeros_(m.bias.data)




    def forward_construct_dynamic_graph(self, x):

        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))

        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, input_x,word_seq_lengths):
        Embedding_x = self.Embedding(input_x)
        sequence_output = self.encoder(Embedding_x,word_seq_lengths)
        x_at = self.extractATENLA(sequence_output)
        x = x_at

        dynamic_adj = self.forward_construct_dynamic_graph(x)
        tmp = x.transpose(1,2)

        x = self.forward_dynamic_gcn(x, dynamic_adj)
        tmp = torch.cat((tmp,x.transpose(1,2)),-1)
        x = x+x_at
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        tmp = torch.cat((tmp,x.transpose(1,2)),-1)
        x = x+x_at
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        tmp = torch.cat((tmp,x.transpose(1,2)),-1)
        x = self.tmp_linear(tmp.transpose(1,2))
        x = self.relu(x)
        out2 = self.last_linear(x)
        mask_mat = self.mask_mat.detach()
        out3 = (out2 * mask_mat).sum(-1)/out2.size(-1)

        return out3





