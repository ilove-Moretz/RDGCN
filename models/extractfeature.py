import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtractATENLA(nn.Module):
    def __init__(self, args):
        super(ExtractATENLA, self).__init__()
        self.args = args
        self.in_features = args.encoder_dim
        self.out_features = args.relationHeadDim
        self.k_linear = nn.Linear(args.encoder_dim, args.relationHeadDim, bias=True)
        self.q_linear = nn.Linear(args.relationHeadDim, args.num_nodes, bias=True)

    def forward(self, BLSTM_Hidden):
        K =  torch.tanh(self.k_linear(BLSTM_Hidden))
        A = self.q_linear(K)
        A = F.softmax(A, dim=-1)
        out = torch.bmm(BLSTM_Hidden.transpose(1,2),A)

        return out