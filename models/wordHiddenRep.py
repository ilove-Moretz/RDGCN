# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class WordHiddenRep(nn.Module):
    def __init__(self,args ):
        super(WordHiddenRep, self).__init__()
        print("build word embedding...")
        self.args = args
        self.input_size = args.embedding_dim

        self.drop = nn.Dropout(args.lstmdropout)

        self.hidden_dim = self.init_hidden_dim()
        # if args.encoder_Bidirectional:
        #     self.hidden_dim = args.encoder_dim // 2
        # else:
        #     self.hidden_dim = args.encoder_dim

        if self.args.if_inint_hidden:
            print("hidden init ")
            self.hidden = self.init_hidden()
        else:
            self.hidden = None

    # word hidden rep
        if args.encoderExtractor == "LSTM":
            self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=args.encoder_Bidirectional)
        else:
            if args.encoderExtractor == "GRU":
                self.GRU = nn.GRU(self.input_size, self.hidden_dim, num_layers=1, batch_first=True)
            else:
                print("Error char feature selection, please check parameter data.char_feature_extractor (LSTM).")
                exit(0)




    def init_hidden(self):
        if self.args.encoderExtractor == "GRU":
            return torch.randn(1, self.args.batch_size, self.hidden_dim).cuda()

        if self.args.encoderExtractor == "LSTM":
            if self.args.encoder_Bidirectional:
                return (torch.randn(2, self.args.batch_size, self.hidden_dim).cuda(),
                        torch.randn(2, self.args.batch_size, self.hidden_dim).cuda())
            else:
                return (torch.randn(1, self.args.batch_size, self.hidden_dim).cuda(),
                        torch.randn(1, self.args.batch_size, self.hidden_dim).cuda())

    def init_hidden_dim(self):
        if self.args.encoderExtractor == "GRU":
            return self.args.encoder_dim

        if self.args.encoder_Bidirectional:
            return self.args.encoder_dim // 2
        else:
            return self.args.encoder_dim


    def forward(self, embedding_represent,word_seq_lengths):


        word_seq_lengths = word_seq_lengths.squeeze(-1).cpu().numpy()
        packed_words = pack_padded_sequence(embedding_represent, word_seq_lengths, batch_first=True,enforce_sorted = False)
        if self.args.encoderExtractor == "LSTM":
            lstm_out, hidden = self.lstm(packed_words, self.hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            feature_out = self.drop(lstm_out)
        elif self.args.encoderExtractor == "GRU":
            GRU_out, hidden = self.GRU(packed_words, self.hidden)
            GRU_out,_ = pad_packed_sequence(GRU_out, batch_first=True)
            feature_out = self.drop(GRU_out)


        return feature_out
