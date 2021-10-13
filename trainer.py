import time
from tqdm import tqdm
import torch
import numpy as np
import os
import shutil
from util import precision_k,evaluation_prf
import json
import torch.backends.cudnn as cudnn
import random
import numpy as np
import sys

class Trainer(object):
    def __init__(self, model, train_loader, val_loader,start_epoch,best_score, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.start_epoch = start_epoch
        self.epoch = args.epoch
        self.best_score = best_score
        self.args = args
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
        self.criterion = torch.nn.BCELoss()
        self.criterion = self.criterion.cuda()
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.criterion = self.criterion.cuda()

    def save_checkpoint(self,checkpoint, is_best,epoch):
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        # filename = 'Epoch-{}.pth'.format(self.epoch)
        filename = 'last_checkpoint.pth'
        res_path = os.path.join(self.args.model_dir, filename)

        torch.save(checkpoint, res_path)
        if is_best:
            filename_best = 'checkpoint_best.pth'
            res_path_best = os.path.join(self.args.model_dir, filename_best)
            shutil.copyfile(res_path, res_path_best)
            print('Save best checkpoint to {}'.format(res_path_best))
        pass


    def train(self):
        best_Of1 = 0
        best_Cf1 = 0
        st = time.time()
        loss_list = []
        undate_flag = 0
        for epoch in range(self.start_epoch, self.epoch):
            temp_loss_list = []
            temp_loss_list.append(epoch)
            loss_train = 0
            loss_train_list = []
            self.model.train()
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed(self.args.seed)
            cudnn.deterministic = True
            np.random.seed(self.args.seed)
            if self.args.data_name=='RCV1' and undate_flag>5:
                sys.exit()

            if self.args.data_name=='AAPD' and undate_flag>10:
                sys.exit()
            print("epoch: ", epoch, "stat train")


            for batch_idx, train in enumerate(tqdm(self.train_loader)):
                x_train, y_train,seq_lengh_train = train
                x_train = x_train.cuda()
                y_train = y_train.cuda()
                pred = self.model(x_train.long(), seq_lengh_train.long())
                loss = self.criterion(torch.sigmoid(pred), y_train.float())


                self.optimizer.zero_grad()
                # print(type(loss))
                loss.backward()
                self.optimizer.step()
                loss_train = loss_train + loss.item()
                loss_train_list.append(loss.item())

            print("epoch: ", epoch, "eval start")
            del train
            loss_train = loss_train/batch_idx+1

            with torch.no_grad():
                self.model.eval()
                random.seed(self.args.seed)
                torch.manual_seed(self.args.seed)
                torch.cuda.manual_seed(self.args.seed)
                cudnn.deterministic = True
                np.random.seed(self.args.seed)
                pred_tag = []
                gold_tag = []
                loss_valid = 0
                loss_valid_list = []
                with torch.no_grad():
                    for batch_idx, valid in enumerate(tqdm(self.val_loader)):
                        x_valid, y_valid, seq_lengh_valid = valid
                        x_valid = x_valid.cuda()
                        y_valid = y_valid.cuda()
                        pred = self.model(x_valid.long(), seq_lengh_valid.long())
                        loss = self.criterion(torch.sigmoid(pred), y_valid.float()).item()

                        loss_valid += loss
                        loss_valid_list.append(loss)
                        pred_tag.extend(list(torch.sigmoid(pred).cpu().data.numpy()))
                        gold_tag.extend(list(y_valid.cpu().data.numpy()))
                gold_tag_array = np.array(gold_tag)
                pred_tag_array = np.array(pred_tag)
                loss_valid = loss_valid/batch_idx+1
                pre_k = precision_k(gold_tag_array, pred_tag_array, 5)
                OP, OR, OF1,hamming_loss = evaluation_prf(pred_tag_array,gold_tag_array)
                print("model",self.args.model_dir,"epoch", epoch+1, "time: ", (time.time() - st) / 60, " train loss:", loss_train, " valid loss:",
                      loss_valid, " P@1", pre_k[0], " P@3", pre_k[2], " P@5", pre_k[4],
                      " OP:",OP," OR:", OR, " OF1:",OF1, " best_Of1:",best_Of1,  " hamming_loss:",hamming_loss, "undate_flag",undate_flag)

                temp_loss_list.append(loss_valid)
                temp_loss_list.append(loss_train)
                loss_list.append(temp_loss_list)

            is_best = False

            if best_Of1<OF1 :
                best_Of1 = OF1
                is_best = True
                undate_flag = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_name': "test",
                'state_dict': self.model.module.state_dict() if torch.cuda.is_available() else self.model.state_dict(),
                # 'state_dict': model.state_dict(),
                'best_score': pre_k
            }

            self.save_checkpoint(checkpoint, is_best,epoch+1)
            undate_flag = undate_flag+1
            f = open(os.path.join(self.args.model_dir,"loss.json"), 'w', encoding='utf-8')
            json.dump(loss_list, f)
            f.close()


    def evaluate(self):
        with torch.no_grad():

            self.model.eval()
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed(self.args.seed)
            cudnn.deterministic = True
            np.random.seed(self.args.seed)
            # with torch.no_grad():
            pred_tag = []
            gold_tag = []
            for batch_idx, valid in enumerate(tqdm(self.val_loader)):
                x_valid, y_valid, seq_lengh_valid = valid
                x_valid = x_valid.cuda()
                y_valid = y_valid.cuda()
                pred = self.model(x_valid.long(), seq_lengh_valid.long())
                pred_tag.extend(list(torch.sigmoid(pred).cpu().data.numpy()))
                gold_tag.extend(list(y_valid.cpu().data.numpy()))
        gold_tag_array = np.array(gold_tag)

        pred_tag_array = np.array(pred_tag)
        pre_k = precision_k(gold_tag_array, pred_tag_array, 5)
        OP, OR, OF1,hamming_loss = evaluation_prf(pred_tag_array, gold_tag_array)

        print(" P@1", pre_k[0], " P@3", pre_k[2], " P@5", pre_k[4],
              " OP:",OP," OR:", OR, " OF1:",OF1," hamming_loss:",hamming_loss)

















