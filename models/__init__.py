from .RDGCN import DynamicGraphConvolution
import torch
import os
def load_checkpoint(model,path):
    print("* Loading checkpoint '{}'".format(path))
    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch']
    best_score = checkpoint['best_score']
    model_dict = model.state_dict()
    for k, v in checkpoint['state_dict'].items():
        if k in model_dict and v.shape == model_dict[k].shape:
            model_dict[k] = v
            print(k)
        else:
            print ('\tMismatched layers: {}'.format(k))
    model.load_state_dict(model_dict)
    return model,start_epoch,best_score

def get_model(args):
    model = DynamicGraphConvolution(args)
    best_score = [0, 0, 0, 0]
    start_epoch = 0
    if args.continue_train or args.evaluate:
        if args.evaluate:
            model, start_epoch, best_score = load_checkpoint(model, args.evaluate_chechpoint_path)
        else:
            model, start_epoch, best_score = load_checkpoint(model, args.chechpoint_path)

    return model, start_epoch, best_score

