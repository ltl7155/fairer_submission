import os,sys
import argparse
import pandas as pd
import numpy as np
from numpy.random import beta
from tqdm import tqdm
# import pickle5 as pickle
import pickle

from pprint import pprint

import torch
import torch.nn as nn
from torch import optim

import random
import torch.backends.cudnn as cudnn



cudnn.deterministic = True

from model_prune_tv import *

from utils_alexnet import *


from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

from pruning_engine import pytorch_pruning, PruningConfigReader, prepare_pruning_list
from pruning_utils import save_checkpoint, adjust_learning_rate, AverageMeter, accuracy, load_model_pytorch, dynamic_network_change_local, get_conv_sizes, connect_gates_with_parameters_for_flops
from pruning_utils import group_lasso_decay,get_conv_sizes

from utils_train_alexnet import fit_model_eo, fit_model_dp


device=  torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CelebA Experiment')
    parser.add_argument('--fair_method',"-fair_method", default='mixup', type=str, help='mixup/GapReg/erm')
    parser.add_argument('--mode', default='eo', type=str, help='dp/eo')
    parser.add_argument('--epochs', default=20, type=int)

    parser.add_argument('--lam', "-lam",default=0.5, type=float, help='Lambda for regularization')
    parser.add_argument('--lam2',"-lam2", default=0.5, type=float, help='Lambda for regularization')
    parser.add_argument('--bs', default=32, type=int, help='Lambda for regularization')
    parser.add_argument('--sl', default=0, type=int, help='start layer')

    parser.add_argument('--target_id', default=33, type=int, help='2:attractive/31:smile/33:wavy hair')
    parser.add_argument('--data_celeba_dir', default="",
                         type=str, help='123')
    parser.add_argument('--save_root', default="",
                         type=str, help='123')

    parser.add_argument('--name', default="celeba", type=str, )
#     parser.add_argument('--warm',"-warm", default="scratch", type=str, )
    parser.add_argument('--frequency',"-freq", default=10, type=int, )
    parser.add_argument('--prune_per_iteration',"-ppi", default=100, type=int, )
    parser.add_argument('--maximum_pruning_iterations',"-mpi", default=120, type=int, )
    parser.add_argument('--prune_neurons_max',"-pnm", default=400, type=int, )
    parser.add_argument('--pruning_threshold',"-pt", default=10, type=float, )
    #pruning 
    # ============================PRUNING added
    parser.add_argument('--log_interval', default=5, type=int, help='Lambda for regularization')
    parser.add_argument('--group_wd_coeff', type=float, default=0.0,
                        help='group weight decay')
    parser.add_argument('--pruning_config', default="./config/prune_config.json", type=str,
                        help='path to pruning configuration file, will overwrite all pruning parameters in arguments')
    parser.add_argument('--pruning_mask_from', default='', type=str,
                        help='path to mask file precomputed')
    parser.add_argument('--dataset', default='adult', type=str,
                        help='dataset name')
    parser.add_argument('--compute_flops', action='store_false',
                        help='if True, will run dummy inference of batch 1 before training to get conv sizes')
    parser.add_argument('--pruning', action='store_true',
                        help='enable or not pruning, def False')
    # ============================END pruning added
    parser.add_argument('--model_name', default="alexnet", type=str, )

    parser.add_argument('--exp', default=0, type=int, )
    parser.add_argument('--print_freq', default=40, type=int, )
    parser.add_argument('--tensorname', default="", type=str, )


    args =parser.parse_args()


    group_wd_optimizer=None 
    pruning_engine=None

    model = AlexNet().cuda()
#     output_sizes = get_conv_sizes(args, model)
    # log_save_folder = "%s"%args.name
    # if not os.path.exists(log_save_folder):
    #     os.makedirs(log_save_folder)
    #
    # folder_to_write = "%s"%log_save_folder+"/"
    # log_folder = folder_to_write
    
    i = args.exp
    mode = args.mode
    model_name = args.model_name
    
    random.seed(i)
    torch.manual_seed(i)
    np.random.seed(i)
    
    if args.pruning:
        pruning_settings = dict()
        if not (args.pruning_config is None):
            pruning_settings_reader = PruningConfigReader()
            pruning_settings_reader.read_config(args.pruning_config)
            pruning_settings = pruning_settings_reader.get_parameters()
        for k,v in vars(args).items():
            pruning_settings[k]=v 
        # print (pruning_settings,"pruning_settings")
        for k,v in pruning_settings.items():
            setattr(args,k,v)

    if args.pruning:
        if args.fair_method == "CAIGA":
            log_folder = f"logs_alexnet/{args.fair_method}/Exp{i}_{args.mode}_{args.name}_{args.fair_method}-lam_{args.lam}-lam2_{args.lam2}-pp1_{pruning_settings['prune_per_iteration']}" \
                         f"-mpi_{pruning_settings['maximum_pruning_iterations']}-pnm_{pruning_settings['prune_neurons_max']}-pt_{pruning_settings['pruning_threshold']}-freq_{args.frequency}"
            pth_folder = f"{args.save_root}/save_alexnet/Exp{i}_{args.mode}_{args.name}_{args.fair_method}-lam_{args.lam}-lam2_{args.lam2}-pp1_{pruning_settings['prune_per_iteration']}" \
                         f"-mpi_{pruning_settings['maximum_pruning_iterations']}-pnm_{pruning_settings['prune_neurons_max']}-pt_{pruning_settings['pruning_threshold']}-freq_{args.frequency}"
        else:
            log_folder = f"logs_alexnet/{args.fair_method}/Exp{i}_{args.mode}_{args.name}_{args.fair_method}-lam_{args.lam}-pp1_{pruning_settings['prune_per_iteration']}" \
                         f"-mpi_{pruning_settings['maximum_pruning_iterations']}-pnm_{pruning_settings['prune_neurons_max']}-pt_{pruning_settings['pruning_threshold']}-freq_{args.frequency}"
            pth_folder = f"{args.save_root}/save_alexnet/Exp{i}_{args.mode}_{args.name}_{args.fair_method}-lam_{args.lam}-pp1_{pruning_settings['prune_per_iteration']}" \
                         f"-mpi_{pruning_settings['maximum_pruning_iterations']}-pnm_{pruning_settings['prune_neurons_max']}-pt_{pruning_settings['pruning_threshold']}-freq_{args.frequency}"
    else:
        if "CAIGA" in args.fair_method:
            log_folder = f"logs_alexnet_id{args.target_id}/{args.fair_method}/{args.mode}/Exp{i}_{args.mode}_{args.name}_{args.fair_method}-lam_{args.lam}-lam2_{args.lam2}_sl_{args.sl}"
            pth_folder = f"{args.save_root}/save_alexnet/Exp{i}_{args.mode}_{args.name}_{args.fair_method}-lam_{args.lam}-lam2_{args.lam2}_sl_{args.sl}"
        else:
            log_folder = f"logs_alexnet_id{args.target_id}/{args.fair_method}/{args.mode}/Exp{i}_{args.mode}_{args.name}_{args.fair_method}-lam_{args.lam}_sl_{args.sl}"
            pth_folder = f"{args.save_root}/save_alexnet/Exp{i}_{args.mode}_{args.name}_{args.fair_method}-lam_{args.lam}_sl_{args.sl}"

    os.makedirs(pth_folder, exist_ok=True)
    writer = SummaryWriter(log_folder)

    parameters_for_update = []
    parameters_for_update_named = []
    for name, m in model.named_parameters():
        if "gate" not in name:
            parameters_for_update.append(m)
            parameters_for_update_named.append((name, m))
        else:
            if args.pruning:
                print("skipping parameter", name, "shape:", m.shape)
    # ============================PRUNING end

    if args.pruning:
        has_attribute = lambda x: any([x in a for a in sys.argv])

        if has_attribute('pruning-momentum'):
            pruning_settings['pruning_momentum'] = vars(args)['pruning_momentum']
        if has_attribute('pruning-method'):
            pruning_settings['pruning_method'] = vars(args)['pruning_method']

        pruning_parameters_list = prepare_pruning_list(pruning_settings, model, model_name=model_name,
                                                       pruning_mask_from=args.pruning_mask_from, name=args.name)
        print("Total pruning layers:", len(pruning_parameters_list))

        # print ("pruning_parameters_list" , pruning_parameters_list ,"pruning_settings", pruning_settings )
        pruning_engine = pytorch_pruning(pruning_parameters_list, pruning_settings=pruning_settings, log_folder=log_folder, writer=writer)

        # pruning_engine.connect_tensorboard(train_writer)
        # pruning_engine.dataset = args.dataset
        pruning_engine.model = model_name #args.model
        pruning_engine.pruning_mask_from = args.pruning_mask_from
        pruning_engine.load_mask()
        gates_to_params = connect_gates_with_parameters_for_flops(model_name , parameters_for_update_named)
        pruning_engine.gates_to_params = gates_to_params
        group_wd_optimizer = group_lasso_decay(parameters_for_update, group_lasso_weight=args.group_wd_coeff, named_parameters=parameters_for_update_named, output_sizes=output_sizes)

    # ============================PRUNING end


    # print (pruning_settings,"pruning_settings")


    # log_folder= f"logs/test_{args.name}_{args.fair_method}_lam_{args.lam}_lam2_{args.lam2}_targetid_{args.target_id}_epochs_{args.epochs}"
    # pth_folder= f"save/test_{args.name}_{args.fair_method}_lam_{args.lam}_lam2_{args.lam2}_targetid_{args.target_id}_epochs_{args.epochs}"
    # os.makedirs(pth_folder, exist_ok=True)
    # writer = SummaryWriter(log_folder)

    # Load Celeb dataset
    target_id = args.target_id

    data_celeba_dir= args.data_celeba_dir 
    with open(os.path.join(data_celeba_dir,'celeba/data_frame.pickle'), 'rb') as handle:
        df = pickle.load(handle)

    train_df = df['train']
    valid_df = df['val']
    test_df = df['test']
    # data_loader

    criterion = nn.BCELoss()
    
    total_size_params = sum([np.prod(par.shape) for par in parameters_for_update])
    print("Total number of parameters, w/o usage of bn consts: ", total_size_params)


    optimizer = optim.Adam(parameters_for_update, lr=1e-4)

    final_list=  []
    
    ap_val_epoch = []
    gap_logit_val_epoch = []
    gap_05_val_epoch = []
    ap_test_epoch = []
    gap_logit_test_epoch = []
    gap_05_test_epoch = []

    for j in range(0, args.epochs):

        pprint("Epoch: {}".format(j))
        ap_test_eo ,gap_test_eo, gap_real_eo,gap_test_dp ,gap_real_dp = 0,0,0,0,0
        gap_real05_eo ,gap_real05_dp = 0,0

        def pretest_call(namespace="Testing"):
            print(namespace+":")
            if mode == 'dp':
                print("Val:")
                ap_val, gap_logit_val, gap_05_val= evaluate_dp(model, 
                                                               valid_dataloader, valid_dataloader_0, valid_dataloader_1)
                print("Test:")
                ap_test, gap_logit_test, gap_05_test= evaluate_dp(model, 
                                                                  test_dataloader, test_dataloader_0, test_dataloader_1)
            if mode == 'eo':
                # ap_valal, gap_logit_val, gap_05_v = evaluate_eo(model, X_val, y_val, A_val)
                print("Val:")
                ap_val, gap_logit_val, gap_05_val = evaluate_eo(model,  
                                                                valid_dataloader, valid_dataloader_00, valid_dataloader_01, 
                                                                valid_dataloader_10, valid_dataloader_11)
                print("Test:")
                ap_test, gap_logit_test, gap_05_test = evaluate_eo(model, 
                                                                   test_dataloader, test_dataloader_00, test_dataloader_01, 
                                                                   test_dataloader_10, test_dataloader_11)
            writer.add_scalar('Val/ap_val', ap_val, j)
            writer.add_scalar('Val/gap_logit_val', gap_logit_val, j)
            writer.add_scalar('Val/gap_05_val', gap_05_val, j)
            writer.add_scalar('Test/ap_test', ap_test, j)
            writer.add_scalar('Test/gap_logit_test', gap_logit_test, j)
            writer.add_scalar('Test/gap_05_test', gap_05_test, j)
            
            if j > 2:
                ap_val_epoch.append(ap_val)
                ap_test_epoch.append(ap_test)
                gap_logit_val_epoch.append(gap_logit_val)
                gap_logit_test_epoch.append(gap_logit_test)
                gap_05_val_epoch.append(gap_05_val)
                gap_05_test_epoch.append(gap_05_test)
               
            metainfo = {
                         "ap_val":ap_val,
                         "gap_logit_val":gap_logit_val,
                         "gap_05_val":gap_05_val,
                         "ap_test":ap_test,
                         "gap_logit_test":gap_logit_test,
                         "gap_05_test":gap_05_test,
                         "epoch":j , 
                          **(vars(args))
                         }
            return metainfo  
        
        if args.mode == "dp":
            
            train_dataloader = get_loader(train_df, os.path.join(data_celeba_dir, 'celeba/split/train/'), target_id, args.bs)
            valid_dataloader = get_loader(valid_df, os.path.join(data_celeba_dir, 'celeba/split/val/'), target_id, args.bs)
            test_dataloader = get_loader(test_df, os.path.join(data_celeba_dir, 'celeba/split/test/'), target_id, args.bs)

            train_dataloader_0 = get_loader(train_df, os.path.join(data_celeba_dir, 'celeba/split/train/'), target_id, args.bs, gender = '0')
            train_dataloader_1 = get_loader(train_df, os.path.join(data_celeba_dir, 'celeba/split/train/'), target_id, args.bs, gender = '1')
            valid_dataloader_0 = get_loader(valid_df, os.path.join(data_celeba_dir, 'celeba/split/val/'), target_id, args.bs, gender = '0')
            valid_dataloader_1 = get_loader(valid_df, os.path.join(data_celeba_dir, 'celeba/split/val/'), target_id, args.bs, gender = '1')
            test_dataloader_0 = get_loader(test_df, os.path.join(data_celeba_dir, 'celeba/split/test/'), target_id, args.bs, gender = '0')
            test_dataloader_1 = get_loader(test_df, os.path.join(data_celeba_dir, 'celeba/split/test/'), target_id, args.bs, gender = '1')
            
            metainfo = fit_model_dp(j, model,  train_dataloader, train_dataloader_0, train_dataloader_1, args.fair_method, args.lam, lam2=args.lam2, group_wd_optimizer=group_wd_optimizer, 
                             pruning_engine=pruning_engine, args=args, pretest_call=pretest_call, 
                                    criterion=criterion, writer=writer, optimizer=optimizer)

        elif args.mode == "eo":
            
            train_dataloader = get_loader(train_df, os.path.join(data_celeba_dir, 'celeba/split/train/'), target_id, 64)
            valid_dataloader = get_loader(valid_df, os.path.join(data_celeba_dir, 'celeba/split/val/'), target_id, 64)
            test_dataloader = get_loader(test_df, os.path.join(data_celeba_dir, 'celeba/split/test/'), target_id, 64)

            test_dataloader_0 = get_loader(test_df, os.path.join(data_celeba_dir, 'celeba/split/test/'), target_id, args.bs, gender = '0')
            test_dataloader_1 = get_loader(test_df, os.path.join(data_celeba_dir, 'celeba/split/test/'), target_id, args.bs, gender = '1')

            train_dataloader_00 = get_loader(train_df, os.path.join(data_celeba_dir, 'celeba/split/train/'), target_id, args.bs, gender='0', target='0')
            train_dataloader_01 = get_loader(train_df, os.path.join(data_celeba_dir, 'celeba/split/train/'), target_id, args.bs, gender='0', target='1')
            train_dataloader_10 = get_loader(train_df, os.path.join(data_celeba_dir, 'celeba/split/train/'), target_id, args.bs, gender='1', target='0')
            train_dataloader_11 = get_loader(train_df, os.path.join(data_celeba_dir, 'celeba/split/train/'), target_id, args.bs, gender='1', target='1')

            valid_dataloader_00 = get_loader(valid_df, os.path.join(data_celeba_dir, 'celeba/split/val/'), target_id, args.bs, gender = '0', target='0')
            valid_dataloader_01 = get_loader(valid_df, os.path.join(data_celeba_dir, 'celeba/split/val/'), target_id, args.bs, gender = '0', target='1')
            valid_dataloader_10 = get_loader(valid_df, os.path.join(data_celeba_dir, 'celeba/split/val/'), target_id, args.bs, gender = '1', target='0')
            valid_dataloader_11 = get_loader(valid_df, os.path.join(data_celeba_dir, 'celeba/split/val/'), target_id, args.bs, gender = '1', target='1')

            test_dataloader_00 = get_loader(test_df, os.path.join(data_celeba_dir, 'celeba/split/test/'), target_id, args.bs, gender = '0', target='0')
            test_dataloader_01 = get_loader(test_df, os.path.join(data_celeba_dir, 'celeba/split/test/'), target_id, args.bs, gender = '0', target='1')
            test_dataloader_10 = get_loader(test_df, os.path.join(data_celeba_dir, 'celeba/split/test/'), target_id, args.bs, gender = '1', target='0')
            test_dataloader_11 = get_loader(test_df, os.path.join(data_celeba_dir, 'celeba/split/test/'), target_id, args.bs, gender = '1', target='1')
            
            metainfo = fit_model_eo(j, model,  train_dataloader, train_dataloader_00, train_dataloader_01, train_dataloader_10, train_dataloader_11, args.fair_method,
                       lam=args.lam, lam2= args.lam2 ,
                              group_wd_optimizer=group_wd_optimizer, 
                             pruning_engine=pruning_engine, args=args, 
                                    pretest_call=pretest_call, 
                                    criterion=criterion, writer=writer, optimizer=optimizer, sl=args.sl)

        # metainfo = pretest_call(namespace="EpochTest")
        assert metainfo is not None 
        
        if args.exp == 100:
            try:
                torch.save({**metainfo , 
                            "state_dict":model.state_dict(), 
                             }, os.path.join(pth_folder,"{}.pth".format(j) ))
            except :
                pass 
            try :
                torch.save({**metainfo , 
                            "optim":optimizer.state_dict(),
                             }, os.path.join(pth_folder,"{}_optim.pth".format(j) ))
            except :
                pass 
        
        final_list.append(metainfo)
        
    idx = gap_logit_val_epoch.index(min(gap_logit_val_epoch))
#     gap_logit.append(gap_logit_test_epoch[idx])
#     gap_05.append()
#     ap.append(ap_test_epoch[idx])
    
    print('--------AVG---------')
    print('Average Precision', ap_test_epoch[idx])
    print(mode + ' gap_logit', gap_logit_test_epoch[idx])
    print(mode + ' gap_05', gap_05_test_epoch[idx])
    
    import pandas as pd 
    df = pd.DataFrame( final_list )
    df.to_csv( os.path.join(pth_folder, f"Exp{i}_eval.csv" ), index=False)
    
    os.makedirs(f"results_alexnet/Normal/{args.fair_method}", exist_ok=True)
    fname = f"results_alexnet/Normal/{args.fair_method}/Exp{i}_{args.mode}_{args.name}_{args.fair_method}-lam-{args.lam}-lam2-{args.lam2}_sl_{args.sl}.txt"
    
    
    with open(fname, "w") as f:
        f.write(f'{ap_test_epoch[idx]}\n')
        f.write(f'{gap_logit_test_epoch[idx]}\n')
        f.write(f'{gap_05_test_epoch[idx]}\n')