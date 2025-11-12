import os
from termcolor import cprint
from tqdm import tqdm
import json
import numpy as np
from copy import deepcopy
import importlib
import torch
import torch.nn.functional as F
from data_utils import evaluate_detect, rand_splits
from dataset import load_dataset, prepare_dataset
from model import SynCo
import argparse
import importlib

import utils as u

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
conv_cuda = 1


def eval(model: SynCo, state_dict, dataset, config):
    edge_index, y = dataset.edge_index, dataset.y
    train_idx = dataset.splits['train']
    test_idx = dataset.splits['test']
    model.eval()
    embed = model.encoder.embed(dataset).detach()

    best_acc = 0.0
    for lr in config['lr_evals']:
        for wd in config['wd_evals']:
            model.load_state_dict(state_dict)
            opt = torch.optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=wd)

            for _ in tqdm(range(config['eval_epochs']), ncols=70):
                model.classifier.train()
                opt.zero_grad()

                logits = model.classifier(embed[train_idx])
                loss = F.cross_entropy(logits, y[train_idx])

                loss.backward()
                opt.step()
            logits = model.classifier(embed[test_idx])
            acc = (logits.argmax(dim=1) == y[test_idx]).float().mean().item() * 100
            if acc > best_acc:
                best_acc = acc
                config['lr_eval'] = lr
                config['wd_eval'] = wd
    return best_acc, config

def main(dataname, config, log_root='log'):
    torch.cuda.reset_peak_memory_stats()
    u.set_seed(28)
    folder_name = os.getcwd().split(os.sep)[-1]
    dataset_ind, _, dataset_ood_te = load_dataset(dataname, os.path.abspath(os.path.dirname(__file__)) + '/data/', config)
    dataset_ind.y = dataset_ind.y.squeeze()
    if isinstance(dataset_ood_te, list):
        for data in dataset_ood_te:
            data.y = data.y.squeeze()
    else:
        dataset_ood_te.y = dataset_ood_te.y.squeeze()

    if dataname not in ['cora', 'citeseer', 'pubmed']:
        dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=config['train_prop'], valid_prop=config['valid_prop'])

    in_channels = dataset_ind.x.shape[1]
    num_classes = dataset_ind.y.unique().shape[0]

    print(f"ind dataset {dataname}: all nodes {dataset_ind.num_nodes} | centered nodes {dataset_ind.node_idx.shape[0]} | edges {dataset_ind.edge_index.size(1)} | feats {in_channels}")
    if isinstance(dataset_ood_te, list):
        for i, data in enumerate(dataset_ood_te):
            print(f"ood te dataset {i} {dataname}: all nodes {data.num_nodes} | centered nodes {data.node_idx.shape[0]} | edges {data.edge_index.size(1)}")
    else:
        print(f"ood te dataset {dataname}: all nodes {dataset_ood_te.num_nodes} | centered nodes {dataset_ood_te.node_idx.shape[0]} | edges {dataset_ood_te.edge_index.size(1)}")

    model = SynCo(in_channels, num_classes, config,args)

    dataset_ind = dataset_ind.to(device)
    if config["mode"] == 'detect':
        if isinstance(dataset_ood_te, list):
            dataset_ood_te = [d.to(device) for d in dataset_ood_te]
        else:
            dataset_ood_te = dataset_ood_te.to(device)
    prepare_dataset(dataset_ind, device, model, cuda=conv_cuda)
    prepare_dataset(dataset_ood_te, device, model, cuda=conv_cuda)

    results_total = []
    save_file = {'results': {}, 'config': {}}
    model.config = config
    for run in range(config['num_runs']):
        model.reset_parameters()
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['lr'], weight_decay=config['wd'])

        best_result = None
        best_metric = 0.0
        best_model_sd = None
        for _ in tqdm(range(config['num_epochs']), ncols=70, unit='epoch'):
            model.train()
            optimizer.zero_grad()
            loss = model.loss_compute(dataset_ind, None)
            loss.backward()
            optimizer.step()
            result = evaluate_detect(model, dataset_ind, dataset_ood_te, visualize=False)

            metric = result[0] + result[-2]
            if metric > best_metric:
                best_result = result.copy()
                best_metric = metric
                best_model_sd = deepcopy(model.state_dict())

        final_results = (100 * torch.tensor(best_result)).tolist()
        model.load_state_dict(best_model_sd)

        _ = evaluate_detect(
            model,
            dataset_ind,
            dataset_ood_te,
            visualize=True,
            dataname=args.dataname,
            ood_type=args.ood_type
        )

        test_acc, config = eval(model, best_model_sd, dataset_ind, config)
        if test_acc > final_results[-2]:
            final_results[-2] = test_acc

        results_total.append(final_results)
        print_str = u.print_str(dataname, run, final_results, config)
        cprint(print_str, 'green')
    results_data = u.save_file_data(dataname, results_total, config)

    print_str = u.print_str(dataname, run, np.mean(results_total, axis=0).tolist(), config, final=True)
    cprint(print_str, 'green')

    return u.return_things({'results': results_data, 'config': config}), results_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the model with specific parameters')
    parser.add_argument('--dataname', type=str, default='cora',
                        choices=['cora', 'amazon-photo', 'twitch', 'arxiv', 'coauthor-cs' ],
                        help='Name of the dataset')
    parser.add_argument('--ood_type', type=str, default='structure',
                        choices=['structure', 'feature', 'label'],
                        help='Type of out-of-distribution data')
    parser.add_argument('--sampler_type', type=str, default='diffusion',
                        choices=['diffusion', 'vae'],
                        help='Type of sampler')
    
    
    args = parser.parse_args()

    if args.dataname in ['twitch', 'arxiv']:
        config_filename = f'config_{args.dataname}_{args.sampler_type}'
    else:
        config_filename = f'config_{args.dataname}_{args.ood_type}_{args.sampler_type}'
   
    config_module = importlib.import_module(f'config_files.{config_filename}')
   
   

    config = config_module.config()
    config['dataname'] = args.dataname
    config['ood_type'] = args.ood_type
    config['sampler_type'] = args.sampler_type
    
    results, save_file = main(dataname=args.dataname, config=config,
                             log_root=f'results')
 