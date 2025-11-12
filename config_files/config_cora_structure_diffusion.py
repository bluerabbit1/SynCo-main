import os
SEED = 28
DATA_DIR = os.path.abspath(os.path.dirname(__file__)) + '/data/'

# twitch
test_sg = ['ES', 'FR', 'RU']
# arxiv
test_year = [2018, 2019, 2020]
title = {
    'detect': ['auroc', 'aupr', 'fpr', 'test_acc'], 
}

def default_config():
    config = {
        'mode': 'detect', 
        'temperature': 0.1,  
        'sampler_type': 'diffusion',
        'diffusion_steps': 40,
        'num_runs': 1, 
        'num_epochs': 200, 
        'lam_cl': 2.0,
        'lam_cls': 0.5,
        'lam_sampler': 0.05,
        'lam_gen': 0.5,
        'train_prop': .1, 
        'valid_prop': .1, 
        'tau': 1,
        'cl_hdim': 64,
        'hidden_channels': 32,
        'lam_r': 1.,
        'coef_reg': 0.5,
        'lr': 0.01,
        'wd': 0.0005,
        'dropout': 0.3,
        'eval_epochs': 100, 
        'lr_evals': [0.05, 0.01, 0.001, 0.0001], 
        'wd_evals': [0.0, 1e-3, 5e-4], 
        'lr_eval': 0.001,
        'wd_eval': 0.001
    }

    return config


def config(ds='cora'):
    config = default_config()

    return config