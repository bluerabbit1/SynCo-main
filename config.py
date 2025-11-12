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

def config(ds='cora'):
    return config