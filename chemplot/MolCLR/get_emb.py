import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from .dataset.dataset import MolTestDatasetWrapper

class Get_Embeddings(object):
    def __init__(self, config, smiles_list):
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        self.config = config
        self.device = self._get_device()
        dataset = MolTestDatasetWrapper(config['batch_size'], config['dataset']['num_workers'], smiles_list)
        self.dataset = dataset

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def get(self):
        data_loader, mol_list = self.dataset.get_data_loaders()
        print('data loading is done')
        
        if self.config['model_type'] == 'gin':
            from .models.ginet_molclr import GINet
            model = GINet(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        elif self.config['model_type'] == 'gcn':
            from .models.gcn_molclr import GCN
            model = GCN(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        print('model init is done')
        print('get emb start')
        emb_list = self._test(model, data_loader)
        print('get emb done')
        return mol_list, emb_list

    def _load_pre_trained_weights(self, model):
        try:
            #checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            #state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            load_path = os.path.join(os.path.dirname(__file__), self.config['fine_tune_from'])
            state_dict = torch.load(load_path, map_location=self.device)
            model.load_state_dict(state_dict)
            #model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _test(self, model, test_loader):
        # test steps
        predictions = []
        with torch.no_grad():
            model.eval()

            for bn, data in enumerate(tqdm(test_loader)):
                data = data.to(self.device)
                __, pred = model(data)
                    
                predictions.append(list(pred.cpu().detach().numpy().squeeze()))

        return predictions
