import os
import csv
import math
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def read_smiles(smiles):
    
    mol_list = [Chem.MolFromSmiles(s) for s in smiles]
    #mol_list = [Chem.AddHs(mol) for mol in mol_list]
    
    return mol_list

class MolTestDataset(Dataset):
    def __init__(self, smiles_list):
        super(Dataset, self).__init__()
        self.mol_list = read_smiles(smiles_list)

    def __getitem__(self, index):
        mol = self.mol_list[index]
        #mol = Chem.AddHs(mol)
        
        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            
            
            if str(bond.GetBondDir()).split('.')[-1] == 'EITHERDOUBLE':
                edge_feat.append([BOND_LIST.index(bond.GetBondType()), BONDDIR_LIST.index(Chem.rdchem.BondDir.NONE)])
                edge_feat.append([BOND_LIST.index(bond.GetBondType()), BONDDIR_LIST.index(Chem.rdchem.BondDir.NONE)])
            else:
                edge_feat.append([
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir()),
                ])
                edge_feat.append([
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir())
                ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def __len__(self):
        return len(self.mol_list)


class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size, num_workers, smiles_list,
    ):
        super(object, self).__init__()
        self.smiles_list = smiles_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def get_data_loaders(self):
        dataset = MolTestDataset(smiles_list=self.smiles_list)
        data_loader = self.get_data_loader(dataset)
        return data_loader, dataset.mol_list

    def get_data_loader(self, dataset):

        data_loader = DataLoader(
            dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, drop_last=False
        )
        return data_loader
