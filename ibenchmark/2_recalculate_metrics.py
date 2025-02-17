import os
import os.path as osp
import random

import numpy as np
import torch
from pymatgen.io.ase import AseAtomsAdaptor
from rdkit import Chem, RDLogger
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ibenchmark.preprocess import preprocess_data
from models import Config, PAMNet

RDLogger.DisableLog("rdApp.*")
adaptor = AseAtomsAdaptor()

####################### configuration ###############################

dataset = 'amide_class'
train_filename = f"data/ibenchmark/processed/{dataset}_train_lbl.sdf"
test_filename = f"data/ibenchmark/processed/{dataset}_test_lbl.sdf"
n_layer = 6
dim = 128
batch_size = 32
cutoff_l = 5.0
cutoff_g = 5.0
seed = 0
device = 'cuda:0'

weights_folder = osp.join(".", "save", dataset)
if not osp.exists(weights_folder):
    os.makedirs(weights_folder)

######################## functions ###################################

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def test(model, loader, device):
    mae = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        mae += (output - data.y).abs().sum().item()
    return mae / len(loader.dataset)

####################### setup ##################################

set_seed(seed)

##################### process dataset ######################

train_molecules = Chem.SDMolSupplier(
    train_filename, removeHs=False, sanitize=False)
test_molecules = Chem.SDMolSupplier(
    test_filename, removeHs=False, sanitize=False)

train_val_data = preprocess_data(
    tqdm(train_molecules, desc="Processing training molecules"))
train_data, val_data = train_test_split(
    train_val_data, test_size=0.2, random_state=seed)
test_data = preprocess_data(
    tqdm(test_molecules, desc="Processing test molecules"))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

##################### train model ######################

config = Config(dataset="QM9", dim=dim, n_layer=n_layer,
                cutoff_l=cutoff_l, cutoff_g=cutoff_g)

model = PAMNet(config).to(device)
model.load_state_dict(torch.load(osp.join(weights_folder, "E_0.25pr_0_seed.h5")))
model.eval()

train_mae = test(model, train_loader, device)
val_mae = test(model, val_loader, device)
test_mae = test(model, test_loader, device)

print(f"Train MAE: {train_mae}")
print(f"Val MAE: {val_mae}")
print(f"Test MAE: {test_mae}")
