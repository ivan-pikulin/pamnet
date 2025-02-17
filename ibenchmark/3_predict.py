import os
import os.path as osp
import random

import numpy as np
import torch
from pymatgen.io.ase import AseAtomsAdaptor
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from tqdm import tqdm

from ibenchmark.preprocess import preprocess_data
from models import Config, PAMNet

RDLogger.DisableLog("rdApp.*")
adaptor = AseAtomsAdaptor()

####################### configuration ###############################

dataset = 'amide_class'
sdf_to_predict_filename = f"data/ibenchmark/processed/{dataset}_test_lbl.sdf"
n_layer = 6
dim = 128
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

def predict(model, data: Data, device: torch.device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data)
    return output.cpu().item()

####################### setup ##################################

set_seed(seed)

##################### process dataset ######################

molecules = Chem.SDMolSupplier(
    sdf_to_predict_filename, removeHs=False, sanitize=False)

processed_data = preprocess_data(
    tqdm(molecules, desc="Preprocess molecules"))

##################### train model ######################

config = Config(dataset="QM9", dim=dim, n_layer=n_layer,
                cutoff_l=cutoff_l, cutoff_g=cutoff_g)

model = PAMNet(config).to(device)
model.load_state_dict(torch.load(osp.join(weights_folder, "E_0.25pr_0_seed.h5")))
model.eval()

predictions = [
    predict(model, data, device)
    for data in tqdm(processed_data, desc="Make predictions")
]

print(predictions)
