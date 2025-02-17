import argparse
import os
import os.path as osp
import random

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pymatgen.io.ase import AseAtomsAdaptor
from rdkit import Chem, RDLogger
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

from ibenchmark.preprocess import preprocess_data
from models import Config, PAMNet
from utils import EMA

RDLogger.DisableLog("rdApp.*")
adaptor = AseAtomsAdaptor()

####################### configuration ###############################

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--train_filename', type=str,
                    help='Training data filename')
parser.add_argument('--test_filename', type=str,
                    help='Testing data filename')
parser.add_argument('--epochs', type=int, default=900,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay for optimizer')
parser.add_argument('--n_layer', type=int, default=6,
                    help='Number of layers in the model')
parser.add_argument('--dim', type=int, default=128,
                    help='Dimension of the model')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('--cutoff_l', type=float,
                    default=5.0, help='Cutoff length')
parser.add_argument('--cutoff_g', type=float, default=5.0,
                    help='Cutoff graph')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--experiment_name', type=str,
                    help='Experiment name for mlflow')
parser.add_argument('--device', type=str, default='cpu',
                    help='Device to run the model on')

args = parser.parse_args()
args.dataset = 'QM9'
args.model = 'PAMNet'

mlflow.set_experiment(args.experiment_name)
mlflow.start_run()

######################## functions ###################################

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test(model, loader, ema, device):
    mae = 0
    ema.assign(model)
    for data in loader:
        data = data.to(device)
        output = model(data)
        mae += (output - data.y).abs().sum().item()
    ema.resume(model)
    return mae / len(loader.dataset)

####################### setup ##################################

set_seed(args.seed)

##################### process dataset ######################

train_molecules = Chem.SDMolSupplier(
    args.train_filename, removeHs=False, sanitize=False)
test_molecules = Chem.SDMolSupplier(
    args.test_filename, removeHs=False, sanitize=False)

train_val_data = preprocess_data(
    tqdm(train_molecules, desc="Processing training molecules"))
train_data, val_data = train_test_split(
    train_val_data, test_size=0.2, random_state=args.seed)
test_data = preprocess_data(
    tqdm(test_molecules, desc="Processing test molecules"))

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

##################### train model ######################

config = Config(dataset=args.dataset, dim=args.dim, n_layer=args.n_layer,
                cutoff_l=args.cutoff_l, cutoff_g=args.cutoff_g)

model = PAMNet(config).to(args.device)
mlflow.log_params({
    "dataset": args.dataset,
    "dim": args.dim,
    "n_layer": args.n_layer,
    "cutoff_l": args.cutoff_l,
    "cutoff_g": args.cutoff_g,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "seed": args.seed,
    "batch_size": args.batch_size,
})
print("Number of model parameters: ", count_parameters(model))
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
scheduler_warmup = GradualWarmupScheduler(
    optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)

ema = EMA(model, decay=0.999)

print("Start training!")

best_val_loss = None
for epoch in range(args.epochs):
    loss_all = 0
    step = 0
    model.train()
    for data in train_loader:
        data = data.to(args.device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.l1_loss(output, data.y)
        loss_all += loss.item() * data.num_graphs
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2)
        optimizer.step()

        curr_epoch = epoch + float(step) / (len(train_data) / args.batch_size)
        scheduler_warmup.step(curr_epoch)

        ema(model)
        step += 1
    loss = loss_all / len(train_loader.dataset)

    val_loss = test(model, val_loader, ema, args.device)

    save_folder = osp.join(".", "save", args.experiment_name)
    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    if best_val_loss is None or val_loss <= best_val_loss:
        test_loss = test(model, test_loader, ema, args.device)
        best_val_loss = val_loss
        torch.save(model.state_dict(), osp.join(
            save_folder, "E_0.25pr_0_seed.h5"))

    print('Epoch: {:03d}, Train MAE: {:.7f}, Val MAE: {:.7f}, '
          'Test MAE: {:.7f}'.format(epoch+1, loss, val_loss, test_loss))
    mlflow.log_metric("train_loss", loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("test_loss", test_loss, step=epoch)

print('Best Validation MAE:', best_val_loss)
print('Testing MAE:', test_loss)
