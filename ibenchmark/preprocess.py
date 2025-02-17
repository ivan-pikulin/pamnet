import tempfile

import torch
import torch.nn.functional as F
from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.data import Data
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
adaptor = AseAtomsAdaptor()

##################### need for rdkit ######################
elements = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'P', 'B', 'I', 'Si', 'Se', 'As', 'Te']

types = {element: i for i, element in enumerate(elements)}
symbols = {element: Chem.Atom(element).GetAtomicNum() for element in elements}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

##################### data processing ######################


def preprocess_data(molecules: list[Molecule]) -> list[Data]:
    data_list = []

    for mol in tqdm(molecules):
        num_atoms = mol.GetNumAtoms()

        ##### make a pos list with coordinate lists #####
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as temp_xyz:
            Chem.MolToXYZFile(mol, temp_xyz.name)
            with open(temp_xyz.name, 'r') as f:
                positions = torch.tensor([
                    list(map(float, line.split()[1:]))
                    for line in f.readlines()[2:num_atoms+2]
                ], dtype=torch.float)

        ##### make a tensor  #####
        type_idx = []
        atomic_number = []
        aromatic = []
        sp = []
        sp2 = []
        sp3 = []

        for atom in mol.GetAtoms():
            type_idx.append(types[atom.GetSymbol()])
            atomic_number.append(atom.GetAtomicNum())
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            sp.append(1 if hybridization == HybridizationType.SP else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

        z = torch.tensor(atomic_number, dtype=torch.long)

        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [bonds[bond.GetBondType()]]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type,
                              num_classes=len(bonds)).to(torch.float)

        perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
        edge_attr = edge_attr[perm]

        row, col = edge_index

        x = torch.tensor(type_idx).to(torch.float)

        y = float(mol.GetProp('activity'))
        y = torch.tensor(y, dtype=torch.float)
        y = y.unsqueeze(0)

        data = Data(x=x, pos=positions, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list
