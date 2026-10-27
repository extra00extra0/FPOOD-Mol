import numpy as np
import rdkit
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from .fingerprint import EstateFP, FragmentFP, MACCSFP, MorganFP, PubChemFP, RdkitFP, RGroupFP


FP_CLASS_DICT = {
    'MorganFP': MorganFP,
    'RdkitFP': RdkitFP,
    'EstateFP': EstateFP,
    'MACCSFP': MACCSFP,
    'PubChemFP': PubChemFP,
    'RGroupFP': RGroupFP,
    'FragmentFP': FragmentFP,
}

FP_KWARGS_DICT = {
    'MorganFP': {'nBits': 1024, 'radius': 2},
    'RdkitFP': {'nBits': 1024, 'minPath': 1, 'maxPath': 5},
    'EstateFP': {},
    'MACCSFP': {},
    'PubChemFP': {},
    'RGroupFP': {'nBits': 2048},
    'FragmentFP': {},
}


def get_atom_features(atom):
    # The usage of features is along with the Attentive FP.
    feature = np.zeros(39)

    # Symbol
    symbol = atom.GetSymbol()
    symbol_list = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']
    if symbol in symbol_list:
        loc = symbol_list.index(symbol)
        feature[loc] = 1
    else:
        feature[15] = 1

    # Degree
    degree = atom.GetDegree()
    if degree > 5:
        print("atom degree larger than 5. Please check before featurizing.")
        raise RuntimeError

    feature[16 + degree] = 1

    # Formal Charge
    charge = atom.GetFormalCharge()
    feature[22] = charge

    # radical electrons
    radelc = atom.GetNumRadicalElectrons()
    feature[23] = radelc

    # Hybridization
    hyb = atom.GetHybridization()
    hybridization_list = [rdkit.Chem.rdchem.HybridizationType.SP,
                          rdkit.Chem.rdchem.HybridizationType.SP2,
                          rdkit.Chem.rdchem.HybridizationType.SP3,
                          rdkit.Chem.rdchem.HybridizationType.SP3D,
                          rdkit.Chem.rdchem.HybridizationType.SP3D2]
    if hyb in hybridization_list:
        loc = hybridization_list.index(hyb)
        feature[loc + 24] = 1
    else:
        feature[29] = 1

    # aromaticity
    if atom.GetIsAromatic():
        feature[30] = 1

    # hydrogens
    hs = atom.GetNumImplicitHs()
    feature[31 + hs] = 1

    # chirality, chirality type
    if atom.HasProp('_ChiralityPossible'):
        # TODO what kind of error
        feature[36] = 1

        try:
            chi = atom.GetProp('_CIPCode')
            chi_list = ['R', 'S']
            loc = chi_list.index(chi)
            feature[37 + loc] = 1
        except KeyError:
            feature[37] = 0
            feature[38] = 0

    return feature


def get_bond_features(bond):
    feature = np.zeros(10)

    # bond type
    type = bond.GetBondType()
    bond_type_list = [rdkit.Chem.rdchem.BondType.SINGLE,
                      rdkit.Chem.rdchem.BondType.DOUBLE,
                      rdkit.Chem.rdchem.BondType.TRIPLE,
                      rdkit.Chem.rdchem.BondType.AROMATIC]
    if type in bond_type_list:
        loc = bond_type_list.index(type)
        feature[0 + loc] = 1
    else:
        print("Wrong type of bond. Please check before feturization.")
        raise RuntimeError

    # conjugation
    conj = bond.GetIsConjugated()
    feature[4] = conj

    # ring
    ring = bond.IsInRing()
    feature[5] = ring

    # stereo
    stereo = bond.GetStereo()
    stereo_list = [rdkit.Chem.rdchem.BondStereo.STEREONONE,
                   rdkit.Chem.rdchem.BondStereo.STEREOANY,
                   rdkit.Chem.rdchem.BondStereo.STEREOZ,
                   rdkit.Chem.rdchem.BondStereo.STEREOE]
    if stereo in stereo_list:
        loc = stereo_list.index(stereo)
        feature[6 + loc] = 1
    else:
        print("Wrong stereo type of bond. Please check before featurization.")
        raise RuntimeError

    return feature


def smile2graph4drugood(smile):
    mol = Chem.MolFromSmiles(smile)
    # if (mol is None):
    #     return None
    src = []
    dst = []
    atom_feature = []
    bond_feature = []

    for atom in mol.GetAtoms():
        one_atom_feature = get_atom_features(atom)
        atom_feature.append(one_atom_feature)
    atom_feature = np.array(atom_feature)
    atom_feature = torch.tensor(atom_feature).float()

    if len(mol.GetBonds()) > 0:  # mol has bonds
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            one_bond_feature = get_bond_features(bond)
            src.append(i)
            dst.append(j)
            bond_feature.append(one_bond_feature)
            src.append(j)
            dst.append(i)
            bond_feature.append(one_bond_feature)

        src = torch.tensor(src).long()
        dst = torch.tensor(dst).long()
        bond_feature = np.array(bond_feature)
        bond_feature = torch.tensor(bond_feature).float()
        edge_index = torch.vstack([src, dst])
        # graph_cur_smile = dgl.graph((src, dst), num_nodes=len(mol.GetAtoms()))
        # graph_cur_smile.ndata['x'] = atom_feature
        # graph_cur_smile.edata['x'] = bond_feature
    else:
        edge_index = torch.empty((2, 0)).long()
        bond_feature = torch.empty((0, 10)).float()

    return atom_feature, edge_index, bond_feature


def get_atom_features_fppool(atom, symbols, hybridizations):
    symbol = [0.0] * len(symbols)
    if atom.GetSymbol() in symbols:
        symbol[symbols.index(atom.GetSymbol())] = 1.0
    else:
        symbol[symbols.index('other')] = 1.0

    degree = [0.0] * 6
    if atom.GetDegree() >= 6:
        degree[5] = 1.0
    else:
        degree[atom.GetDegree()] = 1.0

    formal_charge = atom.GetFormalCharge()
    radical_electrons = atom.GetNumRadicalElectrons()

    hybridization = [0.0] * len(hybridizations)
    atom_hybridization = atom.GetHybridization()
    if atom_hybridization in hybridizations:
        hybridization[hybridizations.index(atom_hybridization)] = 1.0
    else:
        hybridization[hybridizations.index('other')] = 1.0

    aromaticity = 1.0 if atom.GetIsAromatic() else 0.0
    hydrogens = [0.0] * 5
    hydrogens[atom.GetTotalNumHs()] = 1.0
    chirality = 1.0 if atom.HasProp('_ChiralityPossible') else 0.0
    chirality_type = [0.0] * 2
    if atom.HasProp('_CIPCode'):
        chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.0

    return symbol + degree + [formal_charge] + [radical_electrons] + hybridization + [aromaticity] + hydrogens + [chirality] + chirality_type


def get_bond_features_fppool(bond, stereos):
    bond_type = bond.GetBondType()
    single = 1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0
    double = 1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0
    triple = 1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0
    aromatic = 1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0
    conjugation = 1.0 if bond.GetIsConjugated() else 0.0
    ring = 1.0 if bond.IsInRing() else 0.0
    stereo = [0.0] * 4
    stereo[stereos.index(bond.GetStereo())] = 1.0
    return [single, double, triple, aromatic, conjugation, ring] + stereo


def get_morgan_atom_fp_arr(mol, radius=2, n_bits=1024):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    additional_output = rdFingerprintGenerator.AdditionalOutput()
    additional_output.AllocateBitInfoMap()
    generator.GetFingerprint(mol, additionalOutput=additional_output)
    bit_info = additional_output.GetBitInfoMap()
    atom_fp_arr = np.zeros((mol.GetNumAtoms(), n_bits), dtype=np.int64)

    for bit_idx, bit_paths in bit_info.items():
        for atom_idx, bit_radius in bit_paths:
            env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, bit_radius, atom_idx)
            atoms_to_use = {atom_idx}
            for bond_idx in env_bonds:
                bond = mol.GetBondWithIdx(bond_idx)
                atoms_to_use.add(bond.GetBeginAtomIdx())
                atoms_to_use.add(bond.GetEndAtomIdx())
            atom_fp_arr[list(atoms_to_use), bit_idx] = 1

    return atom_fp_arr


def get_atom_fp_arr(mol, fp_type='MorganFP'):
    if fp_type not in FP_CLASS_DICT:
        raise ValueError(f'Unsupported fingerprint type: {fp_type}')

    fp_builder = FP_CLASS_DICT[fp_type](**FP_KWARGS_DICT[fp_type])
    mol_fp = fp_builder(mol)
    atom_fp_arr = mol_fp.atom_fp_arr.astype(np.bool_)
    return atom_fp_arr, fp_builder.nBits


def smile2graph4drugood_fppool(smile, fp_type='MorganFP'):
    mol = Chem.MolFromSmiles(smile)

    symbols = [
        'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br',
        'Te', 'I', 'At', 'other'
    ]
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other',
    ]
    stereos = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]

    atom_features = [
        torch.tensor(get_atom_features_fppool(atom, symbols, hybridizations), dtype=torch.float)
        for atom in mol.GetAtoms()
    ]
    x = torch.stack(atom_features, dim=0)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
        edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

        edge_attr = torch.tensor(get_bond_features_fppool(bond, stereos), dtype=torch.float)
        edge_attrs += [edge_attr, edge_attr]

    if len(edge_attrs) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 10), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attrs, dim=0)

    atom_fp_arr, fp_bits = get_atom_fp_arr(mol, fp_type=fp_type)
    fp = torch.tensor(atom_fp_arr, dtype=torch.bool)
    fp_length = torch.tensor([[fp_bits]], dtype=torch.long)

    return x, edge_index, edge_attr, fp, fp_length


def featurize_atoms(mol):
    feats = []
    for atom in mol.GetAtoms():
        feats.append(atom.GetAtomicNum())
    return {'atomic': torch.tensor(feats).reshape(-1).to(torch.int64)}


def featurize_bonds(mol):
    feats = []
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    for bond in mol.GetBonds():
        btype = bond_types.index(bond.GetBondType())
        # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
        feats.extend([btype, btype])
    return {'type': torch.tensor(feats).reshape(-1).to(torch.int64)}
