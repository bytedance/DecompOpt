import itertools
import os
import pickle
from collections import defaultdict
import random

import lmdb
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm

from utils.data import PDBProtein
from utils.data import ProteinLigandData, torchify_dict, parse_sdf_file
from utils.prior import compute_golden_prior_from_data
from multiprocessing.pool import ThreadPool

def get_decomp_dataset(config, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = DecompPLPairDataset(root, mode=config.mode, 
                                      load_retrieved_feat=getattr(config.transform, 'add_retrieved_feat', False),
                                      include_dummy_atoms=config.include_dummy_atoms, 
                                      version=config.version, 
                                      top_k_sim=getattr(config.transform, 'top_k_sim', 11),
                                      supocket_radius=getattr(config.transform, 'supocket_radius', 10),
                                      **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config:
        split_by_name = torch.load(config.split)
        split = {
            k: [dataset.name2id[n[1][:-4]] for n in names if n[1][:-4] in dataset.name2id]
            for k, names in split_by_name.items()
        }
        for k, v in split.items():
            split[k] = list(itertools.chain(*v))
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset


def extract_info_from_meta(dataset, idx, meta_info, arm_list=None, sub_ligand_file_dict=None):
    print(idx)
    try:
        with open(meta_info['data']['meta_file'].lstrip('./'), 'rb') as f:
            meta = pickle.load(f)
        m = meta['data']
        num_arms, num_scaffold = m['num_arms'], m['num_scaffold']
        if dataset.mode == 'full':
            protein = PDBProtein(m['protein_file'])
            protein_dict = protein.to_dict_atom()
            ligand_dict = parse_sdf_file(m['ligand_file'], kekulize=dataset.kekulize)
            num_protein_atoms, num_ligand_atoms = len(protein.atoms), ligand_dict['rdmol'].GetNumAtoms()
            assert num_ligand_atoms == sum([len(x) for x in m['all_submol_atom_idx']])

            # extract pocket atom mask
            protein_atom_serial = [atom['atom_id'] for atom in protein.atoms]
            pocket_atom_masks = []
            assert len(m['all_pocket_atom_serial']) == num_arms
            for pocket_atom_serial in m['all_pocket_atom_serial']:
                pocket_atom_idx = [protein_atom_serial.index(i) for i in pocket_atom_serial]
                pocket_atom_mask = torch.zeros(num_protein_atoms, dtype=torch.bool)
                pocket_atom_mask[pocket_atom_idx] = 1
                pocket_atom_masks.append(pocket_atom_mask)
            pocket_atom_masks = torch.stack(pocket_atom_masks)

            # extract ligand atom mask
            ligand_atom_mask = torch.zeros(num_ligand_atoms, dtype=int)
            for arm_idx, atom_idx in enumerate(m['all_submol_atom_idx']):
                if arm_idx == len(m['all_submol_atom_idx']) - 1:
                    ligand_atom_mask[atom_idx] = -1
                else:
                    ligand_atom_mask[atom_idx] = arm_idx
            assert len(ligand_atom_mask.unique()) == num_arms + num_scaffold

            data = ProteinLigandData.from_protein_ligand_dicts(
                protein_dict=torchify_dict(protein_dict),
                ligand_dict=torchify_dict(ligand_dict),
            )

            sub_ligand_file = m['sub_ligand_file']
            arm_indices = sub_ligand_file_dict[sub_ligand_file]
            all_processed_sim_arms = []
            for arm_id in arm_indices:
                # use self as condition for training
                arm = arm_list[arm_id]
                du = Chem.MolFromSmiles('*')
                nodummy_frag = AllChem.ReplaceSubstructs(
                    arm, du, Chem.MolFromSmiles('[H]'), True)[0]
                nodummy_frag = Chem.RemoveHs(nodummy_frag)

                # get condition arm's subpocket: extract by radius
                selected_atom_serial, union_residues = protein.query_residues_centers(nodummy_frag.GetConformer(0).GetPositions(), dataset.supocket_radius)
                pocket_atom_idx = [protein_atom_serial.index(i) for i in selected_atom_serial]
                pocket_atom_mask = torch.zeros(num_protein_atoms, dtype=torch.bool)
                pocket_atom_mask[pocket_atom_idx] = 1
                pdb_block_pocket = protein.residues_to_pdb_block(union_residues)
                protein = PDBProtein(pdb_block_pocket) # debug: rm later
                protein_dict = protein.to_dict_atom()

                # save arms-pocket sub-complex
                sim_arm_dict = parse_sdf_file(nodummy_frag)
                sim_arm_data = ProteinLigandData.from_protein_ligand_dicts(
                    protein_dict=torchify_dict(protein_dict),
                    ligand_dict=torchify_dict(sim_arm_dict)
                )
                sim_arm_data.pocket_atom_mask = pocket_atom_mask
                all_processed_sim_arms.append([sim_arm_data])

            data.sim_arms = all_processed_sim_arms

            data.src_protein_filename = meta_info['src_protein_filename']
            data.src_ligand_filename = meta_info['src_ligand_filename']
            for k, v in meta_info['data'].items():
                data[k] = v
            data.num_arms, data.num_scaffold = num_arms, num_scaffold
            data.pocket_atom_masks, data.ligand_atom_mask = pocket_atom_masks, ligand_atom_mask
            data = compute_golden_prior_from_data(data)
            data = data.to_dict()  # avoid torch_geometric version issue

        elif dataset.mode == 'arms':
            frags_sdf_path = m['sub_ligand_file']
            frags = list(Chem.SDMolSupplier(frags_sdf_path))
            for arm_idx in range(num_arms):
                protein = PDBProtein(m['sub_pocket_files'][arm_idx])
                protein_dict = protein.to_dict_atom()
                if dataset.include_dummy_atoms:
                    ligand_dict = parse_sdf_file(frags[arm_idx])
                else:
                    du = Chem.MolFromSmiles('*')
                    nodummy_frag = AllChem.ReplaceSubstructs(
                        frags[arm_idx], du, Chem.MolFromSmiles('[H]'), True)[0]
                    nodummy_frag = Chem.RemoveHs(nodummy_frag)
                    ligand_dict = parse_sdf_file(nodummy_frag)

                data = ProteinLigandData.from_protein_ligand_dicts(
                    protein_dict=torchify_dict(protein_dict),
                    ligand_dict=torchify_dict(ligand_dict),
                )
                if data.protein_pos.size(0) == 0:
                    continue
                data.src_protein_filename = meta_info['src_protein_filename']
                data.src_ligand_filename = meta_info['src_ligand_filename']
                
                data.arm_idx = arm_idx
                data.occupancy = m['pocket_occupancies_by_submol'][arm_idx]
                for k, v in meta_info['data'].items():
                    data[k] = v
                data.num_arms, data.num_scaffold = num_arms, num_scaffold
                data = data.to_dict()  # avoid torch_geometric version issue
                
        elif dataset.mode == 'scaffold':
            raise NotImplementedError
        else:
            raise ValueError
                
        return data
    except Exception as e:
        print(e)
        return None


class DecompPLPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, mode='full',
                 include_dummy_atoms=False, kekulize=True, load_retrieved_feat=False, version='v1', top_k_sim=11, supocket_radius=10):
        super().__init__()
        self.load_retrieved_feat = load_retrieved_feat

        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.mode = mode  # ['arms', 'scaffold', 'full']
        self.include_dummy_atoms = include_dummy_atoms
        self.kekulize = kekulize
        self.top_k_sim = top_k_sim
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_{mode}_{version}.lmdb')
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path),
                                         os.path.basename(self.raw_path) + f'_{mode}_{version}_name2id.pt')
        self.supocket_radius = supocket_radius
        
        self.transform = transform
        self.mode = mode
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        print('Load dataset from %s' % self.processed_path)
        if not os.path.exists(self.name2id_path):
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _precompute_name2id(self):
        name2id = defaultdict(list)
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = data.src_ligand_filename[:-4]
            name2id[name].append(i)
        torch.save(name2id, self.name2id_path)

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        num_data = 0

        arm_info = torch.load('data/arm_info/arm_info_3.pt') # obtained form dock_training_arms
        sub_ligand_file_dict = arm_info['sub_ligand_file_dict']
        docked_arms = arm_info['docked_arms']

        with db.begin(write=True, buffers=True) as txn:
            with ThreadPool(processes=300) as pool:
                multiple_fns = [pool.apply_async(extract_info_from_meta, (self, idx, meta_info, \
                    docked_arms, sub_ligand_file_dict)) for idx, meta_info in enumerate(index)]
                for idx, fn in enumerate(multiple_fns):
                    res = fn.get()
                    if res is not None:
                        data = res
                        txn.put(
                            key=f'{num_data:08d}'.encode(),
                            value=pickle.dumps(data)
                        )
                        num_data += 1
                    else:
                        num_skipped += 1
                        print('Skipping (%d) %s' % (num_skipped, index[idx]['src_ligand_filename'], ))
                        continue

        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        if self.load_retrieved_feat:
            selected_sim_arms = []
            for sim_arms in data.sim_arms:
                selected_sim_arms.append(random.choice(sim_arms[:self.top_k_sim]))
            data.selected_sim_arms = selected_sim_arms

        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_raw_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        return data
        

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--mode', type=str, default='full')
    parser.add_argument('--dummy', type=eval, default=False)
    parser.add_argument('--keku', type=eval, default=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--protein_root', type=str, default='data/crossdocked_v1.1_rmsd1.0_processed')
    parser.add_argument('--exhaustiveness', type=int, default=32)
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')
    dataset = DecompPLPairDataset(args.path, mode=args.mode,
                                  include_dummy_atoms=args.dummy, kekulize=args.keku, version=args.version)
    print(len(dataset))
