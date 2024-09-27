import argparse
import os
import numpy as np
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from copy import deepcopy

from meeko import PDBQTMolecule
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem

from utils import misc
from utils.evaluation import scoring_func
from utils.evaluation.docking import QVinaDockingTask
from utils.evaluation.docking_vina_track import VinaDockingTask
from utils.evaluation import eval_bond_length
from utils.transforms import get_atomic_number_from_index
from utils.data import PDBProtein, ProteinLigandData, torchify_dict, parse_sdf_file
from utils.reconstruct import fix_aromatic, fix_valence, MolReconsError


def decompose_generated_ligand(r):
    mask = np.array(r['decomp_mask'])
    mol = r['mol']
    arms = []

    for arm_idx in range(max(r['decomp_mask']) + 1):
        atom_indices = np.where(mask == arm_idx)[0].tolist()
        
        arm = get_submol_from_mol(mol, atom_indices)
        if arm is None:
            print(f"[fail] to extract submol (arm).")
            return None
        smi = Chem.MolToSmiles(arm)
        if "." in smi:
            print(f"[fail] incompleted arm: {smi}")
            return None 
        arms.append(arm)
    return arms

def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')



def fix_mol(mol):
    try:
        Chem.SanitizeMol(mol)
        fixed = True
    except Exception as e:
        fixed = False

    if not fixed:
        try:
            Chem.Kekulize(deepcopy(mol))
        except Chem.rdchem.KekulizeException as e:
            err = e
            if 'Unkekulized' in err.args[0]:
                mol, fixed = fix_aromatic(mol)

    # valence error for N
    if not fixed:
        mol, fixed = fix_valence(mol)

    # print('s2')
    if not fixed:
        mol, fixed = fix_aromatic(mol, True)

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise MolReconsError()
        # return None

    return mol


def read_pose(pose:str):
    rdkitmol_list = [] 
    pdbqt_mol = PDBQTMolecule(pose, skip_typing=True)
    for pmol in pdbqt_mol:
        try:
            rdmol = pmol.export_rdkit_mol()
        except Exception:
            print(
                f"Failed to parse docking pose"
            )
            continue
        rdkitmol_list.append(rdmol)

    return rdkitmol_list[0]


def eval_single_datapoint(index, id, args, supocket_radius=10):
    if isinstance(index, dict):
        # reference set
        index = [index]

    ligand_filename = index[0]['ligand_filename']
    num_samples = len(index[:100])
    results = []
    n_eval_success = 0
    all_pair_dist, all_bond_dist = [], []
    for sample_idx, sample_dict in enumerate(tqdm(index[:num_samples], desc='Eval', total=num_samples)):
        mol = sample_dict['mol']
        smiles = sample_dict['smiles']

        if mol is None or '.' in smiles:
            continue

        # chemical and docking check
        try:
            chem_results = scoring_func.get_chem(mol)
            if args.docking_mode == 'qvina':
                vina_task = QVinaDockingTask.from_generated_mol(mol, ligand_filename, protein_root=args.protein_root)
                vina_results = vina_task.run_sync()
            elif args.docking_mode == 'vina':
                vina_task = VinaDockingTask.from_generated_mol(mol, ligand_filename, protein_root=args.protein_root)
                vina_results = vina_task.run(mode='dock')
            elif args.docking_mode in ['vina_full', 'vina_score']:
                vina_task = VinaDockingTask.from_generated_mol(deepcopy(mol),
                                                                ligand_filename, protein_root=args.protein_root)
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                vina_results = {
                    'score_only': score_only_results,
                    'minimize': minimize_results
                }
                if args.docking_mode == 'vina_full':
                    dock_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                    vina_results.update({
                        'dock': dock_results,
                    })
            elif args.docking_mode == 'none':
                vina_results = None
            else:
                raise NotImplementedError
        except Exception as e:
            print(e)
            continue

        n_eval_success += 1

        pred_pos, pred_v = sample_dict['pred_pos'], sample_dict['pred_v']
        pred_atom_type = get_atomic_number_from_index(pred_v, mode='add_aromatic')
        pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
        all_pair_dist += pair_dist

        bond_dist = eval_bond_length.bond_distance_from_mol(mol)
        all_bond_dist += bond_dist

        results.append({
            **sample_dict,
            'chem_results': chem_results,
            'vina': vina_results
        })
    logger.info(f'Evaluate No {id} done! {num_samples} samples in total. {n_eval_success} eval success!')
    if args.result_path:
        torch.save(results, os.path.join(args.result_path, f'metrics.pt'))


    if args.select_cond:
        fragopt_mode = 'arm-wise' # 'mol-wise'
        if fragopt_mode == 'mol-wise':
            stable_indices = []
            stability_list = []

            with Chem.SDWriter(os.path.join(args.result_path, 'mols.sdf')) as w:
                for rid, r in enumerate(results):
                    mol = r['mol']
                    if mol is None:
                        continue
                    print(Chem.MolToSmiles(mol))
                    w.write(mol)
                    if 'chem_results' in r and 'vina' in r:
                        qed = r['chem_results']['qed']
                        sa = r['chem_results']['sa']
                        if qed > 0.45 and sa > 0.6:
                            stable_indices.append(rid)
                        stability_list.append((qed+sa, rid))
            
            if len(stable_indices) == 0:
                # no stable indices
                # then use the stablest one
                stability_list.sort(reverse=True)
                best_id = stability_list[0][1]
            else:
                best_vina_id = -1
                best_vina = 9999999
                for rid in stable_indices:
                    r = results[rid]
                    vina_score = r['vina']['score_only'][0]['affinity']
                    if vina_score < best_vina:
                        best_vina = vina_score
                        best_vina_id = rid  
                best_id = best_vina_id 
                
            best_res = results[best_id]
            with Chem.SDWriter(os.path.join(args.result_path, 'best_mol.sdf')) as w:
                best_mol = results[best_id]['mol']
                w.write(best_mol)

            mol = best_res['mol']
            qed = best_res['chem_results']['qed']
            sa = best_res['chem_results']['sa']
            vina_score = best_res['vina']['score_only'][0]['affinity']

            print("Best mol properties:")
            print(f"--> SMILES: {Chem.MolToSmiles(mol)}")
            print(f"--> QED: {qed:.2f}")
            print(f"--> SA: {sa:.2f}")
            print(f"--> Vina Score: {vina_score:.2f}")

            vina_task = VinaDockingTask.from_generated_mol(deepcopy(mol),
                                                            ligand_filename, protein_root=args.protein_root)
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
            pose = minimize_results[0]['pose']
            mol = read_pose(pose)
            best_res['mol'] = mol
            best_arms = decompose_generated_ligand(best_res)
            with Chem.SDWriter(os.path.join(args.result_path, 'best_mol_arms.sdf')) as w:
                for arm in best_arms:
                    arm = fix_mol(arm)
                    w.write(arm)

        elif fragopt_mode == 'arm-wise':
            with Chem.SDWriter(os.path.join(args.result_path, 'mols.sdf')) as w:
                arms_dict = defaultdict(list)
                for rid, r in enumerate(results):
                    mol = r['mol']
                    if mol is None:
                        print(f'{rid}: mol is None.')
                        continue
                    print(f'mol-{rid}:', Chem.MolToSmiles(mol))
                    w.write(mol)
                    if 'chem_results' in r and 'vina' in r:
                        arms = decompose_generated_ligand(r)
                        if arms is not None:
                            for arm_id, arm in enumerate(arms):
                                print(f'mol-{rid}-arm-{arm_id}:', Chem.MolToSmiles(arm))

                                try:
                                    _arm = deepcopy(arm)
                                    Chem.Kekulize(_arm)
                                except:
                                    try:
                                        Chem.SanitizeMol(arm)
                                        _arm = deepcopy(arm)
                                        Chem.Kekulize(_arm)
                                    except:
                                        print("Cannot Keku")
                                        continue
                                
                                try:
                                    vina_task = VinaDockingTask.from_generated_mol(deepcopy(arm),
                                                            ligand_filename, protein_root=args.protein_root)
                                    minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                                    vina_min = minimize_results[0]['affinity']
                                    pose = minimize_results[0]['pose']
                                    arm = read_pose(pose)
                                except:
                                    continue

                                # remove dummy atoms and H
                                du = Chem.MolFromSmiles('*')
                                nodummy_frag = AllChem.ReplaceSubstructs(
                                    arm, du, Chem.MolFromSmiles('[H]'), True)[0]
                                arm = Chem.RemoveHs(nodummy_frag)

                                # get condition arm's subpocket: extract by radius
                                protein = PDBProtein(os.path.join('data/crossdocked_v1.1_rmsd1.0_processed',\
                                    ligand_filename[:-4] + '_pocket.pdb')) # debug: rm later
                                protein_atom_serial = [atom['atom_id'] for atom in protein.atoms]
                                num_protein_atoms = len(protein.atoms)
                                selected_atom_serial, union_residues = protein.query_residues_centers(arm.GetConformer(0).GetPositions(), supocket_radius)
                                pocket_atom_idx = [protein_atom_serial.index(i) for i in selected_atom_serial]
                                pocket_atom_mask = torch.zeros(num_protein_atoms, dtype=torch.bool)
                                pocket_atom_mask[pocket_atom_idx] = 1
                                pdb_block_pocket = protein.residues_to_pdb_block(union_residues)
                                protein = PDBProtein(pdb_block_pocket)
                                protein_dict = protein.to_dict_atom()

                                chem_results = scoring_func.get_chem(arm)
                                qed = chem_results['qed']
                                sa = chem_results['sa']

                                print(f'mol-{rid}-arm-{arm_id} => qed:{qed:.2f},sa:{sa:.2f},vina_min:{vina_min:.2f}')
                                
                                sim_arm_dict = parse_sdf_file(arm)
                                arm_data = ProteinLigandData.from_protein_ligand_dicts(
                                    protein_dict=torchify_dict(protein_dict),
                                    ligand_dict=torchify_dict(sim_arm_dict),
                                )
                                arm_data.pocket_atom_mask = pocket_atom_mask
                                arms_dict[arm_id].append({'reward':{'qed': qed, 'sa': sa, 'vina_min': vina_min}, 'sample_path':args.sample_dir, \
                                    'rid':rid, 'arm':arm_data})
                        else:
                            print(f'{rid}: arms are None')

            if len(arms_dict) == 0:
                # no vaild arms generated, do not update
                return results, all_pair_dist, all_bond_dist

            # load best arm
            if os.path.exists(os.path.join(args.best_res_dir, 'best_mol_arms.pt')):
                best_arms = torch.load(os.path.join(args.best_res_dir, 'best_mol_arms.pt'))
            else:
                best_arms = None
            selected_arms = defaultdict(list)
            os.makedirs(args.best_res_dir, exist_ok=True)
            if best_arms is not None:
                for arm_id, arms in best_arms.items():
                    arms_dict[arm_id].extend(arms)

            for arm_id, arms in arms_dict.items():
                rewards = [a['reward'] for a in arms]
                qeds = [r['qed'] for r in rewards]
                sas = [r['sa'] for r in rewards]
                vinas = [-1 * r['vina_min'] for r in rewards]
                qed_mean, sa_mean, vina_mean = np.mean(qeds), np.mean(sas), np.mean(vinas)
                qed_std, sa_std, vina_std = np.std(qeds), np.std(sas), np.std(vinas)
                qed_zscore = [(s - qed_mean)/(qed_std + 1e-6) for s in qeds]
                sa_zscore = [(s - sa_mean)/(sa_std + 1e-6) for s in sas]
                vina_zscore = [(s - vina_mean)/(vina_std + 1e-6) for s in vinas]
                z_score = [qed_zscore[i] * args.zqed_weight + sa_zscore[i] * args.zsa_weight + vina_zscore[i] * args.zvina_weight for i in range(len(rewards))]

                for z, a in zip(z_score, arms):
                    a['z_score'] = z

                arms.sort(key=lambda x:(x['z_score'], x['sample_path']))
                print(arm_id, "->", [a['z_score'] for a in arms])
                for i in range(min(args.top_k, len(arms))):
                    selected_arms[arm_id].append(arms[-1-i])

            torch.save(selected_arms, os.path.join(args.best_res_dir, 'best_mol_arms.pt'))

    return results, all_pair_dist, all_bond_dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_dir', type=str)
    parser.add_argument('--result_path', type=str, default=None)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--data_id', type=int, default=2)
    parser.add_argument('--protein_root', type=str, default='data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--docking_mode', type=str, default='vina_score',
                        choices=['none', 'qvina', 'vina', 'vina_full', 'vina_score'])
    parser.add_argument('--exhaustiveness', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--best_res_dir', type=str, default='outputs_test/best_record')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--select_cond', type=eval, default=True)
    parser.add_argument('--zqed_weight', type=float, default=1)
    parser.add_argument('--zsa_weight', type=float, default=1)
    parser.add_argument('--zvina_weight', type=float, default=1)
    args = parser.parse_args()

    if args.result_path is None:
        args.result_path = os.path.join(args.sample_dir, "eval")
    os.makedirs(args.result_path, exist_ok=True)

    logger = misc.get_logger('evaluate', args.result_path)
    logger.info(f"data_id={args.data_id}")
    logger.info(args)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')
    
    test_index = torch.load(os.path.join(args.sample_dir, "result.pt"))
    testset_results = []
    testset_pair_dist, testset_bond_dist = [], []
    r, pd, bd = eval_single_datapoint(test_index, args.data_id, args)
    testset_results.append(r)
    testset_pair_dist += pd
    testset_bond_dist += bd

    qed = [x['chem_results']['qed'] for r in testset_results for x in r]
    sa = [x['chem_results']['sa'] for r in testset_results for x in r]
    num_atoms = [len(x['pred_pos']) for r in testset_results for x in r]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    logger.info('Num atoms:   Mean: %.3f Median: %.3f' % (np.mean(num_atoms), np.median(num_atoms)))
    if args.docking_mode in ['vina', 'qvina']:
        vina = [x['vina'][0]['affinity'] for r in testset_results for x in r]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ['vina_full', 'vina_score']:
        vina_score_only = [x['vina']['score_only'][0]['affinity'] for r in testset_results for x in r]
        vina_min = [x['vina']['minimize'][0]['affinity'] for r in testset_results for x in r]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_full':
            vina_dock = [x['vina']['dock'][0]['affinity'] for r in testset_results for x in r]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))
