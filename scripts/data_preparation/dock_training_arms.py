import os, sys
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from copy import deepcopy
from utils.evaluation.docking_vina_track import VinaDockingTask
from scripts.evaluate_mol_in_place_best import read_pose

import ray
ray.init()

@ray.remote
def get_dock_pose(idx, arm, src_ligand_filename):
    print(idx)
    if arm is None:
        return arm
    try:
        du = Chem.MolFromSmiles('*')
        nodummy_frag = AllChem.ReplaceSubstructs(
            arm, du, Chem.MolFromSmiles('[H]'), True)[0]
        nodummy_frag = Chem.RemoveHs(nodummy_frag) # to debug: is there any H contained in arms?
        
        vina_task = VinaDockingTask.from_generated_mol(deepcopy(nodummy_frag),
                                src_ligand_filename, protein_root='data/crossdocked_v1.1_rmsd1.0')
        pose = vina_task.run(mode='minimize', exhaustiveness=32)[0]['pose']
        arm = read_pose(pose)
    except Exception as e:
        global denum
        global dock_error
        denum += 1
        print('dock error:', denum)
        dock_error.append(idx)
        return arm

    return arm

if __name__ == "__main__":
    dock_error = []
    denum = 0

    arm_info = torch.load('data/arm_info/arm_info_2.pt')
    arm_list = arm_info['arm_list']
    sub_ligand_file_dict = list(arm_info['sub_ligand_file_dict'].keys())
    arm2ligand = {}
    for idx, arms in enumerate(arm_info['arm_id_list']):
        for arm in arms:
            arm2ligand[arm] = idx
    sub_ligand_file = [sub_ligand_file_dict[i] for i in arm2ligand.values()]

    fns = []
    for idx, (arm, ligandf) in enumerate(zip(arm_list, sub_ligand_file)):
        fns.append(get_dock_pose.remote(idx, arm, os.path.join(ligandf.split('/')[3], ligandf.split('/')[4][:-10]+'.sdf')))
        
    docked_arms = ray.get(fns)
    arm_info['docked_arms'] = docked_arms
    torch.save(arm_info, 'arm_info_3.pt')
    torch.save(dock_error, 'dock_error.pt')
