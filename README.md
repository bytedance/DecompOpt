# DecompOpt: Controllable and Decomposed Diffusion Models for Structure-based Molecular Optimization

This repository is the official implementation of _DecompOpt: Controllable and Decomposed Diffusion Models for Structure-based Molecular Optimization._


## Dependencies
### Install via Conda and Pip
```bash
conda create -n decompdiff python=3.8
conda activate decompdiff
conda install numpy==1.22.3
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
conda install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge

# For decomposition
conda install -c conda-forge mdtraj
pip install alphaspace2

# For Vina Docking
pip install meeko==0.3.0 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

## Preprocess 
We decomposed molecules in CrossDocked2020 trainig set into arms and stored processed data in `arm_info_2.pt`, which can be downloaded [here](https://huggingface.co/datasets/Annie37/DecompOpt/blob/main/arm_info_2.pt). Then we docked arms with target protein with Vina Minimize and obtained docked arm conformations as conditions for training.
```bash
python scripts/data_preparation/dock_training_arms.py
```
We follow the preprocess of [DecompDiff](https://github.com/bytedance/DecompDiff). We have provided processed dataset [here](https://huggingface.co/datasets/Annie37/DecompOpt/tree/main).

## Training
To train the model from scratch, you need to download the `*.lmdb`, `*_name2id.pt` and `split_by_name.pt` files and put them in the `./data` directory. Then, you can run the following command:
```bash
python scripts/train_diffusion_decompopt.py configs/training.yml
```

## Sampling and Evaluation
To sample molecules given protein pockets in the test set, you need to download `test_index.pkl` and `*_eval.tar.gz` files, unzip it and put them in the `./data` directory. To sample molecules with beta priors, you also need to download `beta_priors.zip` and `natom_models.pkl` and put them in the `./pregen_info` directory. Then, you can run the following command:
```bash
bash scripts/run/sample_compose.sh ${data_id} ${outdir}
```
This script samples for opt prior by default. We have provided the trained model checkpoint [here](https://huggingface.co/datasets/Annie37/DecompOpt/tree/main). You need to download both `decompdiff.pt` and `decompopt.pt`.
After sampling, Vina Dock is evaluated and the best results are selected:
```bash
bash scripts/run/eval_vina_full.sh ${data_id} ${outdir}
python scripts/select_best_arm.py ${outdir}
```

## BibTex
```
@inproceedings{
    zhou2024decompopt,
    title={DecompOpt: Controllable and Decomposed Diffusion Models for Structure-based Molecular Optimization},
    author={Xiangxin Zhou and Xiwei Cheng and Yuwei Yang and Yu Bao and Liang Wang and Quanquan Gu},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=Y3BbxvAQS9}
}
```
