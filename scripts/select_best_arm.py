import argparse
import os, shutil
import json
import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir', type=str)
    parser.add_argument('--metric', type=str, default='success_rate')
    parser.add_argument('--qed_w', type=int, default=1)
    parser.add_argument('--sa_w', type=int, default=1)
    parser.add_argument('--vina_w', type=int, default=1)
    args = parser.parse_args()

    metric_list, complete_list, score_list, ckpt_list = [], [], [], []
    sample_list = os.listdir(args.outdir)
    for sample_path in sample_list:
        if 'sampling_10_ret' not in sample_path:
            continue
        
        try:
            eval_res = torch.load(os.path.join(args.outdir, sample_path, 'eval_full/metrics.pt'))
        except:
            print('eval failed:', sample_path)
            continue

        # evaluate result
        metircs = []
        linear_scalar = []
        for res in eval_res:
            qed = res['chem_results']['qed']
            sa = res['chem_results']['sa']
            vina = res['vina']['score_only'][0]['affinity']
            score = args.qed_w * qed + args.sa_w * sa - args.vina_w * vina / 12
            linear_scalar.append(score)

            if args.metric == 'success_rate':
                if res['chem_results']['qed'] >= 0.25 and res['chem_results']['sa'] >= 0.59 and\
                    res['vina']['dock'][0]['affinity'] <= -8.18:
                    metircs.append(1)
                else:
                    metircs.append(0)
            elif args.metric in ['qed', 'sa']:
                metircs.append(res['chem_results'][args.metric])

            elif args.metric == 'score_only':
                metircs.append(-1 * res['vina'][args.metric][0]['affinity'])

            elif args.metric == 'vina_min':
                metircs.append(-1 * res['vina']['minimize'][0]['affinity'])

            else:
                raise NotImplementedError

        if len(metircs) > 0:
            if args.metric == 'success_rate':
                metric_list.append(np.sum(metircs)/len(eval_res))
            else:
                metric_list.append(np.mean(metircs))
        
        if len(linear_scalar) > 0:
            score_list.append(np.mean(linear_scalar))

        if len(metircs) > 0 or len(linear_scalar) > 0:
            ckpt_list.append(sample_path)
            complete_list.append(len(eval_res)/20)
        
    res_list = list(zip(metric_list, complete_list, score_list, ckpt_list))
    res_list.sort(key=lambda x:(x[0], x[1], x[2]))
    print(res_list)

    best_res_path = os.path.join(args.outdir, res_list[-1][3])
    os.makedirs(os.path.join(os.path.join(args.outdir, 'best_record')), exist_ok=True)
    shutil.copyfile(os.path.join(best_res_path, 'eval/best_mol_arms.pt'), 
                    os.path.join(args.outdir, 'best_record/best_mol_arms.pt'))
    shutil.copytree(best_res_path, 
                    os.path.join(args.outdir, 'best_res'), dirs_exist_ok=True)
