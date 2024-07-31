data_id=$1
outdir=$2
top_k=3
dup_num=30
ref_list=(4 6 12 19 21 22 42 44 49 53 54 59 70 72 73 74 75 76 79 80 82 83 84 88 97 98)

ln -s /mnt/bn/molecule/DecompDiff/outputs outputs
ln -s /mnt/bn/molecule/DecompDiff/data data
ln -s /mnt/bn/molecule/DecompDiff/logs_diffusion_full logs_diffusion_full
ln -s /mnt/bn/molecule/DecompDiff/pregen_info pregen_info

source "$CONDA_DIR"/etc/profile.d/conda.sh
conda activate decompdiff 

python -m pip install numpy==1.23.1
python -m pip install meeko==0.3.0

# opt prior
prior_mode="beta_prior"
for ref_id in "${ref_list[@]}"; do
  if [ "$ref_id" -eq "$data_id" ]; then
    prior_mode="ref_prior"
    break
  fi
done

padded_data_id=`printf "%03d" ${data_id}`
output_dir=${outdir}/sampling_${padded_data_id}

for ((dup_id=0; dup_id<=$dup_num; dup_id++))
do
  padded_dup_id=`printf "%03d" ${dup_id}`
  sample_dir=${output_dir}/sampling_10_ret-${prior_mode}-${padded_data_id}-${padded_dup_id}
  echo sample_dir=${sample_dir}
  python scripts/evaluate_mol_in_place_compose.py \
    ${sample_dir} \
    --data_id $data_id \
    --select_cond False \
    --protein_root data/test_set \
    --docking_mode vina_full \
    --result_path ${sample_dir}/eval_full
done
