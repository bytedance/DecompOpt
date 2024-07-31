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
best_res_dir=${output_dir}/best_record
rm -rf $best_res_dir
mkdir $best_res_dir

for ((dup_id=0; dup_id<=$dup_num; dup_id++))
do
    padded_dup_id=`printf "%03d" ${dup_id}`
    if (( $dup_id==0 ))
    then
      ckpt_path="logs_diffusion_full/pretrained_cond_decompdiff/decompdiff.pt"
    else
      # decompose can fail
      if test -f ${best_res_dir}/best_mol_arms.pt; then
        ckpt_path="logs_diffusion_full/pretrained_cond_decompdiff/cond_decompdiff.pt"
      else
        ckpt_path="logs_diffusion_full/pretrained_cond_decompdiff/decompdiff.pt"
      fi
    fi
    echo $ckpt_path
    
    batch_size=20
    while ((batch_size >= 1)); do
      python scripts/sample_diffusion_decomp_compose.py \
        configs/sampling_10_ret.yml \
        --ckpt_path ${ckpt_path} \
        --outdir ${output_dir} \
        -i ${data_id} \
        --reference_arm_path ${best_res_dir}/best_mol_arms.pt \
        --dup_id ${dup_id} \
        --batch_size ${batch_size} \
        --prior_mode ${prior_mode} || { exit 817; } # beta_prior
      echo output_dir=${output_dir}
      if [ $? -eq 0 ]; then
          echo "Program executed successfully with batch_size: $batch_size"
          break
      elif [ $batch_size -eq 1 ]; then
          echo "Program failed with batch_size 1. Exiting with error."
          exit 1
      else
          echo "Program failed with batch_size: $batch_size. Trying with a smaller batch size."
          # If the program failed, reduce the batch size
          batch_size=$((batch_size / 2))
      fi
    done

    sample_dir=${output_dir}/sampling_10_ret-${prior_mode}-${padded_data_id}-${padded_dup_id}
    echo sample_dir=${sample_dir}
    python scripts/evaluate_mol_in_place_compose.py \
      ${sample_dir} \
      --data_id $data_id \
      --best_res_dir $best_res_dir \
      --top_k $top_k --protein_root data/test_set
    # log best dir
    cp ${best_res_dir}/best_mol_arms.pt ${sample_dir}/eval
done

