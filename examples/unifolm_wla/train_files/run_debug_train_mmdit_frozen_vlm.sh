export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# Debug launch: frozen VLM backbone, single 24GB GPU, single-source Unitree data.
config_yaml=./unifolm_wla/config/training/mmdit_debug_frozen_vlm.yaml
run_root_dir=./playground/Checkpoints
run_id=debug_mmdit_frozen_vlm
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

num_processes=${NUM_PROCESSES:-1}

.venv/bin/accelerate launch \
  --config_file unifolm_wla/config/deepseeds/deepspeed_zero2_debug_cpu_offload.yaml \
  --num_processes ${num_processes} \
  unifolm_wla/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
  # --is_debug True
