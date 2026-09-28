export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# Fine-tune a released UnifoLM-WLA-Base checkpoint (frozen VLM, single 24GB GPU).
#
# === Please modify the following paths according to your environment ===
base_model_dir=${base_model_dir:-/path/to/UnifoLM-WLA-1.0-Base}
config_yaml=./unifolm_wla/config/training/mmdit_finetune_frozen_vlm.yaml
data_config_path=./unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml
run_root_dir=./playground/Checkpoints
run_id=finetune_wla_base_frozen_vlm
# === End of environment variable configuration ===
###########################################################################################
export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

num_processes=${NUM_PROCESSES:-1}

.venv/bin/accelerate launch \
  --config_file unifolm_wla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  unifolm_wla/training/train_unifolm_wla.py \
  --config_yaml ${config_yaml} \
  --framework.qwenvl.base_vlm ${base_model_dir}/tokenizer \
  --trainer.pretrained_checkpoint ${base_model_dir}/checkpoints/model.safetensors \
  --datasets.vla_data.data_config_path ${data_config_path} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}

