ssh draco
cd Documents/Thesis/elsa-cybersecurity/
conda init
conda activate android
export PYTHONPATH="${PYTHONPATH}:android-detectors/src"
echo ${PYTHONPATH}
bash ~ptgm/arsr-servers.sh gpuload

python track_1/incr_evaluation.py -classifier FFNN_small_CEL_weights -num_feats_attacked_init 30 -num_feats_attacked_stop 41 -num_feats_attacked_stride 5 -acc_threshold 0.2 -num_samples_to_attack 200
