ssh draco
cd Documents/Thesis/elsa-cybersecurity/
conda init
conda activate android
export PYTHONPATH="${PYTHONPATH}:android-detectors/src"
echo ${PYTHONPATH}
bash ~ptgm/arsr-servers.sh gpuload

python track_1/FFNN_track_1.py -training normal -structure small -use_CEL True -CEL_weight_pos_class 0.1 -CEL_weight_neg_class 0.9
python track_1/FFNN_track_1.py -training normal -structure big -use_CEL True -CEL_weight_pos_class 0.1 -CEL_weight_neg_class 0.9
python track_1/FFNN_track_1.py -training normal -structure small -use_CEL True -CEL_weight_pos_class 0.2 -CEL_weight_neg_class 0.8
python track_1/FFNN_track_1.py -training ratioed -structure big -dense True
python track_1/FFNN_track_1.py -training normal -structure big
python track_1/FFNN_track_1.py -training normal -structure small
python track_1/FFNN_track_1.py -training ratioed -structure big
python track_1/FFNN_track_1.py -training normal -structure big -dense True
python track_1/FFNN_track_1.py -training normal -structure small -use_CEL True -CEL_weight_pos_class 0.1 -CEL_weight_neg_class 0.9 -dense True
python track_1/FFNN_track_1.py -training normal -structure small -use_CEL True -CEL_weight_pos_class 0.1 -CEL_weight_neg_class 0.9 -sub_addition _big_fsa_eval

python track_1/incr_evaluation.py -classifier FFNN_small_CEL_weights -num_feats_attacked_init 30 -num_feats_attacked_stop 41 -num_feats_attacked_stride 5 -acc_threshold 0.2 -num_samples_to_attack 200

python track_1/data_augmentation.py -classifier "FFNN_normal_small_CEL0109_" -adv_mode "genetic" -n_feats 5 -n_good_samples 2 -n_mal_samples 2
python track_1/data_augmentation.py -classifier "FFNN_normal_small_CEL0109_" -adv_mode "genetic" -n_feats 5 -n_good_samples 10000 -n_mal_samples 1000
python track_1/data_augmentation.py -classifier "FFNN_normal_small_CEL0109_" -adv_mode "genetic" -n_feats 10 -n_good_samples 10000 -n_mal_samples 1000

python track_1/FFNN_track_1.py -training normal -structure small -use_CEL True -CEL_weight_pos_class 0.1 -CEL_weight_neg_class 0.9 -adv_mode "genetic" -n_feats 5 -n_good_samples 10000 -n_mal_samples 1000
python track_1/FFNN_track_1.py -training normal -structure small -use_CEL True -CEL_weight_pos_class 0.1 -CEL_weight_neg_class 0.9 -adv_mode "genetic" -n_feats 10 -n_good_samples 10000 -n_mal_samples 1000

python track_1/data_augmentation.py -classifier "FFNN_normal_small_CEL0208_" -adv_mode "genetic" -n_feats 5 -n_good_samples 10000 -n_mal_samples 1000
python track_1/data_augmentation.py -classifier "FFNN_normal_small_CEL0208_" -adv_mode "genetic" -n_feats 10 -n_good_samples 10000 -n_mal_samples 1000

python track_1/FFNN_track_1.py -training normal -structure small -use_CEL True -CEL_weight_pos_class 0.2 -CEL_weight_neg_class 0.8 -adv_mode "genetic" -n_feats 5 -n_good_samples 10000 -n_mal_samples 1000
python track_1/FFNN_track_1.py -training normal -structure small -use_CEL True -CEL_weight_pos_class 0.2 -CEL_weight_neg_class 0.8 -adv_mode "genetic" -n_feats 10 -n_good_samples 10000 -n_mal_samples 1000

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Feature Selection
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
python track_1/drebin_track_1.py -feat_selection Variance -p 0.4

python track_1/drebin_track_1.py -feat_selection Univariate -selection_type k_best -selection_function chi2 -param 1000
python track_1/drebin_track_1.py -feat_selection Univariate -selection_type k_best -selection_function mutual_info_classif -param 10000
python track_1/drebin_track_1.py -feat_selection Univariate -selection_type k_best -selection_function f_classif -param 10000
python track_1/drebin_track_1.py -feat_selection Univariate -selection_type percentile -selection_function chi2 -param 60
python track_1/drebin_track_1.py -feat_selection Univariate -selection_type percentile -selection_function mutual_info_classif -param 60
python track_1/drebin_track_1.py -feat_selection Univariate -selection_type percentile -selection_function f_classif -param 60

python track_1/drebin_track_1.py -feat_selection Recursive -estimator SVR -param 60
python track_1/drebin_track_1.py -feat_selection RecursiveCV -estimator SVR -param 60

python track_1/drebin_track_1.py -feat_selection SelectFromModel -estimator SVR -param 6
python track_1/drebin_track_1.py -feat_selection SelectFromModel -estimator TreeEnsemble -param 6

python track_1/drebin_track_1.py -feat_selection Sequential -estimator SVR -direction forward -param 2
python track_1/drebin_track_1.py -feat_selection Sequential -estimator SVR -direction backward -param 2

python track_1/FFNN_track_1.py -training normal -structure small -use_CEL True -CEL_weight_pos_class 0.1 -CEL_weight_neg_class 0.9 -feat_selection Univariate -selection_type k_best -selection_function mutual_info_classif -param 10000

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Adversarial Training
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
python track_1/adv_training.py -classifier FFNN_normal_small_CEL0109_ -manipulation_algo genetic -manipulation_degree 1 -step 37500 -ATsize 10 -ATratio 9 