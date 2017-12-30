#!/bin/bash
# You need to modify to your dataset path
TEST_WAV_DIR="/home/usuaris/veu/tfgveu9/Task4/wav_files/testing"
TRAIN_WAV_DIR="/home/usuaris/veu/tfgveu9/Task4/wav_files/training"
EVALUATION_WAV_DIR="/home/usuaris/veu/tfgveu9/Task4/wav_files/evaluation"

# You can to modify to your own workspace. 
WORKSPACE="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp"
VERSION="_360"
# Extract features
#python prepare_data.py extract_features --wav_dir=$TEST_WAV_DIR --out_dir=$WORKSPACE"/features/logmel128_360/testing" --recompute=True
#python prepare_data.py extract_features --wav_dir=$TRAIN_WAV_DIR --out_dir=$WORKSPACE"/features/logmel128_360/training" --recompute=True
#python prepare_data.py extract_features --wav_dir=$EVALUATION_WAV_DIR --out_dir=$WORKSPACE"/features/logmel128_360/evaluation" --recompute=True

# Pack features
#python prepare_data.py pack_features --fe_dir=$WORKSPACE"/features/logmel128_360/testing" --csv_path="meta_data/testing_set.csv" --out_path=$WORKSPACE"/packed_features/logmel128_360/testing.h5"
#python prepare_data.py pack_features --fe_dir=$WORKSPACE"/features/logmel128_360/training" --csv_path="meta_data/training_shuffled.csv" --out_path=$WORKSPACE"/packed_features/logmel128_360/training.h5"
#python prepare_data.py pack_features --fe_dir=$WORKSPACE"/features/logmel128_360/evaluation" --csv_path="" --out_path=$WORKSPACE"/packed_features/logmel128_360/evaluation.h5"

# Calculate scaler
#python prepare_data.py calculate_scaler --hdf5_path=$WORKSPACE"/packed_features/logmel128_360/training.h5" --out_path=$WORKSPACE"/scalers/logmel/training"$VERSION".scaler"

# Train SED
#python main_crnn_sed_v3.py train --tr_hdf5_path=$WORKSPACE"/packed_features/logmel128_360/training.h5" --te_hdf5_path=$WORKSPACE"/packed_features/logmel128_360/testing.h5" --scaler_path=$WORKSPACE"/scalers/logmel/training"$VERSION".scaler" --out_model_dir=$WORKSPACE"/models/crnn_sed"$VERSION

# Recognize SED
#python main_crnn_sed_v3.py recognize --te_hdf5_path=$WORKSPACE"/packed_features/logmel128_360/testing.h5" --scaler_path=$WORKSPACE"/scalers/logmel/training"$VERSION".scaler" --model_dir=$WORKSPACE"/models/crnn_sed"$VERSION --out_dir=$WORKSPACE"/preds/crnn_sed"$VERSION

# Get stat of SED
#python main_crnn_sed_v3.py get_stat --pred_dir=$WORKSPACE"/preds/crnn_sed"$VERSION --stat_dir=$WORKSPACE"/stats/crnn_sed" --submission_dir=$WORKSPACE"/submissions/crnn_sed"

#python main_crnn_sed_v3.py recognize --te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_360/testing.h5" --scaler_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/scalers/logmel/training_360.scaler" --model_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/models/crnn_sed_360" --out_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/preds/crnn_sed_360"
#python main_crnn_sed_v3.py get_stat --pred_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/preds/crnn_sed_360" --stat_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/stats/crnn_sed" --submission_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/submissions/crnn_sed"
#python prepare_data_v2.py pack_features --fe_dir=$WORKSPACE"/features/logmel128_360/training" --csv_path="meta_data/training_shuffled.csv" --out_path=$WORKSPACE"/packed_features/logmel128_360/training.h5"

#python prepare_data_v2.py calculate_scaler --hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_360/training.h5" --out_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/scalers/logmel/training128_360.scaler"

#python main_crnn_sed_v3.py train --tr_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_360/training.h5" --te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_360/testing.h5" --scaler_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/scalers/logmel/training128_360.scaler" --out_model_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/models/crnn_sed_128_360"

#python main_crnn_sed_v3.py recognize_trained --te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_360/testing.h5" --scaler_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/scalers/logmel/training128_360.scaler" --model_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/models/crnn_sed_128_360" --out_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/preds/crnn_sed_360_128"
#python main_crnn_sed_v3.py get_stat --pred_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/preds/crnn_sed_360_128" --stat_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/stats/crnn_sed" --submission_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/submissions/crnn_sed"

#python prepare_data.py pack_features --fe_dir=$WORKSPACE"/features/logmel128_360/training" --csv_path="meta_data/training_shuffled.csv" --out_path=$WORKSPACE"/packed_features/logmel128_360/training.h5"

#python prepare_data.py calculate_scaler --hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_360/training.h5" --out_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/scalers/logmel/training128_360.scaler"

#python main_crnn_sed_v4.py train --tr_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_360/training.h5" --te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_360/testing.h5" --scaler_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/scalers/logmel/training128_360.scaler" --out_model_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/models/crnn_sed_128_360"


#python prepare_data.py pack_features --fe_dir=$WORKSPACE"/features/logmel128_360/training" --csv_path="meta_data/training_shuffled.csv" --out_path=$WORKSPACE"/packed_features/logmel128_360/training.h5"

#python prepare_data.py calculate_scaler --hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_360/training.h5" --out_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/scalers/logmel/training128_360.scaler"

#python main_crnn_sed_v5.py train --tr_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_360/training.h5" --te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_360/testing.h5" --scaler_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/scalers/logmel/training128_360.scaler" --out_model_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/models/crnn_sed_128_360"

#python main_crnn_sed_v3.py extract_features --tr_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel64_240/training.h5" --te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel64_240/testing.h5" --scaler_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/scalers/logmel/training64_240.scaler" --model_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/models/crnn_sed_64_240/extractor_240_64_.27-0.9446.hdf5" --out_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/"
#python main_crnn_sed_v3.py extract_features --tr_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_240/training.h5" --te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_240/testing.h5" --scaler_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/scalers/logmel/training128_240.scaler" --model_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/models/crnn_sed_128_240/extractor_240_128_.33-0.9427.hdf5" --out_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/"
#python main_crnn_sed_v3.py extract_features --tr_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel256_240/training.h5" --te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel256_240/testing.h5" --scaler_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/scalers/logmel/training256_240.scaler" --model_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/models/crnn_sed_256_240/extractor_240_256_.30-0.9429.hdf5" --out_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/"
#python main_crnn_sed_v3.py extract_features --tr_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_360/training.h5" --te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_360/testing.h5" --scaler_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/scalers/logmel/training128_360.scaler" --model_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/models/crnn_sed_128_360/extractor_360_128_.35-0.9441.hdf5" --out_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/"
#python main_crnn_sed_v3.py extract_features --tr_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_120/training.h5" --te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel128_120/testing.h5" --scaler_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/scalers/logmel/training128_120.scaler" --model_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/models/crnn_sed_128_120/extractor_120_128_.19-0.9409.hdf5" --out_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/"

#python main_crnn_sed_v3.py train_classif --tr_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_train_240_64.hdf5" --tr_hdf5_path_2="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_train_360_128.hdf5" --tr_hdf5_path_3="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_train_240_128.hdf5" --te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_test_240_64.hdf5" --te_hdf5_path_2="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_test_240_128.hdf5" --te_hdf5_path_3="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_test_120_128.hdf5" --out_model_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/classif_models/"
#python main_crnn_sed_v3.py recognize_trained_classif --feat_te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel64_240/testing.h5" --te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_test_240_64.hdf5" --te_hdf5_path_2="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_test_360_128.hdf5" --te_hdf5_path_3="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_test_240_128.hdf5" --model_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/classif_models/classif_.108-0.9451.hdf5" --out_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/preds/crnn_sed"
#python main_crnn_sed_v3.py get_stat --pred_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/preds/crnn_sed" --stat_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/stats/crnn_sed" --submission_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/submissions/crnn_sed"

python main_crnn_sed_v3.py train_classif --tr_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_train_240_64.hdf5" --tr_hdf5_path_2="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_train_360_128.hdf5" --te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_test_240_64.hdf5" --te_hdf5_path_2="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_test_240_128.hdf5" --out_model_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/classif_models/"
#python main_crnn_sed_v3.py recognize_trained_classif --feat_te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/packed_features/logmel64_240/testing.h5" --te_hdf5_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_test_240_64.hdf5" --te_hdf5_path_2="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/extracted_features/ensamble_features_test_360_128.hdf5" --model_path="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/classif_models/classif_.236-0.9441.hdf5" --out_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/preds/crnn_sed"
#python main_crnn_sed_v3.py get_stat --pred_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/preds/crnn_sed" --stat_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/stats/crnn_sed" --submission_dir="/home/usuaris/veu/tfgveu9/Task4/dcase2017_task4_cvssp/submissions/crnn_sed"
