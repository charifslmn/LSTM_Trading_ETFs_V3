#------------------------------------------------------------------------------------------------------------------------
#                                             ABOUT 

#       - all functions will be defined here, and imported into a notebook as needed
#       - functions that are currently being tested (ie. as of Aug 5 ensemble functions) will not be fully here (may be in part)
#       - Another Equations.py file is likely to be created later if needed 


#------------------------------------------------------------------------------------------------------------------------

import torch.nn.functional as F
import copy
import numpy as np
import pandas as pd
import pickle
import random
import numpy as np
import os
import itertools
from joblib import Parallel, delayed
from collections import defaultdict
import math
import torch.nn as nn


import torch

#/home/charifslmn/


# with open('/home/charifslmn/short_dfs.pkl', 'rb') as f:
#     loaded_dfs = pickle.load(f)

# with open("/home/charifslmn/lagged_cache.pkl", "rb") as f:
#     lagged_cache = pickle.load(f)


# with open('/Users/cs/Desktop/LSTM_ETF_V3/short_dfs.pkl', 'rb') as f:
#     loaded_dfs = pickle.load(f)

with open("/Users/cs/Desktop/LSTM_ETF_V3/lagged_cache.pkl", "rb") as f:
    lagged_cache = pickle.load(f)

#*#*#* CHANGED: Set torch.backends.cudnn.enabled = True to ensure cuDNN is enabled for NVIDIA GPU acceleration.
torch.backends.cudnn.enabled = True




### notice this eq below is also in the Equations_Ensembles_Dist file , must be on both or run into erorrs due to circular imports
def evaluate_binary_0_1_selective_ensemble(predicted_array_flat, actual_array_flat,do_print : bool):

    predicted_array_correction = []
    actual_array_correction = []
    actual_array_all = []
        
    for idx, (pred,act) in enumerate(zip(predicted_array_flat,actual_array_flat)):
        if not isinstance(pred, str) and not None :
            predicted_array_correction.append(pred)
            actual_array_correction.append(act)
        
        actual_array_all.append(act)

    # print(predicted_array_correction)
    # # print(predicted_array_correction)
    # # print(actual_array_correction)

    if not predicted_array_correction:      # if predicted_array_correction == [] or None:

        return {
        'accuracy': 'No Agreed Predictions',
        'precision_up': 'No Agreed Predictions',
        'recall_up': 'No Agreed Predictions',
        'precision_down': 'No Agreed Predictions',
        'recall_down': 'No Agreed Predictions',
    }
    
    else:
        # predicted_array_correction = [i for i in predicted_array_correction]
        # actual_array_correction = [i for i in actual_array_correction]

        # actual_array_all = [i for i in actual_array_all] ### FIX RECALL

        predicted_array_correction = np.array(predicted_array_correction)
        actual_array_correction = np.array(actual_array_correction)
        actual_array_all = np.array(actual_array_all) ### FIX RECALL

        pred_direction = (predicted_array_correction > 0.5).astype(int)
        actual_direction = (actual_array_correction > 0.5).astype(int)
        actual_all_direction = (actual_array_all > 0.5).astype(int) ### FIX RECALL


        correct = (pred_direction == actual_direction).astype(int)
        
        accuracy = correct.sum() / len(correct) * 100
        actual_ups = (actual_direction == 1)

        actual_all_ups = (actual_all_direction == 1) ### FIX RECALL

        predicted_ups = (pred_direction == 1)
        true_positives_up = (predicted_ups & actual_ups).sum()
        precision_up = true_positives_up / predicted_ups.sum() * 100 if predicted_ups.sum() > 0 else float('nan')
        recall_up = true_positives_up / actual_all_ups.sum() * 100 if actual_all_ups.sum() > 0 else float('nan')
        actual_downs = (actual_direction == 0)

        actual_all_downs = (actual_all_direction == 0) ### FIX RECALL

        predicted_downs = (pred_direction == 0)
        true_positives_down = (predicted_downs & actual_downs).sum()
        precision_down = true_positives_down / predicted_downs.sum() * 100 if predicted_downs.sum() > 0 else float('nan')
        recall_down = true_positives_down / actual_all_downs.sum() * 100 if actual_all_downs.sum() > 0 else float('nan')

        if actual_ups.sum() == 0 and predicted_ups.sum() == 0:
            precision_up = None
            recall_up = None

        if actual_ups.sum() == 0 and predicted_ups.sum() > 0:
            precision_up = 0
            recall_up = None      

        if actual_ups.sum() > 0 and predicted_ups.sum() == 0:
            precision_up = None
            recall_up = 0

            ####################################

        if actual_downs.sum() == 0 and predicted_downs.sum() == 0:
            precision_down = None
            recall_down = None

        if actual_downs.sum() == 0 and predicted_downs.sum() > 0:
            precision_down = 0
            recall_down = None
        
        if actual_downs.sum() > 0 and predicted_downs.sum() == 0:
            precision_down = None
            recall_down = 0


        # if do_print:
        #     print(f"Directional Accuracy: {accuracy:.2f}%")
        #     print(f'Up Precision: {precision_up:.2f}%')
        #     print(f'Up Recall:    {recall_up:.2f}%')
        #     print(f'Down Precision: {precision_down:.2f}%')
        #     print(f'Down Recall:    {recall_down:.2f}%')
        return {
            'accuracy': accuracy,
            'precision_up': precision_up,
            'recall_up': recall_up,
            'precision_down': precision_down,
            'recall_down': recall_down,
        }



def format_to_tensor(df_merged, lag_steps, target_col="predictor_value", date_col="predictor_pred_date"):
    tensor_formatted_data = []
    predictor_data_wti_vals = df_merged[target_col].to_numpy()
    prediction_dates = df_merged[date_col].to_numpy()
    lagged_cols = {}
    for lag in range(1, lag_steps + 1):
        cols = sorted([col for col in df_merged.columns if f't_minus{lag}_' in col])
        lagged_cols[lag] = cols
    for i in range(len(df_merged)):
        sample = []
        for lag in range(lag_steps, 0, -1):
            sample.append(df_merged.loc[i, lagged_cols[lag]].tolist())
        tensor_formatted_data.append(sample)
    return np.array(tensor_formatted_data), np.array(predictor_data_wti_vals), np.array(prediction_dates)

from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, use_bidirectional, use_dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.bi_dir = 2 if use_bidirectional else 1
        # Only use dropout if use_dropout is True AND num_stacked_layers > 1
        if use_dropout and num_stacked_layers > 1:
            dropout_rate = 0.25
        else:
            dropout_rate = 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            dropout=dropout_rate, bias=True, batch_first=True,
                            bidirectional=use_bidirectional)
        self.fc = nn.Linear(hidden_size * self.bi_dir, 1)
    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers * self.bi_dir, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_stacked_layers * self.bi_dir, batch_size, self.hidden_size, device=device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_one_epoch(model, train_loader, optimizer, loss_function):
    model.train()
    
    running_loss = 0.0
    total_loss = 0.0
    
    device = 'cpu'

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        output = model(x_batch)
        y_batch = y_batch.view(-1, 1) # change shape to (batch_size, 1) since output is (batch_size, 1)
        loss = loss_function(output, y_batch)
        loss_value = loss.item()
        running_loss += loss_value
        total_loss += loss_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # avg_epoch_loss = total_loss / len(train_loader)
    # train_losses.append(avg_epoch_loss)



#*#*#*#* #*#*#*#* #*#*#*#* #*#*#*#* #*#*#*#*.     NEW NEW NEW SEPT 7


def train_one_epoch_custom_loss_BCE_THRESH(model, train_loader, optimizer, 
                           balancing_Weight_factor , use_LOW_weights : bool):
    
    model.train()
    
    running_loss = 0.0
    total_loss = 0.0
    
    device = 'cpu'

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        output = model(x_batch)
        y_batch = y_batch.view(-1, 1) # change shape to (batch_size, 1) since output is (batch_size, 1)
        loss = custom_loss_BCE_THRESH_PENALIZATION(output, y_batch, balancing_Weight_factor , use_LOW_weights=use_LOW_weights)
        loss_value = loss.item()
        running_loss += loss_value
        total_loss += loss_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # avg_epoch_loss = total_loss / len(train_loader)
    # train_losses.append(avg_epoch_loss)

def train_one_epoch_custom_loss_BCE_THRESH_AND_SEVERITY(model, train_loader, train_loader_RAW_Y_vals, optimizer , 
                           balancing_Weight_factor , use_LOW_weights : bool):
    
    model.train()
    
    running_loss = 0.0
    total_loss = 0.0
    
    device = 'cpu'

    for batch_index, (batch, batch_RAW_Y_vals) in enumerate(zip(train_loader, train_loader_RAW_Y_vals)):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        x_batch_RAW, y_batch_RAW = batch_RAW_Y_vals[0].to(device), batch_RAW_Y_vals[1].to(device)
        output = model(x_batch)
        y_batch = y_batch.view(-1, 1) # change shape to (batch_size, 1) since output is (batch_size, 1)
        y_batch_RAW = y_batch_RAW.view(-1, 1)


        loss = custom_loss_BCE_THRESH_AND_SEVERITY_PENALIZATION(output, y_batch, ACTUALS_RAW=y_batch_RAW, balancing_Weight_factor=balancing_Weight_factor , 
                                                                use_LOW_weights=use_LOW_weights)
        loss_value = loss.item()
        running_loss += loss_value
        total_loss += loss_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # avg_epoch_loss = total_loss / len(train_loader)
    # train_losses.append(avg_epoch_loss)



def custom_loss_BCE_THRESH_PENALIZATION(preds, actuals, balancing_Weight_factor , use_LOW_weights : bool):
    probs = torch.sigmoid(preds)
    
    thresh_bucket_1 = (0.7, 0.8); thresh_bucket_1_factor = 1.3 if use_LOW_weights else 1.5
    thresh_bucket_2 = (0.8, 0.9); thresh_bucket_2_factor = 1.7 if use_LOW_weights else 2.0 
    thresh_bucket_3 = (0.9, 1.0); thresh_bucket_3_factor = 2.5 if use_LOW_weights else 3.0

    weights = []
    
    # Ensure actuals and probs are the same shape and flattened
    actuals_flat = actuals.view(-1)
    probs_flat = probs.view(-1)
    
    for a, p in zip(actuals_flat, probs_flat):
        a_val = a.item()  # Convert tensor to Python scalar
        p_val = p.item()
        
        if (a_val > 0.5) and (p_val >= 0.5):
            weights.append(1.0)
        elif (a_val > 0.5) and (p_val < 0.5):
            weights.append(balancing_Weight_factor)  
        elif (a_val < 0.5) and (p_val < 0.5):
            weights.append(1.0)
        elif (a_val < 0.5) and (p_val >= 0.5):
            if (thresh_bucket_1[0] <= p_val < thresh_bucket_1[1]):
                weights.append(thresh_bucket_1_factor)
            elif (thresh_bucket_2[0] <= p_val < thresh_bucket_2[1]):
                weights.append(thresh_bucket_2_factor)
            elif (thresh_bucket_3[0] <= p_val <= thresh_bucket_3[1]):
                weights.append(thresh_bucket_3_factor)
            else:
                weights.append(1.0) # Default weight if none of the above conditions are met --- should not happen, all combos are accounted for

    # Create weights tensor on the same device as preds
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=preds.device).detach()
    
    # Reshape weights to match the original shape if needed
    weights_tensor = weights_tensor.view_as(actuals)
    
    loss = F.binary_cross_entropy_with_logits(preds, actuals, weight=weights_tensor)
    
    return loss



def custom_loss_BCE_THRESH_AND_SEVERITY_PENALIZATION(preds, actuals, ACTUALS_RAW, balancing_Weight_factor , use_LOW_weights : bool):
    probs = torch.sigmoid(preds)
    
    thresh_bucket_1 = (0.7, 0.8); thresh_bucket_1_factor = 1.3 if use_LOW_weights else 1.5
    thresh_bucket_2 = (0.8, 0.9); thresh_bucket_2_factor = 1.7 if use_LOW_weights else 2.0
    thresh_bucket_3 = (0.9, 1.0); thresh_bucket_3_factor = 2.5 if use_LOW_weights else 3.0

    SEVERE_CASE_VAL = - 0.12 ### for HOD 
    SEVERE_CASE_FACTOR = 1.5 
    # SEVERE_CASE_FACTOR = 2.5 


    weights = []
    
    # Ensure actuals and probs are the same shape and flattened
    actuals_flat = actuals.view(-1)
    probs_flat = probs.view(-1)
    ACTUALS_RAW_flat = ACTUALS_RAW.view(-1)

    for a, a_raw, p in zip(actuals_flat, ACTUALS_RAW_flat, probs_flat):
        a_val = a.item()  # Convert tensor to Python scalar
        p_val = p.item()
        a_raw_val = a_raw.item()
        
        if (a_val > 0.5) and (p_val >= 0.5):
            weights.append(1.0)
        elif (a_val > 0.5) and (p_val < 0.5):
            weights.append(balancing_Weight_factor)  
        elif (a_val < 0.5) and (p_val < 0.5):
            weights.append(1.0)
        elif (a_val < 0.5) and (p_val >= 0.5):
            if (thresh_bucket_1[0] <= p_val < thresh_bucket_1[1]):
                weights.append(thresh_bucket_1_factor)
            elif (thresh_bucket_2[0] <= p_val < thresh_bucket_2[1]):
                weights.append(thresh_bucket_2_factor)
            elif (thresh_bucket_3[0] <= p_val <= thresh_bucket_3[1]) and (a_raw_val > SEVERE_CASE_VAL):
                weights.append(thresh_bucket_3_factor)
            elif (thresh_bucket_3[0] <= p_val <= thresh_bucket_3[1]) and (a_raw_val <= SEVERE_CASE_VAL):
                weights.append(thresh_bucket_3_factor * SEVERE_CASE_FACTOR)  # Increase weight for severe cases
            else:
                weights.append(1.0) # Default weight if none of the above conditions are met --- should not happen, all combos are accounted for

    # Create weights tensor on the same device as preds
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=preds.device).detach()
    
    # Reshape weights to match the original shape if needed
    weights_tensor = weights_tensor.view_as(actuals)
    
    loss = F.binary_cross_entropy_with_logits(preds, actuals, weight=weights_tensor)
    
    return loss


 #*#*#*#* #*#*#*#* #*#*#*#* #*#*#*#* #*#*#*#*.    NEW NEW NEW SEPT 7



def validate_one_epoch(model, val_loader, loss_function,val_losses):
    
    device = next(model.parameters()).device
    model.train(False)
    running_loss = 0.0
    for batch_index, batch in enumerate(val_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            output = model(x_batch)
            y_batch = y_batch.view(-1, 1)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
    avg_loss_across_batches = running_loss / len(val_loader)
    val_losses.append(avg_loss_across_batches)



import numpy as np

def evaluate_binary_0_1(predicted_array, actual_array,one_fold:bool ,do_print : bool):

    if not one_fold:
            
        predicted_array = [pred for fold in predicted_array for pred in fold]
        actual_array = [pred for fold in actual_array for pred in fold]

        predicted_array = np.array(predicted_array)
        actual_array = np.array(actual_array)

    
    pred_direction = (predicted_array > 0.5).astype(int)
    actual_direction = (actual_array > 0.5).astype(int)
    correct = (pred_direction == actual_direction).astype(int)
    
    accuracy = correct.sum() / len(correct) * 100
    actual_ups = (actual_direction == 1)
    predicted_ups = (pred_direction == 1)
    true_positives_up = (predicted_ups & actual_ups).sum()
    precision_up = true_positives_up / predicted_ups.sum() * 100 if predicted_ups.sum() > 0 else float('nan')
    recall_up = true_positives_up / actual_ups.sum() * 100 if actual_ups.sum() > 0 else float('nan')
    actual_downs = (actual_direction == 0)
    predicted_downs = (pred_direction == 0)
    true_positives_down = (predicted_downs & actual_downs).sum()
    precision_down = true_positives_down / predicted_downs.sum() * 100 if predicted_downs.sum() > 0 else float('nan')
    recall_down = true_positives_down / actual_downs.sum() * 100 if actual_downs.sum() > 0 else float('nan')

    if actual_ups.sum() == 0 and predicted_ups.sum() == 0:
        precision_up = None
        recall_up = None

    if actual_ups.sum() == 0 and predicted_ups.sum() > 0:
        precision_up = 0
        recall_up = None      

    if actual_ups.sum() > 0 and predicted_ups.sum() == 0:
        precision_up = None
        recall_up = 0

        ####################################

    if actual_downs.sum() == 0 and predicted_downs.sum() == 0:
        precision_down = None
        recall_down = None

    if actual_downs.sum() == 0 and predicted_downs.sum() > 0:
        precision_down = 0
        recall_down = None
    
    if actual_downs.sum() > 0 and predicted_downs.sum() == 0:
        precision_down = None
        recall_down = 0


    if do_print:
        print(f"Directional Accuracy: {accuracy:.2f}%")
        print(f'Up Precision: {precision_up:.2f}%')
        print(f'Up Recall:    {recall_up:.2f}%')
        print(f'Down Precision: {precision_down:.2f}%')
        print(f'Down Recall:    {recall_down:.2f}%')

    return {
        'accuracy': accuracy,
        'precision_up': precision_up,
        'recall_up': recall_up,
        'precision_down': precision_down,
        'recall_down': recall_down,
    }




def evaluate_signed_neg1_1(predicted_array, actual_array,do_print : bool ):
    T = len(actual_array)
    correct = [1 if predicted_array[i] * actual_array[i] > 0 else 0 for i in range(1, T)]
    accuracy = sum(correct) / len(correct) * 100
    actual_ups = actual_array > 0
    predicted_ups = predicted_array > 0
    true_positives_up = (predicted_ups & actual_ups).sum()
    precision_up = true_positives_up / predicted_ups.sum() * 100 if predicted_ups.sum() > 0 else float('nan')
    recall_up = true_positives_up / actual_ups.sum() * 100 if actual_ups.sum() > 0 else float('nan')
    actual_downs = actual_array < 0
    predicted_downs = predicted_array < 0
    true_positives_down = (predicted_downs & actual_downs).sum()
    precision_down = true_positives_down / predicted_downs.sum() * 100 if predicted_downs.sum() > 0 else float('nan')
    recall_down = true_positives_down / actual_downs.sum() * 100 if actual_downs.sum() > 0 else float('nan')


    if actual_ups.sum() == 0 and predicted_ups.sum() == 0:
        precision_up = None
        recall_up = None

    if actual_ups.sum() == 0 and predicted_ups.sum() > 0:
        precision_up = 0
        recall_up = None

    if actual_ups.sum() > 0 and predicted_ups.sum() == 0:
        precision_up = None
        recall_up = 0

        ####################################

    if actual_downs.sum() == 0 and predicted_downs.sum() == 0:
        precision_down = None
        recall_down = None

    if actual_downs.sum() == 0 and predicted_downs.sum() > 0:
        precision_down = 0
        recall_down = None
    
    if actual_downs.sum() > 0 and predicted_downs.sum() == 0:
        precision_down = None
        recall_down = 0


    if do_print:
        print(f"Directional Accuracy: {accuracy:.2f}%")
        print(f'Up Precision: {precision_up:.2f}%')
        print(f'Up Recall:    {recall_up:.2f}%')
        print(f'Down Precision: {precision_down:.2f}%')
        print(f'Down Recall:    {recall_down:.2f}%')
    return {
        'accuracy': accuracy,
        'precision_up': precision_up,
        'recall_up': recall_up,
        'precision_down': precision_down,
        'recall_down': recall_down, 
    }







def run_combo_V_4(INDEX, combo, total_offset , use_print_acc_vs_pred : bool , pred_threshold_sigmoid01_up_bool : bool , store_model_weights : bool):

    def GRIDSEARCH_FUNCTION_WITH_CV_forTS(


            use_USO_wticoncat_predictor_WEEKLY_END_MO : bool , use_UCO_wticoncat_predictor_WEEKLY_END_MO : bool ,
            use_HUC_wticoncat_predictor_WEEKLY_END_MO : bool , use_HOD_wticoncat_predictor_WEEKLY_END_MO : bool ,
            use_CRUD_wticoncat_predictor_WEEKLY_END_MO : bool , use_SCO_wticoncat_predictor_WEEKLY_END_MO : bool ,

        learning_rate: float, num_epochs: int,
        batch_size: int, use_bidirectional: bool,
        lag: int, input_size: int,
        hidden_size: int, num_layers: int,

        use_monthly_dfs_only: bool,
        
        use_binary_0_1_retRate: bool,

        use_binary_neg1_1: bool,
        use_ret_rate: bool,
        use_print_acc: bool,
        use_dropout: bool,
        # iter_per_valSET: int,
        use_class_weighting: bool,
        is_deterministic: bool,
        seed_num: int,

        use_existing_lagged_data : bool,
 
        use_dynamic_weights : bool ,

        use_binary_0_1_retRate_custom_neg : bool ,
        use_binary_0_1_retRate_custom_pos : bool ,
        binary_0_1_cutoff_ret_rate_percentage : float,  ### cutoff for the  use_binary_0_1_retRate_custom_pos ot use_binary_0_1_retRate_custom_neg


        POS_weight_multiplier : float , 
        use_rolling_fixed_train_size : bool , 
        
        use_existing_initial_weights : bool ,

        state_dict ,

                use_custom_loss_function_BCE_THRESH: bool, # NEW NEW NEW

                use_custom_loss_function_BCE_THRESH_AND_SEVERITY: bool, # NEW NEW NEW

                use_LOW_weights_for_BCE_custom_loss : bool, # NEW NEW NEW


        train_start_month , # "2005-02",
        val_start_month , # '2020-01' ,    # test_start_month = '2022-01' ; test_end_month = '2022-12' 
        val_end_month , # '2021-12' ,
        num_preds_per_fold  , # should be 8 folds in the algo

                pred_threshold_sigmoid01_up  = None  , # NEW NEW NEW

        combo_index=INDEX  ### THIS IS THE INDEX in the parent function , it is passed there and does not need ot be explicitly stated when calling the GS func via parent function
    ):


######### --------------------------------------- DATA IMPORT ---------------------------------------

        if use_existing_lagged_data: 
            if use_USO_wticoncat_predictor_WEEKLY_END_MO:
                cache = lagged_cache[f"outer_lag_{lag}"]['USO_wticoncat_predictor_WEEKLY_END_MO'][f'lag_{lag}']
                df_lagged_PREDICTOR_short_EXPLANATORY = cache["df_lagged_USO_wticoncat_predictor_WEEKLY_END_MO_short_EXPLANATORY"]
            if use_UCO_wticoncat_predictor_WEEKLY_END_MO:
                cache = lagged_cache[f"outer_lag_{lag}"]['UCO_wticoncat_predictor_WEEKLY_END_MO'][f'lag_{lag}']
                df_lagged_PREDICTOR_short_EXPLANATORY = cache["df_lagged_UCO_wticoncat_predictor_WEEKLY_END_MO_short_EXPLANATORY"]
            if use_HUC_wticoncat_predictor_WEEKLY_END_MO:
                cache = lagged_cache[f"outer_lag_{lag}"]['HUC_wticoncat_predictor_WEEKLY_END_MO'][f'lag_{lag}']
                df_lagged_PREDICTOR_short_EXPLANATORY = cache["df_lagged_HUC_wticoncat_predictor_WEEKLY_END_MO_short_EXPLANATORY"]
            if use_HOD_wticoncat_predictor_WEEKLY_END_MO:
                cache = lagged_cache[f"outer_lag_{lag}"]['HOD_wticoncat_predictor_WEEKLY_END_MO'][f'lag_{lag}']
                df_lagged_PREDICTOR_short_EXPLANATORY = cache["df_lagged_HOD_wticoncat_predictor_WEEKLY_END_MO_short_EXPLANATORY"]
            if use_CRUD_wticoncat_predictor_WEEKLY_END_MO:
                cache = lagged_cache[f"outer_lag_{lag}"]['CRUD_wticoncat_predictor_WEEKLY_END_MO'][f'lag_{lag}']
                df_lagged_PREDICTOR_short_EXPLANATORY = cache["df_lagged_CRUD_wticoncat_predictor_WEEKLY_END_MO_short_EXPLANATORY"]
            if use_SCO_wticoncat_predictor_WEEKLY_END_MO:
                cache = lagged_cache[f"outer_lag_{lag}"]['SCO_wticoncat_predictor_WEEKLY_END_MO'][f'lag_{lag}']
                df_lagged_PREDICTOR_short_EXPLANATORY = cache["df_lagged_SCO_wticoncat_predictor_WEEKLY_END_MO_short_EXPLANATORY"]




            ###### _____________ FROM PREV VERSION 
            #### MONTLY 
            df_lagged_US_energy_PPI = cache["df_lagged_US_energy_PPI"] ; df_lagged_EU28_PPI = cache["df_lagged_EU28_PPI"]
            df_lagged_US_PMI = cache["df_lagged_US_PMI"] ; df_lagged_oecd_pet_stocks = cache["df_lagged_oecd_pet_stocks"]
            #### MONTLY 
            #### WEEKLY
            df_lagged_oecd_stocks_oilSPR_wkly = cache["df_lagged_oecd_stocks_oilSPR_wkly"]
            df_lagged_oecd_stocks_oilnonSPR_wkly = cache["df_lagged_oecd_stocks_oilnonSPR_wkly"] ; df_lagged_spec = cache["df_lagged_spec"]
            df_lagged_wklyUSdollarIndex = cache["df_lagged_wklyUSdollarIndex"] ; df_lagged_futures_3m_copper_weekly = cache["df_lagged_futures_3m_copper_weekly"]
            df_lagged_wti_crack_321 = cache["df_lagged_wti_crack_321"] ; df_lagged_brent_crack_321 = cache["df_lagged_brent_crack_321"]

            # df_lagged_wti_weekly_y_short_EXPLANATORY = cache["df_lagged_wti_weekly_y_short_EXPLANATORY"]

            #### WEEKLY
            ###### _____________ FROM PREV VERSION  
            #**#*#*# LAGGED DFS FOR WEEKLY DFS TURNED INTO MONTHLY DFS ##### MONTHLY MONTHLY 

            # df_lagged_wti_monthly_y_short_EXPLANATORY = cache["df_lagged_wti_monthly_y_short_EXPLANATORY"] 
            
            df_lagged_oecd_stocks_oilSPR_monthly_wkTOmo = cache["df_lagged_oecd_stocks_oilSPR_monthly_wkTOmo"]
            df_lagged_oecd_stocks_oilnonSPR_monthly_wkTOmo = cache["df_lagged_oecd_stocks_oilnonSPR_monthly_wkTOmo"] ; df_lagged_spec_monthly_wkTOmo = cache["df_lagged_spec_monthly_wkTOmo"]
            df_lagged_wklyUSdollarIndex_monthly_wkTOmo = cache["df_lagged_wklyUSdollarIndex_monthly_wkTOmo"] ; df_lagged_futures_3m_copper_monthly_wkTOmo = cache["df_lagged_futures_3m_copper_monthly_wkTOmo"]
            df_lagged_wti_crack_321_monthly_wkTOmo = cache["df_lagged_wti_crack_321_monthly_wkTOmo"] ; df_lagged_brent_crack_321_monthly_wkTOmo = cache["df_lagged_brent_crack_321_monthly_wkTOmo"]
            #**#*#*# LAGGED DFS FOR WEEKLY DFS TURNED INTO MONTHLY DFS ##### MONTHLY MONTHLY 


        lagged_dfs_monthly_only= [  #### MONTLY ONLY 
            df_lagged_US_energy_PPI[0],df_lagged_EU28_PPI[0],df_lagged_US_PMI[0],
            df_lagged_oecd_pet_stocks[0], 
            
            df_lagged_PREDICTOR_short_EXPLANATORY[0],
            
            df_lagged_oecd_stocks_oilSPR_monthly_wkTOmo[0],
            df_lagged_oecd_stocks_oilnonSPR_monthly_wkTOmo[0],df_lagged_spec_monthly_wkTOmo[0],df_lagged_wklyUSdollarIndex_monthly_wkTOmo[0],
            df_lagged_futures_3m_copper_monthly_wkTOmo[0],df_lagged_wti_crack_321_monthly_wkTOmo[0],df_lagged_brent_crack_321_monthly_wkTOmo[0],
        ]

        lagged_dfs_monthly_weekly = [ ### WEEKLY AND MONTHLY 
            df_lagged_US_energy_PPI[0],df_lagged_EU28_PPI[0],df_lagged_US_PMI[0],
            df_lagged_oecd_pet_stocks[0],df_lagged_oecd_stocks_oilSPR_wkly[0],df_lagged_oecd_stocks_oilnonSPR_wkly[0],df_lagged_spec[0],
            df_lagged_wklyUSdollarIndex[0],df_lagged_futures_3m_copper_weekly[0],df_lagged_wti_crack_321[0],df_lagged_brent_crack_321[0] ]
            
            # ,df_lagged_wti_monthly_y_short_EXPLANATORY[0] , 
             
            # df_lagged_wti_weekly_y_short_EXPLANATORY[0] ] 

        lagged_df = lagged_dfs_monthly_only if use_monthly_dfs_only else lagged_dfs_monthly_weekly #*#* choose df based on input to function 

        ###### MONTHLY ONLY MERGE
        df_merged = lagged_df[0].copy() # Use a copy to avoid modifying the original (pickled) DataFrame

       #   IMPORTANT IMPORTANT                       --- Use copies of each DataFrame to avoid in-place modification of pickled data---                    ********* IMPORTANT IMPORTANT IMPORTANT 
        lagged_df_copies = [df.copy() for df in lagged_df[1:]]

        for df in lagged_df_copies:
            if 'predictor_value' in df.columns:
                df.drop(columns='predictor_value', inplace=True)

        for df in lagged_df_copies:
            df_merged = pd.merge(df_merged, df, on='predictor_pred_date', how='inner')
        df_merged = df_merged[2::].reset_index(drop=True) #**#*# first two vals are nans  
        #   IMPORTANT IMPORTANT # Use copies of each DataFrame to avoid in-place modification of pickled data ********* IMPORTANT IMPORTANT IMPORTANT


        tensor_formatted_data_full , predictor_data_wti_vals_full , prediction_date_full =  format_to_tensor(df_merged, lag_steps=lag ,target_col="predictor_value", date_col="predictor_pred_date" )

        raw_actuals_full = predictor_data_wti_vals_full[:]  # same length as tensor_formatted_data

        if use_binary_0_1_retRate : 
            predictor_data_wti_vals_full = [1 if val > 0 else 0 for val in predictor_data_wti_vals_full]
        if use_binary_neg1_1 : 
            predictor_data_wti_vals_full = [1 if val > 0 else -1 for val in predictor_data_wti_vals_full]
        if use_ret_rate : 
            predictor_data_wti_vals_full = predictor_data_wti_vals_full
        if use_binary_0_1_retRate_custom_neg :             
            predictor_data_wti_vals_full = [1 if val < - binary_0_1_cutoff_ret_rate_percentage else 0 for val in predictor_data_wti_vals_full] #  NEW
        if use_binary_0_1_retRate_custom_pos :
             predictor_data_wti_vals_full = [1 if val > binary_0_1_cutoff_ret_rate_percentage else 0 for val in predictor_data_wti_vals_full] #  NEW


######### --------------------------------------- DATA LOADERS  - TRAIN/TEST/VAL ---------------------------------------

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        from torch.utils.data import DataLoader

        prediction_date_full = pd.to_datetime(prediction_date_full)

        # train_start_month = '2005-02' 
        # val_start_month = '2020-01' ; val_end_month = '2021-12'   # test_start_month = '2022-01' ; test_end_month = '2022-12' 

        idx_train_start_month = [idx for idx, date in enumerate(prediction_date_full) if str(date)[:7] == train_start_month][0]  ### this should be the first date of the train set
        idx_val_start_month_full = [idx for idx, date in enumerate(prediction_date_full) if str(date)[:7] == val_start_month][0]  ### this hsould be the fist date of the val set

        tensor_formatted_data = tensor_formatted_data_full[idx_train_start_month:]
        predictor_data_wti_vals = predictor_data_wti_vals_full[idx_train_start_month:]
        prediction_date = prediction_date_full[idx_train_start_month:]
        raw_actuals = raw_actuals_full[idx_train_start_month:]

        # print(predictor_data_wti_vals)

        idx_val_start_month = [idx for idx, date in enumerate(prediction_date) if str(date)[:7] == val_start_month][0]  ### this hsould be the fist date of the val set
        idx_val_end_month   = [idx for idx, date in enumerate(prediction_date) if str(date)[:7] == val_end_month][0]

        # num_preds_per_fold = 3  
        num_folds = ((idx_val_end_month - idx_val_start_month) + 1) // num_preds_per_fold  

        cut_idxs = [idx_val_start_month + i * num_preds_per_fold for i in range(num_folds + 1)]

        raw_actuals_val_span = raw_actuals[idx_val_start_month:idx_val_end_month]
        raw_actuals_per_fold = [raw_actuals[cut_idxs[i]:cut_idxs[i+1]] for i in range(num_folds)]

        if use_rolling_fixed_train_size:
            units_per_val_set = num_preds_per_fold
        else:
            units_per_val_set = 0

        train_loader_LIST, X_vals_LIST, Y_vals_LIST = [], [], []
        train_loader_LIST_RAW_Y_VALS = [] 
        Y_vals_dates_LIST = []

        for i in range(num_folds):
            X_train = torch.tensor(tensor_formatted_data[      i   *   units_per_val_set      :  cut_idxs[i]]   ).float()
            Y_train = torch.tensor(predictor_data_wti_vals[      i   *   units_per_val_set      :  cut_idxs[i]]  ).float()

            Y_TRAIN_RAW = torch.tensor(raw_actuals[    i   *   units_per_val_set      :  cut_idxs[i]]  ).float()  ### NEW NEW NEW

            X_val = torch.tensor(tensor_formatted_data[cut_idxs[i]:cut_idxs[i+1]]).float()
            Y_val = torch.tensor(predictor_data_wti_vals[cut_idxs[i]:cut_idxs[i+1]]).float()
            Y_val_dates = prediction_date[cut_idxs[i]:cut_idxs[i+1]] ### NEW NEW
            train_loader_LIST.append(DataLoader(TimeSeriesDataset(X_train, Y_train), batch_size=batch_size, shuffle=False))
            train_loader_LIST_RAW_Y_VALS.append(DataLoader(TimeSeriesDataset(X_train, Y_TRAIN_RAW), batch_size=batch_size, shuffle=False))  ### NEW NEW NEW
            
            X_vals_LIST.append(X_val)
            Y_vals_LIST.append(Y_val)
            Y_vals_dates_LIST.append(Y_val_dates) ### NEW NEW

    # GRIDSEARCH_FUNCTION_WITH_CV_forTS(**combo)




######### --------------------------------------- MODEL TRAINING AND EVALUATION  ---------------------------------------
 

        from collections import defaultdict

        cv_data = {}  
               
        model_weight_dict = {f"combo_number{total_offset + INDEX + 1}": {"initial": {}, "final": {}}}  # Initialize the model weight dictionary for the first combo


        all_preds = []
        all_actuals = []
        
        # === Loop over each CV set ===
        for set_idx, (train_loader_RAW_Y_vals ,train_loader, X_val, Y_val) in enumerate(zip(train_loader_LIST_RAW_Y_VALS,train_loader_LIST, X_vals_LIST, Y_vals_LIST)):


            if is_deterministic:  # --- DETERMINISM BLOCK  ---    ###### NOTE NOTE NOTE thsi must be insdie the loop !!! not outsdie or sle it gets diff weights each time from the RNG
                # SEED = 42
                SEED = seed_num
                random.seed(SEED)
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                torch.cuda.manual_seed_all(SEED)
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                torch.backends.cudnn.enabled = True
                # --- DETERMINISM BLOCK  ---
                # print(f"Running combo {combo_index} for set {set_idx + 1}" , flush=True) 


            if use_class_weighting and use_binary_0_1_retRate and use_dynamic_weights: 
                y_train_np = torch.cat([y for _, y in train_loader], dim=0).numpy()
                num_pos = (y_train_np > 0.5).sum()
                num_neg = (y_train_np <= 0.5).sum()
                pos_weight_value = (num_neg / num_pos) * POS_weight_multiplier if num_pos > 0 else 1.0
                pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)


            elif use_class_weighting and (use_binary_neg1_1 or use_ret_rate) and use_dynamic_weights:
                y_train_np = torch.cat([y for _, y in train_loader], dim=0).numpy()
                num_pos = (y_train_np > 0).sum()
                num_neg = (y_train_np <= 0).sum()
                pos_weight_value = (num_neg / num_pos) * POS_weight_multiplier if num_pos > 0 else 1.0
                pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)



            elif use_binary_0_1_retRate_custom_neg and (use_custom_loss_function_BCE_THRESH or use_custom_loss_function_BCE_THRESH_AND_SEVERITY or use_LOW_weights_for_BCE_custom_loss):

                length = idx_val_start_month_full -1

                num_1 = (raw_actuals_full[:length]    < - binary_0_1_cutoff_ret_rate_percentage).astype(int).sum()
                num_0 =  (raw_actuals_full[:length]   > - binary_0_1_cutoff_ret_rate_percentage).astype(int).sum()

                pos_weight_value = (num_0/ num_1)   # This will upweight the positive class
                pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
                # print('f', pos_weight , flush=True)



            elif use_binary_0_1_retRate_custom_pos and (use_custom_loss_function_BCE_THRESH or use_custom_loss_function_BCE_THRESH_AND_SEVERITY or use_LOW_weights_for_BCE_custom_loss):

                length = idx_val_start_month_full -1   

                num_1 = (raw_actuals_full[:length]    >  binary_0_1_cutoff_ret_rate_percentage).astype(int).sum()
                num_0 =  (raw_actuals_full[:length]   <  binary_0_1_cutoff_ret_rate_percentage).astype(int).sum()

                pos_weight_value = (num_0/ num_1)   # This will upweight the positive class
                pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
                # print('v', pos_weight , flush=True)


            elif use_class_weighting and use_binary_0_1_retRate_custom_pos:

                length = idx_val_start_month_full -1

                num_1 = (raw_actuals_full[:length]    >  binary_0_1_cutoff_ret_rate_percentage).astype(int).sum()
                num_0 =  (raw_actuals_full[:length]   <  binary_0_1_cutoff_ret_rate_percentage).astype(int).sum()

                pos_weight_value = (num_0/ num_1)  * POS_weight_multiplier  # This will upweight the positive class
                pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
                # print('a', pos_weight , flush=True) 


            elif use_class_weighting and use_binary_0_1_retRate_custom_neg:

                length = idx_val_start_month_full -1

                num_1 = (raw_actuals_full[:length] < -  binary_0_1_cutoff_ret_rate_percentage).astype(int).sum()
                num_0 = (raw_actuals_full[:length]  >  -  binary_0_1_cutoff_ret_rate_percentage).astype(int).sum()

                pos_weight_value = (num_0 / num_1) * POS_weight_multiplier  #CHCH  # 
                pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
                # print('b', pos_weight , flush=True) 

            else:
                pos_weight = None

            # print('c', pos_weight , flush=True)



##############################################################################################################################

        # for i in range(iter_per_valSET):
            # Check if multiple NVIDIA GPUs are available and use DataParallel if so
            model = LSTM(input_size, hidden_size, num_layers, use_bidirectional, use_dropout)  # create model instance
                                                    ###NEW
            if use_existing_initial_weights:

                model.load_state_dict(state_dict)
                                                            ###NEW
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:  # check for multiple NVIDIA GPUs
                model = torch.nn.DataParallel(model)  # wrap model to use all available GPUs
                print(f"Using {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}") #*#*#* CHANGED
            model = model.to(device)  # move model to the selected device (GPU or CPU)


            ##### TESTING: print model architecture                  ##### TESTING: print model architecture
            if store_model_weights:

                if set_idx == 0:

                    model_weight_dict[f"combo_number{total_offset + INDEX + 1}"]["initial"][f"set_{set_idx + 1}"] = copy.deepcopy(model.state_dict())   # NOTE NTOE NOTE MUST USE DDEPCOPY HERE OR ELS EHT DATA CHANGED 
                            
            ##### TESTING: print model architecture                ##### TESTING: print model architecture



            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # --- Use weighted loss if requested and binary classification ---
            if use_binary_0_1_retRate or use_binary_0_1_retRate_custom_pos or use_binary_0_1_retRate_custom_neg:
                if use_class_weighting and pos_weight is not None:
                    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                else:
                    loss_function = nn.BCEWithLogitsLoss()
            else:
                loss_function = nn.MSELoss()

            # train_losses = []
            # val_losses = []

            if (not use_custom_loss_function_BCE_THRESH) and (not use_custom_loss_function_BCE_THRESH_AND_SEVERITY):

                for epoch in range(num_epochs):
                    train_one_epoch(model, train_loader, optimizer, loss_function   )

            if use_custom_loss_function_BCE_THRESH:

                for epoch in range(num_epochs):
                    train_one_epoch_custom_loss_BCE_THRESH(model, train_loader, optimizer , balancing_Weight_factor = pos_weight_value ,use_LOW_weights = use_LOW_weights_for_BCE_custom_loss  )

            if use_custom_loss_function_BCE_THRESH_AND_SEVERITY:

                for epoch in range(num_epochs):
                    train_one_epoch_custom_loss_BCE_THRESH_AND_SEVERITY(model, train_loader, train_loader_RAW_Y_vals, optimizer , balancing_Weight_factor = pos_weight_value ,use_LOW_weights = use_LOW_weights_for_BCE_custom_loss  )

            ##### TESTING: print model architecture            ##### TESTING: print model architecture
            if store_model_weights:
                    
                if set_idx == 0:
                    model_weight_dict[f"combo_number{total_offset + INDEX + 1}"]["final"][f"set_{set_idx + 1}"] = copy.deepcopy(model.state_dict())  # NOTE NTOE NOTE MUST USE DDEPCOPY HERE OR ELS EHT DATA CHA
            ##### TESTING: print model architecture            ##### TESTING: print model architecture


            with torch.no_grad():
                val_output = model(X_val.to(device))
                val_predictions = torch.sigmoid(val_output).detach().cpu().numpy().flatten() if (use_binary_0_1_retRate or use_binary_0_1_retRate_custom_neg or use_binary_0_1_retRate_custom_pos) \
                                else val_output.detach().cpu().numpy().flatten()

            predicted_array = val_predictions
            actual_array = Y_val.numpy().flatten()


            all_preds.append(val_predictions)
            all_actuals.append(actual_array)

            # --- Plot actual vs predicted if true ---
            if use_print_acc_vs_pred:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 2))
                plt.axhline(0.5, color='red', linestyle='--', linewidth=1)
                plt.plot(actual_array, '.' , label='Actual' )
                plt.plot(predicted_array, '.' , label='Predicted')
                plt.title(f'Actual vs Predicted (Set {set_idx+1}, Iter {i+1})')
                plt.xlabel('Sample')
                plt.ylabel('Value')
                plt.legend()
                plt.tight_layout()
                plt.show()
                # --- Plot actual vs predicted if true ---


            if use_binary_0_1_retRate or use_binary_0_1_retRate_custom_neg or use_binary_0_1_retRate_custom_pos:
                metrics = evaluate_binary_0_1(predicted_array, actual_array, one_fold= True ,do_print=use_print_acc)
            elif use_binary_neg1_1 or use_ret_rate:
                metrics = evaluate_signed_neg1_1(predicted_array, actual_array, one_fold= True ,do_print=use_print_acc)


            cv_data[f"set_{set_idx + 1}"] = metrics


        # === Compute overall average ===
        metrics_keys = cv_data[f"set_1"].keys()

        # print(metrics_keys)

        overall_avg = {}
        for k in metrics_keys:
            values = [cv_data[f"set_{i + 1}"][k] for i in range(num_folds)]

            numeric_values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v)) and isinstance(v, (int, float))]
            if len(numeric_values) > 0:
                overall_avg[k] = np.mean(numeric_values)

            else:
                overall_avg[k] = None
            # else:
            #     # If all values are string messages, keep the message
            #     string_values = [v for v in values if isinstance(v, str)]
            #     overall_avg[k] = string_values[0] if string_values else np.nan
        cv_data["avg_across_all_sets"] = overall_avg
        cv_data["overall_metrics"] = evaluate_binary_0_1(all_preds , all_actuals ,one_fold= False ,do_print=False)


        ################ POS/NEG THRESHOLD TEST

        if  (pred_threshold_sigmoid01_up_bool) and (use_binary_0_1_retRate_custom_neg or use_binary_0_1_retRate_custom_pos):


            all_actuals_threshold_per_fold = [] 
            all_preds_threshold_per_fold = []
            for p_fold,a_fold in zip(all_preds , raw_actuals_per_fold):

                new_p_fold = []
                new_a_fold = []

                for p,a in zip(p_fold,a_fold):
                    if p > 0.5 and p > pred_threshold_sigmoid01_up:
                        new_p_fold.append(p)
                        new_a_fold.append(a)

                    if p > 0.5 and p < pred_threshold_sigmoid01_up:
                            new_p_fold.append('below_threshold')
                            new_a_fold.append('below_threshold')

                    else:
                        new_p_fold.append(p)
                        new_a_fold.append(a)


                all_preds_threshold_per_fold.append(new_p_fold)
                all_actuals_threshold_per_fold.append(new_a_fold)

        
            all_actuals_threshold_per_fold_flattened = [j for parts in all_actuals_threshold_per_fold for j in parts] 
            all_preds_threshold_per_fold_flattened = [j for parts in all_preds_threshold_per_fold for j in parts ]
            
            cv_data["overall_metrics_thresh"] = evaluate_binary_0_1_selective_ensemble(all_preds_threshold_per_fold_flattened , all_actuals_threshold_per_fold_flattened  ,do_print=False)
    



        ################ POS/NEG THRESHOLD TEST



        # Return with explicit branches (avoid ternary precedence issues)
        if pred_threshold_sigmoid01_up_bool:
            return (
                cv_data, model_weight_dict, all_preds, all_actuals, raw_actuals_per_fold,
                all_actuals_threshold_per_fold, all_preds_threshold_per_fold
            )
        else:
            return (
                cv_data, model_weight_dict, all_preds, all_actuals, raw_actuals_per_fold , Y_vals_dates_LIST
            )


        # return cv_data , model_weight_dict  , all_preds , all_actuals ,raw_actuals_per_fold , \
        #         all_actuals_threshold_per_fold , all_preds_threshold_per_fold if pred_threshold_sigmoid01_up_bool  \
        #         else cv_data , model_weight_dict  , all_preds , all_actuals ,raw_actuals_per_fold  \
    


    if  not pred_threshold_sigmoid01_up_bool:

        cv_data , model_weight_dict , all_preds , all_actuals , raw_actuals_per_fold , Y_vals_dates_LIST = GRIDSEARCH_FUNCTION_WITH_CV_forTS(**combo)

        result_entry = {
        "combo_number": total_offset + INDEX + 1,
        "parameters": combo,
        "cv_sets": cv_data,
        "all_preds" : all_preds ,
        "all_actuals" : all_actuals,
        "raw_actuals" : raw_actuals_per_fold ,
        "Y_vals_dates_LIST" : Y_vals_dates_LIST
    }

    else:
        cv_data , model_weight_dict , all_preds , all_actuals , raw_actuals_per_fold , all_actuals_threshold_per_fold , all_preds_threshold_per_fold = GRIDSEARCH_FUNCTION_WITH_CV_forTS(**combo)

        result_entry = {
        "combo_number": total_offset + INDEX + 1,
        "parameters": combo,
        "cv_sets": cv_data,
        "all_preds" : all_preds ,
        "all_actuals" : all_actuals,
        "raw_actuals" : raw_actuals_per_fold ,
        "all_actuals_threshold_per_fold" :all_actuals_threshold_per_fold , 
        "all_preds_threshold_per_fold" : all_preds_threshold_per_fold
    }

    print(
        f"--- Running Combo {total_offset + INDEX + 1} ---" 
        f"Parameters: {combo}\n" ,
        # f"→ Fold Accuracies: " +
        # "  ".join(
        #     f"{k}: {cv_data[k][0]['accuracy']}"  # <-- each set is now a list of metric dicts
        #     for k in sorted(cv_data) if k.startswith("set_")
        flush=True
    )

    if store_model_weights:
        return result_entry ,model_weight_dict
    else:
        return result_entry



