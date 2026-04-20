# The goal of this file is to run a test but with more metrics - and more inputs. Ideally, this will loop over multiple inputs in a folder and then print metrics for them.

import torch
import os
import gc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bonito import util
from bonito.reader import Reader, read_chunks
from nnsight import NNsight
from dataclasses import dataclass
from pathlib import Path


##############################
######## MODEL SETUP #########
##############################

# Disabling the compiler because there are issues with mutating 
torch._dynamo.disable() 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.attention.flex_attention")


model_path = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0" ### The specific model being used!

bonito_model = util.load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu") 

model = NNsight(bonito_model._orig_mod) # Use unoptimized model to avoid conflicts with dynamo
model_dtype = next(model.parameters()).dtype #torch.float16


##############################
########  FILE SETUP #########
##############################
file_name, extension = os.path.splitext(__file__)

file_name += "_" + "R0_C_First_String" # Extra name here

##############################
########## FUNCTIONS #########
##############################

# @dataclass
# class check_output_timestamps_obj:
#     timestamp: int
#     target_transition: any #TODO not sure what type this is
#     wrong_transition: any

def run_patching_sweep(model, source_input, target_input, check_output_timestamps_list, 
                       component="mlp", head_idx=None, metric_mode="recovery"):
    """
    component options: "mlp", "attn", "head"
    """

    # Each row is a patch (differ in layer #, component type, & timestep)
    results = []
    
    ######## Get baseline clean and corrupt outputs for comparison #######
    with model.trace(source_input):
        source_output_proxy = model.output.save()
    source_output = source_output_proxy.detach() #TODO What does proxy and detach do?

    with model.trace(target_input):
        target_output_proxy = model.output.save()
    target_output = target_output_proxy.detach()

    ## Calculate some stats
    global_mse_baseline_diff = torch.nn.functional.mse_loss(source_output.to(torch.float32), target_output.to(torch.float32)).item()

    ####### Run the actual patching ########################
    for layer_idx in range(NUM_T_LAYERS):
        layer = model.encoder.transformer_encoder[layer_idx]

        ### Get source activations ###
        with model.trace(source_input):
            if component == "mlp":
                src_act_proxy = layer.ff.fc2.output.save()
            elif component == "attn":
                src_act_proxy = layer.self_attn.output[0].save()
            elif component == "head":
                src_act_proxy = layer.self_attn.out_proj.input[0].save()

        src_act = src_act_proxy.detach()
        del src_act_proxy

        ### Patch target ###
        for time_offset, t in enumerate(range(PATCHING_SWEEP_WINDOW_START_IDX, PATCHING_SWEEP_WINDOW_END_IDX)):
            with model.trace(target_input):
                # Calculate safe slice bounds for all components
                start = max(0, t - 1)
                
                # Need sequence length for the upper bound
                seq_len, d_model=src_act.shape[-2:]
                assert seq_len != 1 and d_model != 1
                
                end = min(seq_len, t + 2)

                if component == "mlp":
                    target = layer.ff.fc2.output.clone() # Shape: [1, seq_len, 512]
                    assert target.numel() == seq_len * d_model
                    # target[0, t-1:t+2, :] = src_act[0, t-1:t+2, :]
                    target[0, start:end, :] = src_act[0, start:end, :]
                    layer.ff.fc2.output = target

                elif component == "attn":
                    target = layer.self_attn.output[0].clone() # Shape: [seq_len, 512]
                    assert target.numel() == seq_len * d_model
                    # target[t-1:t+2, :] = src_act[t-1:t+2, :]
                    target[start:end, :] = src_act[start:end, :]
                    layer.self_attn.output[0] = target
                
                elif component == "head":
                    target = layer.self_attn.out_proj.input[0] # Shape: [seq_len, 512]
                    assert target.numel() == seq_len * d_model
                    # seq_len, d_model = target_act.shape
                    # 8 heads -> 64 = size of each head
                    nhead, head_dim = 8, 64
                    target_reshaped = target.clone().reshape(-1, nhead, head_dim)
                    src_reshaped = src_act.reshape(-1, nhead, head_dim)
                    
                    target_reshaped[start:end, head_idx, :] = src_reshaped[start:end, head_idx, :]

                    patched = target_reshaped.reshape(seq_len, d_model)
                    
                    # Reshape and put back into activations
                    layer.self_attn.out_proj.input = patched 

                patched_scores_proxy = model.output.save()

            patched_scores = patched_scores_proxy.detach()

            ####### Calculate metrics ##########
            for timestep in check_output_timestamps_list:
                target_transition_idx = torch.argmax(source_output[timestep, 0, :]).item() # Note that "target" now means correct instead of "corrupted string"
                wrong_transition_idx = torch.argmax(target_output[timestep, 0, :]).item() 
                logit_diff_source = source_output[timestep,0,target_transition_idx] - source_output[timestep,0,wrong_transition_idx]
                logit_diff_target = target_output[timestep,0,target_transition_idx] - target_output[timestep,0,wrong_transition_idx]
                baseline_logit_diff = logit_diff_source - logit_diff_target


                ### Get additional metrics ###
                # Logits and strings
                target_logit = patched_scores[timestep, 0, target_transition_idx].item()
                wrong_logit = patched_scores[timestep, 0, wrong_transition_idx].item()
                actual_max_pred_idx = torch.argmax(patched_scores[timestep, 0, :]).item()
                target_string = transition_idx_to_str(target_transition_idx)
                wrong_string = transition_idx_to_str(wrong_transition_idx)
                actual_string = transition_idx_to_str(actual_max_pred_idx)

                # Posteriors and strings - this includes global most likely path calculations #TODO There is definitely a cleaner way to do this
                source_posteriors = bonito_model.seqdist.posteriors(source_output.to(torch.float32)) + 1e-8
                target_posteriors = bonito_model.seqdist.posteriors(target_output.to(torch.float32)) + 1e-8
                patched_posteriors = bonito_model.seqdist.posteriors(patched_scores.to(torch.float32)) + 1e-8
                posteriors_target_transition_idx = torch.argmax(source_posteriors[timestep, 0, :]).item()
                posteriors_wrong_transition_idx = torch.argmax(target_posteriors[timestep, 0, :]).item()
                posteriors_actual_transition_idx = torch.argmax(patched_posteriors[timestep, 0, :]).item()
                target_state_prob = patched_posteriors[timestep, 0, posteriors_target_transition_idx]
                wrong_state_prob = patched_posteriors[timestep, 0, posteriors_wrong_transition_idx]
                actual_state_prob = patched_posteriors[timestep, 0, posteriors_actual_transition_idx]
                posteriors_target_string = transition_idx_to_str(posteriors_target_transition_idx)
                posteriors_wrong_string = transition_idx_to_str(posteriors_wrong_transition_idx)
                posteriors_actual_string = transition_idx_to_str(posteriors_actual_transition_idx) 

                   
                
                # Logit calculations
                patched_logit_diff = patched_scores[timestep, 0, target_transition_idx] - patched_scores[timestep, 0, wrong_transition_idx]

                if baseline_logit_diff == 0:
                    logit_score = 0
                elif metric_mode == "recovery":
                    logit_score = 1 - (logit_diff_source - patched_logit_diff) / baseline_logit_diff # 1 means full recovery, 0 means no change
                    logit_score = logit_score.item()
                elif metric_mode == "degradation":
                    logit_score = (logit_diff_source - patched_logit_diff) / baseline_logit_diff # 1 means full degradation, 0 means no change
                    logit_score = logit_score.item()
             

                # Probability calculations (posteriors)
                
                patched_prob_diff = target_state_prob - wrong_state_prob
                prob_diff_source = source_posteriors[timestep, 0, posteriors_target_transition_idx] - source_posteriors[timestep, 0, posteriors_wrong_transition_idx]
                prob_diff_target = target_posteriors[timestep, 0, posteriors_target_transition_idx] - target_posteriors[timestep, 0, posteriors_wrong_transition_idx]
                baseline_prob_diff = prob_diff_source - prob_diff_target 
                
                if baseline_prob_diff == 0:
                    posteriors_score = 0
                elif metric_mode == "recovery":
                    posteriors_score = 1 - (prob_diff_source - patched_prob_diff) / baseline_prob_diff # 1 means full recovery, 0 means no change
                    posteriors_score = posteriors_score.item()
                elif metric_mode == "degradation":
                    posteriors_score = (prob_diff_source - patched_prob_diff) / baseline_prob_diff # 1 means full degradation, 0 means no change
                    posteriors_score = posteriors_score.item()


                # If the MSE is greater than clean vs corrupt or the target prediction for this particular timestep isn't correct, decode and save the string
                is_target_logit_argmax = actual_max_pred_idx == target_transition_idx
                is_target_prob_argmax = posteriors_actual_transition_idx == posteriors_target_transition_idx
                global_mse_diff = torch.nn.functional.mse_loss(source_output.to(torch.float32), patched_scores.to(torch.float32)).item()
                should_decode = not is_target_logit_argmax or not is_target_prob_argmax or (global_mse_diff > global_mse_baseline_diff)

                string = None
                if should_decode:
                    string = bonito_model.decode(patched_scores[:, 0, :])
                
                #### Save everything to a dictionary
                results.append({
                    "Layer": layer_idx,
                    "Time_Offset": time_offset,
                    "Target_Timestep": timestep,
                    "Component": component if head_idx is None else f"head_{head_idx}",
                    "Metric_Mode": metric_mode,
                    "Logit_Score": logit_score,
                    "Posteriors_Score": posteriors_score,
                    "Target_Transition_Prior": target_transition_idx,
                    "Target_String_Prior": target_string,
                    "Target_Logit": target_logit,
                    "Wrong_Transition_Prior": wrong_transition_idx,
                    "Wrong_String_Prior": wrong_string,
                    "Wrong_Logit": wrong_logit,
                    "Actual_Argmax_Prior": actual_max_pred_idx,
                    "Actual_String_Prior": actual_string,
                    "Actual_Argmax_Logit": patched_scores[timestep, 0, actual_max_pred_idx],
                    "Target_Transition_Post": posteriors_target_transition_idx,
                    "Target_String_Post": posteriors_target_string,
                    "Target_Prob": target_state_prob,
                    "Wrong_Transition_Post": posteriors_wrong_transition_idx,
                    "Wrong_String_Post": posteriors_wrong_string,
                    "Wrong_Prob": wrong_state_prob,
                    "Actual_Transition_Post": posteriors_actual_transition_idx,
                    "actual_String_Post": posteriors_actual_string,
                    "actual_Prob": actual_state_prob,
                    "Is_Target_Logit_Argmax": is_target_logit_argmax,
                    "Is_Target_Prob_Argmax": is_target_prob_argmax,
                    "Should_Decode": should_decode,
                    "Global_MSE_Change": global_mse_diff - global_mse_baseline_diff,
                    "New_String": string
                })

            del patched_scores_proxy, patched_scores

    return pd.DataFrame(results)

    
# Plot results
def plot_and_save_outputs(df, component="mlp"):
    
        # Plot and save each timestep individually
        for step in df['Target_Timestep'].unique():

            step_df = df[df['Target_Timestep'] == step]

            folder_path = f"patch_results/{file_name}"
            
            csv_filename = (f"{component}_data_step_{step}.csv")
            csv_full_path = os.path.join(folder_path, csv_filename) 

            os.makedirs(folder_path, exist_ok=True)

            # Logit heatmaps
            png_filename = (f"{component}_logit_heatmap_results_stp_{step}.png")
            png_full_path = os.path.join(folder_path, png_filename) 
            heatmap_matrix = step_df.pivot(index="Layer", columns="Time_Offset", values="Logit_Score")
            plt.figure()
            sns.heatmap(heatmap_matrix)
            plt.title(f"{component} logit heatmap results timestep {step}")
            plt.xlabel(f"Time ticks")
            plt.ylabel(f"Layer")
            plt.savefig(png_full_path)
            plt.close()

            # Probabilitiy heatmaps
            png_filename = (f"{component}_posteriors_heatmap_results_stp_{step}.png")
            png_full_path = os.path.join(folder_path, png_filename) 
            heatmap_matrix = step_df.pivot(index="Layer", columns="Time_Offset", values="Posteriors_Score")
            plt.figure()
            sns.heatmap(heatmap_matrix)
            plt.title(f"{component} posteriors heatmap results timestep {step}")
            plt.xlabel(f"Time ticks")
            plt.ylabel(f"Layer")
            plt.savefig(png_full_path)
            plt.close()
          
            step_df.to_csv(csv_full_path, index=False)

def transition_idx_to_str(transition_idx: int):
    alphabet = model.seqdist.alphabet
    n_states = len(alphabet)
    len_window = model.seqdist.state_len
    n_base = n_states - 1

    # Get the transition (which includes blanks)
    next_base = transition_idx % n_states
    new_idx = transition_idx // n_states

    # Work backwards to get the most to least recent base in the transition
    past_bases_rev = alphabet[next_base]
    # Use mod to get each state (decoded) and append to a string
    for i in range(len_window):
        prev_base = alphabet[(new_idx % n_base) + 1] # No blanks in the context
        new_idx = new_idx // n_base

        past_bases_rev += prev_base

    past_bases = past_bases_rev[::-1]

    return past_bases

##############################
######## DATA CREATION ####### #TODO: We will want to load in reads generated from somewhere else instead probably
##############################

# Load in data
data_dir = "../data/reads/"

reader = Reader(data_dir)

reads = reader.get_reads(
    data_dir, 
    do_trim=True,
    scaling_strategy=model.config.get("scaling"),
    norm_params=model.config.get("standardisation")
)

# Grab the very first read
first_read = next(reads)


# Chop data
start_idx = 54250 # Skip beginning noise
LENGTH = 250
end_idx = start_idx + LENGTH
raw_stndrd_signal = first_read.signal
chopped_signal = raw_stndrd_signal[start_idx:end_idx]

clean_signal = chopped_signal.copy()
def time_to_output_idx(x) -> int:
    return x // 6 # CNN has stride of 3, 2, and 2, linear upsample has scale factor of 2

def time_to_transformer_idx(x) -> int:
    return x // 12 # CNN has stride of 3, 2, and 2

# Make fake signal spike
REPLACE_START = 130
steal_base_len = 6 # Must be even
steal_base_spike_start = 100

part_for_spike = clean_signal[steal_base_spike_start:steal_base_spike_start + steal_base_len]

# Modify data to add a spike

corrupt_signal = clean_signal.copy()
corrupt_signal[REPLACE_START : REPLACE_START + steal_base_len] = part_for_spike


# Move signal to tensors and GPU
clean_input = torch.tensor(clean_signal, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)
corrupted_input = torch.tensor(corrupt_signal, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)
first_input = torch.tensor(chopped_signal, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)

# Save clean and corrupted outputs
with model.trace(clean_input):
    clean_output_proxy = model.output.save()
clean_output = clean_output_proxy.detach()
del clean_output_proxy

with model.trace(corrupted_input):
    corrupted_output_proxy = model.output.save()
corrupted_output = corrupted_output_proxy.detach()
del corrupted_output_proxy


string_clean = bonito_model.decode(clean_output[:, 0, :]) # Need to convert to a numpy array in memory
string_corrupted = bonito_model.decode(corrupted_output[:, 0, :]) 
# string_patched = bonito_model.decode(patched_output[:, 0, :]) 

print(f"Clean string: {string_clean}")
print(f"Corrupted string: {string_corrupted}")


# The homopolymer we want has Cs starting at 14, 15, 16, 20, and 22 with the next A starting at 24 and ending at 26
# Format of the scores is [timesteps, batch, transitions]
score_window_start_idx = 23 #REPLACE_START - 10 # I think this is the wrong index
score_window_end_idx = 25
# INDEX_C = 2
# INDEX_BLANK = 0


# Run sweep of all layers using a timestep of 1
NUM_T_LAYERS = 18
PATCHING_SWEEP_WINDOW_START_IDX = 8#time_to_transformer_idx(REPLACE_START - 30) # 30 timestamps before spike
PATCHING_SWEEP_WINDOW_END_IDX = 16#time_to_transformer_idx(LENGTH)#REPLACE_START + steal_base_len + 30) # 30 timestamps after end of spike

NUM_TIMESTEP_SWEEPS = int((PATCHING_SWEEP_WINDOW_END_IDX - PATCHING_SWEEP_WINDOW_START_IDX) / 1) # TODO Later: divide by timestep patch size


##############################
###### PATCHING SWEEP ########
##############################

timestamps_to_score = [22, 24]
# Call patching sweep function and get heatmap_data things to plot

# Denoising
mlp_recovery = run_patching_sweep(model, clean_input, corrupted_input, timestamps_to_score, component="mlp", metric_mode="recovery")
attn_recovery = run_patching_sweep(model, clean_input, corrupted_input, timestamps_to_score, component="attn", metric_mode="recovery")

head_recoveries = []
NUM_HEADS = 8
for h in range(NUM_HEADS):
    print(f"Sweeping head {h} denoising...")
    head_recoveries.append(run_patching_sweep(model, clean_input, corrupted_input, timestamps_to_score, component="head", head_idx=h, metric_mode="recovery"))
    plot_and_save_outputs(head_recoveries[h], component=f"head {h} denoising")

    
# Noising
mlp_degradation = run_patching_sweep(model, corrupted_input, clean_input, timestamps_to_score, component="mlp", metric_mode="degradation")
attn_degradation = run_patching_sweep(model, corrupted_input, clean_input, timestamps_to_score, component="attn", metric_mode="degradation")

head_degradations = []
NUM_HEADS = 8
for h in range(NUM_HEADS):
    print(f"Sweeping head {h} noising...")
    head_degradations.append(run_patching_sweep(model, corrupted_input, clean_input, timestamps_to_score, component="head", head_idx=h, metric_mode="degradation"))
    plot_and_save_outputs(head_degradations[h], component=f"head {h} noising")


# Plot denoising
plot_and_save_outputs(mlp_recovery, component="mlp_denoising")
plot_and_save_outputs(attn_recovery, component="attn_denoising")
## plot_and_save_outputs(head_recoveries, component="head_denoising")

# Plot noising
plot_and_save_outputs(mlp_degradation, component="mlp_noising")
plot_and_save_outputs(attn_degradation, component="attn_noising")
### plot_and_save_outputs(head_degradations, component="head_noising")

