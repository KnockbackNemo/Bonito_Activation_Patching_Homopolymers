# The goal of this file is to run a test but with more metrics - and more inputs. Ideally, this will loop over multiple inputs in a folder and then print metrics for them.

import torch
import os
import gc
import pod5
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bonito import util
from bonito import util
from bonito.reader import Reader, read_chunks
from nnsight import NNsight


##############################
######## MODEL SETUP #########
##############################

# Disabling the compiler because there are issues with mutating 
torch._dynamo.disable() 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.attention.flex_attention")


model_path = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0"

bonito_model = util.load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu") 

model = NNsight(bonito_model._orig_mod) # Use unoptimized model to avoid conflicts with dynamo
model_dtype = next(model.parameters()).dtype #torch.float16


##############################
########## FUNCTIONS #########
##############################
def run_patching_sweep(model, source_input, target_input, component="mlp", head_idx=None, metric_mode="recovery"):
    """
    component options: "mlp", "attn", "head"
    """

    heatmap_data = [np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS)) for _ in range(3)]

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

        for time_offset, t in enumerate(range(SWEEP_WINDOW_START, SWEEP_WINDOW_END)):
            with model.trace(target_input):
                ############TODO Clean this up too
                # Calculate safe slice bounds for all components
                start = max(0, t - 1)
                
                # Need sequence length for the upper bound
                # Assuming src_act shape is [batch, seq_len, d_model] or [seq_len, d_model]
                if component in ["mlp", "attn"]:
                    seq_len = src_act.shape[1] if len(src_act.shape) == 3 else src_act.shape[0]
                else:
                    seq_len = layer.self_attn.out_proj.input[0].shape[0]
                    
                end = min(seq_len, t + 2)

                if component == "mlp":
                    target = layer.ff.fc2.output.clone()
                    target[0, t-1:t+2, :] = src_act[0, t-1:t+2, :]
                    layer.ff.fc2.output = target

                elif component == "attn":
                    target = layer.self_attn.output[0].clone()
                    target[t-1:t+2, :] = src_act[t-1:t+2, :]
                    layer.self_attn.output[0] = target
                
                elif component == "head":
                    # Shape is [seq_len, 512]
                    target_act = layer.self_attn.out_proj.input[0]

                    seq_len, d_model = target_act.shape
                    # 8 heads -> 64 = size of each head
                    nhead, head_dim = 8, 64
                    target_reshaped = target_act.clone().reshape(-1, nhead, head_dim)
                    src_reshaped = src_act.reshape(-1, nhead, head_dim)
                    
                    start = max(0, t - 1)
                    end = min(seq_len, t + 2)

                    target_reshaped[start:end, head_idx, :] = src_reshaped[start:end, head_idx, :]

                    patched = target_reshaped.reshape(seq_len, d_model)
                    
                    # Reshape and put back into activations
                    layer.self_attn.out_proj.input = patched 

                patched_scores_proxy = model.output.save()
            patched_scores = patched_scores_proxy.detach()

            ### Calculate metrics ###
            for i in range(len(TARGET_TRANSITIONS)):
                if baseline_diffs[i] == 0.0:
                    heatmap_data[i][layer_idx, time_offset] = 0.0
                    continue

                patched_diff = patched_scores[22+i, 0, TARGET_TRANSITIONS[i]] - patched_scores[22+i, 0, WRONG_TRANSITIONS[i]]

                if metric_mode == "recovery":
                    score = 1 - (logit_diffs_clean[i] - patched_diff) / baseline_diffs[i]
                elif metric_mode == "degradation":
                    score = (logit_diffs_clean[i] - patched_diff) / baseline_diffs[i]

                heatmap_data[i][layer_idx, time_offset] = score.item()

            del patched_scores_proxy, patched_scores

    return heatmap_data



# Plot results
def plot_and_save_outputs(heatmap_data, component="mlp", metric_mode="recovery"):
    
    if component=="head": #heatmap_data has NUM_HEADS groups of three
        for j in range(NUM_HEADS):
            for i in range(0, len(TARGET_TRANSITIONS)):
                plt.figure()
                sns.heatmap(heatmap_data[j][i])
                plt.savefig(f"{file_name}_{component}_{j}_heatmap_results_stp_{22 + i}.png")
                np.save(f"{file_name}_{component}_{j}_heatmap_data_{22 + i}.npy", heatmap_data[j][i])
        return
    
    else:
        for i in range(0, len(TARGET_TRANSITIONS)):
            plt.figure()
            sns.heatmap(heatmap_data[i])
            plt.savefig(f"{file_name}_{component}_heatmap_results_stp_{22 + i}.png")
            np.save(f"{file_name}_{component}_heatmap_data_{22 + i}.npy", heatmap_data[i])


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

# Calculate baseline MSE at the time
# The homopolymer we want has Cs starting at 14, 15, 16, 20, and 22 with the next A starting at 24 and ending at 26
# Format of the scores is [timesteps, batch, transitions]
score_window_start_idx = 23 #REPLACE_START - 10 # I think this is the wrong index
score_window_end_idx = 25
INDEX_C = 2
INDEX_BLANK = 0


# Run sweep of all layers using a timestep of 1
NUM_T_LAYERS = 18
SWEEP_WINDOW_START = 8#time_to_transformer_idx(REPLACE_START - 30) # 30 timestamps before spike
SWEEP_WINDOW_END = 16#time_to_transformer_idx(LENGTH)#REPLACE_START + steal_base_len + 30) # 30 timestamps after end of spike

NUM_TIMESTEP_SWEEPS = int((SWEEP_WINDOW_END - SWEEP_WINDOW_START) / 1) # TODO Later: divide by timestep patch size


# Metrics to look at:
# Correct transition vs wrong transition
# Sum of C states - sum of blank states
# later - Sum of A states?
#       - Sum of blank states?
###### Things because I can't think right now but I 100% need to fix this (hardcode 22-24)
TARGET_TRANSITIONS = [torch.argmax(clean_output[22, 0, :]).item(), torch.argmax(clean_output[23, 0, :]).item(), torch.argmax(clean_output[24, 0, :]).item()]

WRONG_TRANSITIONS = [torch.argmax(corrupted_output[22, 0, :]).item(), torch.argmax(corrupted_output[23, 0, :]).item(), torch.argmax(corrupted_output[24, 0, :]).item()]

# heatmap_data_sum_diff_23 = np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS)) # test all three
# heatmap_data_sum_diff_24 = np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS))
# heatmap_data_sum_diff_22 = np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS))
# heatmap_data_argmax_diffs = [np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS)), np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS)), np.zeros((NUM_T_LAYERS, NUM_TIMESTEP_SWEEPS))]

patching_strings = {}
######################################################################

# Logit difference timestep constants
logit_diffs_clean= []
logit_diffs_corrupt= []
for i in range(0, len(TARGET_TRANSITIONS)):
    logit_diffs_clean.append(clean_output[22 + i, 0, TARGET_TRANSITIONS[i]] - clean_output[22 + i, 0, WRONG_TRANSITIONS[i]])
    logit_diffs_corrupt.append(corrupted_output[22 + i, 0, TARGET_TRANSITIONS[i]] - corrupted_output[22 + i, 0, WRONG_TRANSITIONS[i]])

baseline_diffs = []
for i in range(0, len(TARGET_TRANSITIONS)): # Positive if clean is closer to the target than wrong, negative if corrupt is closer to target than wrong, 0 if same
    baseline_diffs.append(logit_diffs_clean[i].to(torch.float32) - logit_diffs_corrupt[i].to(torch.float32))
print(f"Output shape: {clean_output.shape}")
print(f"baseline diff: {baseline_diffs}")

##############################
###### PATCHING SWEEP ########
##############################

# Call patching sweep function and get heatmap_data things to plot

# Denoising
mlp_recovery = run_patching_sweep(model, clean_input, corrupted_input, component="mlp", metric_mode="recovery")
attn_recovery = run_patching_sweep(model, clean_input, corrupted_input, component="attn", metric_mode="recovery")

head_recoveries = []
NUM_HEADS = 8
for h in range(NUM_HEADS):
    print(f"Sweeping head {h} denoising...")
    head_recoveries.append(run_patching_sweep(model, clean_input, corrupted_input, component="head", head_idx=h, metric_mode="recovery")
)
    
# Noising
mlp_degradation = run_patching_sweep(model, corrupted_input, clean_input, component="mlp", metric_mode="degradation")
attn_degradation = run_patching_sweep(model, corrupted_input, clean_input, component="attn", metric_mode="degradation")

head_degradations = []
NUM_HEADS = 8
for h in range(NUM_HEADS):
    print(f"Sweeping head {h} noising...")
    head_degradations.append(run_patching_sweep(model, corrupted_input, clean_input, component="head", head_idx=h, metric_mode="degradation")
)

# Plot denoising
plot_and_save_outputs(mlp_recovery, component="mlp_denoising")
plot_and_save_outputs(attn_recovery, component="attn_denoising")
plot_and_save_outputs(head_recoveries, component="head_denoising")

# Plot noising
plot_and_save_outputs(mlp_degradation, component="mlp_noising")
plot_and_save_outputs(attn_degradation, component="attn_noising")
plot_and_save_outputs(head_degradations, component="head_noising")

# Write data to a file
file_name, extension = os.path.splitext(__file__)



string_clean = bonito_model.decode(clean_output[:, 0, :]) # Need to convert to a numpy array in memory
string_corrupted = bonito_model.decode(corrupted_output[:, 0, :]) 
# string_patched = bonito_model.decode(patched_output[:, 0, :]) 

print(f"Clean string: {string_clean}")
print(f"Corrupted string: {string_corrupted}")


# with open(f"{file_name}_strings.txt", "w") as f:
#     print(f"Clean output string: {string_clean}", file=f)
#     print(f"Corrupt output strn: {string_corrupted}", file=f)
#     print()

#     # Loop through heatmap and print each string
#     for layer_idx in range(NUM_T_LAYERS):
#         print(f"--- Patched Strings for Layer {layer_idx} ---", file=f)
#         for time_offset, t in enumerate(range(SWEEP_WINDOW_START, SWEEP_WINDOW_END)):
#             print(f"Time {time_offset} : {patching_strings[layer_idx, time_offset]} : scores: {heatmap_data_argmax_diffs[0][layer_idx, time_offset]}, {heatmap_data_argmax_diffs[1][layer_idx, time_offset]}, {heatmap_data_argmax_diffs[2][layer_idx, time_offset]}", file=f)

