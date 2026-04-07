# The goal of this file is to prove that activation can be done successfully by taking a simple string of bases, ex: AAAAACCCCC, amd turning in into 
# a similar but slightly different string, ex: AAAAAGGGGG. The reason why it should be slightly the same is so that simply overwriting the entire signal
# is not enough- GGGGGGGGGG is not a success. Also, the goal is to find parts of the model that relate to C, not to just show that overwriting does something
# (which is apparent). 
# I'm going to compare this with the basecaller output.sam and hope the output matches!
# Where this ended up - the beginning indexes and numbers don't quite line up. I think this is due to the offset, but I haven't checked to make sure yet
# If I wanted to check, I'd need to expand my signal being searched for a match, I think.
import torch
import os
import pod5
import matplotlib.pyplot as plt
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


#TODO: How does this find the model? where is it?
model_path = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0" # Need to find model path

bonito_model = util.load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu") # Will eventually run this on LRC machines- may have gpus?

model = NNsight(bonito_model._orig_mod) # Use unoptimized model to avoid conflicts with dynamo
model_dtype = next(model.parameters()).dtype #torch.float16
#print(model_dtype)


##############################
######## DATA LOADING ########
##############################

# Load in data
data_dir = "./data/reads/"

# with pod5.Reader(pod5_filepath) as reader:
#     read = next(reader.reads()) # Get just the first strand/read

#     raw_signal = read.signal # raw signal - 1D numpy in pA
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
start_idx = 50000 # Skip beginning noise
LENGTH = 1000
end_idx = start_idx + LENGTH
raw_stndrd_signal = first_read.signal
chopped_signal = raw_stndrd_signal[start_idx:end_idx]

corrupt_data_offset = int(LENGTH / 2)
chopped_corrupt_signal = raw_stndrd_signal[start_idx + corrupt_data_offset: end_idx + corrupt_data_offset]



### Looking at official signal ###
# 4. Chop it using Bonito's exact chunking logic
chunksize = model.config["basecaller"]["chunksize"]
overlap = model.config["basecaller"]["overlap"]

# Get the first chunk (this is the exact tensor the model sees in the CLI)
chunks = list(read_chunks(first_read, chunksize=chunksize, overlap=overlap))
official_tensor = chunks[0].signal

print(f"Official CLI Tensor Shape: {official_tensor.shape}")
print(f"Official CLI Data (First 20 points): {official_tensor[:20]}")


# Find where Bonito decided to make the cut
trim_index = int(first_read.offset )
print(f"Bonito's calculated trim index: {trim_index}")

# Slice your manual signal starting at that exact index!
my_matched_signal = raw_stndrd_signal[trim_index : trim_index + 20]
print(f"My Matched Signal: {my_matched_signal}")


clean_input = torch.tensor(chopped_signal, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)
corrupted_input = torch.tensor(chopped_corrupt_signal, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)

print("My signal: ", (raw_stndrd_signal)[:20])

import numpy as np

# Search the entire read for the first two numbers of the official tensor
# We use np.isclose to account for float32 rounding differences
first_val = official_tensor[0]
second_val = official_tensor[1]

# Find all indices where the full signal matches the first number
matches = np.where(np.isclose(raw_stndrd_signal, first_val, atol=1e-4))[0]

found = False
for idx in matches:
    # Check if the next number matches too, confirming we found the sequence
    if np.isclose(raw_stndrd_signal[idx + 1], second_val, atol=1e-4):
        print(f"BINGO! The official chunk actually starts at hidden index: {idx}")
        found = True

if not found:
    print("NO MATCH FOUND: Bonito is definitely applying chunk-level scaling math!")


# Let's see if the output suggests A->C and A->G homopolymers
with model.trace(clean_input):
    clean_activation = model.encoder.transformer_encoder[0].output[0].save()
    clean_output = model.output.save()

with model.trace(corrupted_input):
    corrupted_output = model.output.save()

with model.trace(corrupted_input):
    model.encoder.transformer_encoder[0].output[0] = clean_activation
    patched_output = model.output.save()


# print("Clean Activation: ", clean_activation)
# print("Clean Output: ", clean_output)
# print("Corrupted Output: ", corrupted_output)
# print("Patched Output: ", patched_output)



file_name, extension = os.path.splitext(__file__)

torch.set_printoptions(profile="full") # See whole output
with open(f"{file_name}_output.txt", "w") as f:
    print(f"Output shape: {clean_output.shape}", file=f)
    print(f"Clean Output: {clean_output}", file=f)
    print(f"Corrupted Output: {corrupted_output}", file=f)
    print(f"Patched Output: {patched_output}", file=f)


### Check the string output to make sure the basecalls at least make sense ###
# model_outputs = {"clean_output" : clean_output, "corrupted_output" : corrupted_output, "patched_output" : patched_output}
# paths = {}
# strings = {}
# for name, output in model_outputs.items():
#     paths[name] = model.seqdist.viterbi(output.to(dtype=torch.float32)).cpu().numpy()
#     strings[name] = model.seqdist.path_to_str(paths[name])

# path_clean = model.seqdist.viterbi(clean_output.to(dtype=torch.float32)) #  Only float32 supported
# path_corrupted = model.seqdist.viterbi(corrupted_output.to(dtype=torch.float32))
# path_patched = model.seqdist.viterbi(patched_output.to(dtype=torch.float32))

# path_clean_numpy = path_clean.cpu().numpy()
# path_corrupted_numpy = path_corrupted.cpu().numpy()
# path_patched_numpy = path_patched.cpu().numpy()

string_clean = bonito_model.decode(clean_output[:, 0, :]) # Need to convert to a numpy array in memory
string_corrupted = bonito_model.decode(corrupted_output[:, 0, :]) 
string_patched = bonito_model.decode(patched_output[:, 0, :]) 



### I want to see some output change!
### TODO Note that I'm not sure what exactly each part of the output means still.

# Let's just look at the mean transition scores for A, C, G, and T

# reshaped_output
# Reshape the output to (Timesteps, Batch, States, Transitions)
# output_tensor shape is [34, 1, 5120]
reshaped_clean_output = clean_output.view(-1, 1, 1024, 5)
reshaped_corrupted_output = corrupted_output.view(-1, 1, 1024, 5)
reshaped_patched_output = patched_output.view(-1, 1, 1024, 5)

# Example: Get the scores for all 1024 states at the 15th timestep
# Each row is [Blank, A, C, G, T]
mean_scores_clean = torch.mean(reshaped_clean_output, dim=2) # Squish all transitions together
mean_scores_corrupted = torch.mean(reshaped_corrupted_output, dim=2) # Squish all transitions together
mean_scores_patched = torch.mean(reshaped_patched_output, dim=2) # Squish all transitions together


with open(f"{file_name}_strings.txt", "w") as f:
    print(f"Clean output string: {string_clean}", file=f)
    print(f"Corrupted output string: {string_corrupted}", file=f)
    print(f"Patched output string: {string_patched}", file=f)

    print(f"Mean Clean Output: {mean_scores_clean}", file=f)
    print(f"Mean Corrupted Output: {mean_scores_corrupted}", file=f)
    print(f"Mean Patched Output: {mean_scores_patched}", file=f)



# If you want to see the score for transitioning to 'G' from all states:
# g_scores = timestep_15_scores[:, 3]
# clean_output_mean_scores = clean_output

## Patch each layer in turn and see what happens
