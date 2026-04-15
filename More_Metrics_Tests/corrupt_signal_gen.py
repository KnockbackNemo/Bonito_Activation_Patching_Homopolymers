# This script reads in a signal and produces a corrupted signal
# Plots both signals and the string so we can see it, and
# Saves the data as an array so other scripts can use them.

# This file takes in a read and prints out the timestamps for each base

import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from bonito import util
from bonito.reader import Reader, read_chunks

# ##############################
# ######## MODEL SETUP #########
# ##############################

# # Disabling the compiler because there are issues with mutating 
# torch._dynamo.disable() 

# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.attention.flex_attention")


# model_path = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0" # Need to find model path

# bonito_model = util.load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

# model = bonito_model._orig_mod # Use unoptimized model to avoid conflicts with dynamo
# model_dtype = next(model.parameters()).dtype #torch.float16
# #print(model_dtype)


##############################
######## DATA LOADING ########
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

NUM_READ = 1 # Change this to get a different read in the POD5 file

for i in range(1,NUM_READ):
    first_read = next(reads)

raw_stndrd_signal = first_read.signal

# Chop data 
# ##############
# # THIS IS WHERE THE FOUND KMER PARAMS COME IN
# ##############
RAW_START = 9870
INPUT_LENGTH = 11 * 6 # Note that the csv has duration in viterbi output ticks = 1/6 input duration
INPUT_BEGIN_PADDING_LENGTH = 80 # Input length before the homopolymer to start
INPUT_END_PADDING_LENGTH = 80 # Input length after the homopolymer ends
WINDOW_START_ABS = (RAW_START - INPUT_BEGIN_PADDING_LENGTH) # Absolute index equal to 0 in the chopped string
EXTEND = 1 # 1 for extend by 1, 0 for shorten by 1

### Raw values from the CSV
SECOND_TO_LAST_HMER_TOK_START_ABS = 9888
LAST_HMER_TOK_START_ABS = 9912
# NEXT_TOK_START_ABS = 9936
###

LAST_HMER_TOK_START = LAST_HMER_TOK_START_ABS - WINDOW_START_ABS
# NEXT_TOK_START = NEXT_TOK_START_ABS - WINDOW_START_ABS
SECOND_TO_LAST_HMER_TOK_START = SECOND_TO_LAST_HMER_TOK_START_ABS - WINDOW_START_ABS
LENGTH = INPUT_LENGTH + INPUT_END_PADDING_LENGTH + INPUT_BEGIN_PADDING_LENGTH
print(f"Length: {LENGTH} for read")

# Chop out a window around the desired signal
raw_stndrd_signal = raw_stndrd_signal[RAW_START - INPUT_BEGIN_PADDING_LENGTH:RAW_START + INPUT_LENGTH + INPUT_END_PADDING_LENGTH]
input = torch.tensor(raw_stndrd_signal, dtype=model_dtype).view(1, 1, LENGTH).to("cuda" if torch.cuda.is_available() else "cpu")

# Make a corrupt signal that pastes the token over last homopolymer or extends the second to last homopolymer token out
corrupt_input = input.detach().clone()

shift_dist = LAST_HMER_TOK_START - SECOND_TO_LAST_HMER_TOK_START

if EXTEND: # Turn CGAAA*A*ATTC into CGAAAA*AA*ATT
    corrupt_input[0, 0, LAST_HMER_TOK_START : LENGTH] = input[0, 0, SECOND_TO_LAST_HMER_TOK_START : LENGTH - shift_dist]
else: # Shorten string
    corrupt_input[0, 0, SECOND_TO_LAST_HMER_TOK_START : LENGTH - shift_dist] = input[0, 0, LAST_HMER_TOK_START : LENGTH]
    corrupt_input[0, 0, LENGTH - shift_dist : LENGTH] = 0.0


# Cross-fade smoothing
fade_len = 3
if fade_len < shift_dist:
    alpha = torch.linspace(0, 1, fade_len, device=input.device)
    splice_point = LAST_HMER_TOK_START if EXTEND else SECOND_TO_LAST_HMER_TOK_START
    
    left_edge = input[0, 0, splice_point - fade_len : splice_point]
    right_edge = corrupt_input[0, 0, splice_point]
    corrupt_input[0, 0, splice_point - fade_len : splice_point] = (1 - alpha) * left_edge + alpha * right_edge

# Check signal
plt.figure()
plt.plot(input.cpu()[0, 0, :], label="Clean signal", color='blue')
plt.plot(corrupt_input.cpu()[0, 0, :], label="Corrupted signal", color='red')
plt.show()

torch.save(input, 'clean_tensor.pt')
torch.save(corrupt_input, 'corrupt_tensor.pt')
