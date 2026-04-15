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

INPUT_TICKS_PER_OUTPUT_TICK = 6
# ##############################
# ######## MODEL SETUP #########
# ##############################

# # Disabling the compiler because there are issues with mutating 
# torch._dynamo.disable() 

# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.attention.flex_attention")


model_path = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0" # Need to find model path

bonito_model = util.load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

model = bonito_model._orig_mod # Use unoptimized model to avoid conflicts with dynamo
model_dtype = next(model.parameters()).dtype #torch.float16
# #print(model_dtype)


##############################
######## DATA LOADING ########
##############################

INPUT_BEGIN_PADDING_LENGTH = 80 # Input length before the homopolymer to start
INPUT_END_PADDING_LENGTH = 80 # Input length after the homopolymer ends

#### Load in data ####
data_dir = "../data/reads/" 

#### Load in csv ####
read_list_csv = "./kmers_reads.csv" # Change this to the name of the csv

#### Where you want it saved ####
read_requests_csv = "read_requests_list.csv"


# Lines come from find_kmers and have the format
# read_num, base,num_bases,duration_viterbi,raw start idx,raw end idx,input_base_timestamps
# read_num will get that number read from the data_dir using the reader
reads_df = pd.read_csv(str(read_list_csv), converters={'input_base_timestamps': literal_eval}) # Load in the reads we want into a dataframe
reads_df.sort_values(by=reads_df.columns[0], inplace=True) # Sort for reading efficiency
print(reads_df)

requests_output_df = reads_df.copy()

reader = Reader(data_dir)

reads = reader.get_reads(
    data_dir, 
    do_trim=True,
    scaling_strategy=model.config.get("scaling"),
    norm_params=model.config.get("standardisation")
)

highest_read = reads_df['read_num'].max()

# For iterating through all lines in the read request
read_line = 0 # For moving through the list of kmers in this read we want

# Iterate through the reads in the POD5 file - the order should ascend with the read_line read_num
for num_read in range(1, highest_read + 1): # highest_read = 2 means run this twice
   
    # Grab the very first read
    read = next(reads)

    raw_stndrd_signal = read.signal
    chop_request = reads_df.iloc[read_line]

    while chop_request.loc["read_num"] == num_read: # Chop out all the sections we want from this read


        # Use find_kmers params to chop the data
        window_start_abs = chop_request.loc['raw start idx'] - INPUT_BEGIN_PADDING_LENGTH
        length = chop_request.loc['duration_viterbi'] * INPUT_TICKS_PER_OUTPUT_TICK
        window_end_abs = chop_request.loc['raw start idx'] +  + INPUT_END_PADDING_LENGTH
        
        # Chop out a window around the desired signal
        input = (torch.tensor(raw_stndrd_signal[window_start_abs : window_end_abs], dtype=model_dtype)
                 .view(1, 1, length).to("cuda" if torch.cuda.is_available() else "cpu"))

        
### Raw values from the CSV
        num_bases = chop_request.loc['num_bases']
        kmer_base_timestamps = chop_request.loc["input_base_timestamps"]

        second_to_last_hmer_tok_start = kmer_base_timestamps[num_bases - 2] - window_start_abs
        last_hmer_tok_start = kmer_base_timestamps[num_bases - 1] - window_start_abs
# NEXT_TOK_START_ABS = 9936
###

# LAST_HMER_TOK_START = LAST_HMER_TOK_START_ABS - WINDOW_START_ABS
# # NEXT_TOK_START = NEXT_TOK_START_ABS - WINDOW_START_ABS
# SECOND_TO_LAST_HMER_TOK_START = SECOND_TO_LAST_HMER_TOK_START_ABS - WINDOW_START_ABS
# LENGTH = INPUT_LENGTH + INPUT_END_PADDING_LENGTH + INPUT_BEGIN_PADDING_LENGTH
# print(f"Length: {LENGTH} for read")

        shift_dist = last_hmer_tok_start - second_to_last_hmer_tok_start

        # Do both a shorten run (1) and extend run (1) for each read request
        for extend in range(0, 2): 

            # Make a corrupt signal that pastes the token over last homopolymer or extends the second to last homopolymer token out
            corrupt_input = input.detach().clone()


# Chop data 
# ##############
# # THIS IS WHERE THE FOUND KMER PARAMS COME IN
# ##############
# RAW_START = 9870
# INPUT_LENGTH = 11 * 6 # Note that the csv has duration in viterbi output ticks = 1/6 input duration

# WINDOW_START_ABS = (RAW_START - INPUT_BEGIN_PADDING_LENGTH) # Absolute index equal to 0 in the chopped string
        


            if extend: # Turn CGAAA*A*ATTC into CGAAAA*AA*ATT
                corrupt_input[0, 0, last_hmer_tok_start : length] = input[0, 0, second_to_last_hmer_tok_start : length - shift_dist]
            else: # Shorten string
                corrupt_input[0, 0, second_to_last_hmer_tok_start : length - shift_dist] = input[0, 0, last_hmer_tok_start : length]
                corrupt_input[0, 0, length - shift_dist : length] = 0.0


            # Cross-fade smoothing
            fade_len = 3
            if fade_len < shift_dist:
                alpha = torch.linspace(0, 1, fade_len, device=input.device)
                splice_point = last_hmer_tok_start if extend else second_to_last_hmer_tok_start
                
                left_edge = input[0, 0, splice_point - fade_len : splice_point]
                right_edge = corrupt_input[0, 0, splice_point]
                corrupt_input[0, 0, splice_point - fade_len : splice_point] = (1 - alpha) * left_edge + alpha * right_edge

            # Check signal
            plt.figure()
            plt.plot(input.cpu()[0, 0, :], label="Clean signal", color='blue')
            plt.plot(corrupt_input.cpu()[0, 0, :], label="Corrupted signal", color='red')
            plt.title('Read ' + str(num_read) + ' request ' + str(chop_request))
            plt.show()

            # Save this run's signal data to a new dataframe
            if requests_output_df.isna(requests_output_df.iloc[read_line]['clean_data']):
                requests_output_df.iloc[read_line]['clean_data'] = input
                requests_output_df.iloc[read_line]['clean_data'] = input
            if extend:
                requests_output_df.iloc[read_line]['corrupt_extended_data'] = corrupt_input
            else:
                requests_output_df.iloc[read_line]['corrupt_shortened_data'] = corrupt_input

torch.save(input, 'clean_tensor.pt')
torch.save(corrupt_input, 'corrupt_tensor.pt')
