# This script reads in a signal and produces a corrupted signal
# Plots both signals and the string so we can see it, and
# Saves the data as an array so other scripts can use them.

# This file takes in a read and prints out the timestamps for each base

import os
import torch
import ast
import matplotlib.pyplot as plt
import pandas as pd
from bonito import util
from bonito.reader import Reader, read_chunks

INPUT_TICKS_PER_OUTPUT_TICK = 6
##############################
######## MODEL SETUP #########
##############################

# Disabling the compiler because there are issues with mutating 
torch._dynamo.disable() 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.attention.flex_attention")


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

#### Name of file for output read requests ####
read_requests_name = "read_requests_list"


# Lines come from find_kmers and have the format
# read_num, base,num_bases,duration_viterbi,raw start idx,raw end idx,input_base_timestamps
# read_num will get that number read from the data_dir using the reader
reads_df = pd.read_csv(str(read_list_csv), converters={'input_base_timestamps': ast.literal_eval}) # Load in the reads we want into a dataframe
reads_df.sort_values(by=reads_df.columns[0], inplace=True) # Sort for reading efficiency
print(reads_df)

requests_output_df = reads_df.copy()

for col in ['clean_str', 'clean_str_decode', 'corrupt_str_extended', 'corrupt_str_extended_decode', 'corrupt_str_shortened', 'corrupt_str_shortened_decode', 'clean_data', 'corrupt_extended_data',  'corrupt_shortened_data']:
    requests_output_df[col] = None
    requests_output_df[col] = requests_output_df[col].astype('object')

reader = Reader(data_dir)

reads = reader.get_reads(
    data_dir, 
    do_trim=True,
    scaling_strategy=model.config.get("scaling"),
    norm_params=model.config.get("standardisation")
)

highest_read = int(reads_df['read_num'].max())

# For iterating through all lines in the read request
read_line = 0 # For moving through the list of kmers in this read we want

# Iterate through the reads in the POD5 file - the order should ascend with the read_line read_num
for num_read in range(1, highest_read + 1): # highest_read = 2 means run this twice

    if read_line >= len(reads_df):
            break
   
    # Grab the very first read
    read = next(reads)

    raw_stndrd_signal = read.signal
    chop_request = reads_df.iloc[read_line]

    while chop_request.loc["read_num"] == num_read: # Chop out all the sections we want from this read
        
        # Use find_kmers params to chop the data
        window_start_abs = chop_request.loc['raw start idx'] - INPUT_BEGIN_PADDING_LENGTH
        feature_length = chop_request.loc['duration_viterbi'] * INPUT_TICKS_PER_OUTPUT_TICK # Just the length of the homopolymer
        window_end_abs = chop_request.loc['raw start idx'] + feature_length + INPUT_END_PADDING_LENGTH
        length = window_end_abs - window_start_abs # Length of whole signal (in input ticks)

        # Chop out a window around the desired signal
        input = (torch.tensor(raw_stndrd_signal[window_start_abs : window_end_abs], dtype=model_dtype)
                 .view(1, 1, length).to("cuda" if torch.cuda.is_available() else "cpu"))

        
        ### Raw values from the CSV
        num_bases = chop_request.loc['num_bases']
        kmer_base_timestamps = chop_request.loc["input_base_timestamps"]

        second_to_last_hmer_tok_start = kmer_base_timestamps[num_bases - 2] - window_start_abs
        last_hmer_tok_start = kmer_base_timestamps[num_bases - 1] - window_start_abs
        next_tok_start = chop_request.loc['raw end idx'] - window_start_abs


        # Get the string so we can save and check it        
        with torch.no_grad():
            clean_output = bonito_model(input)

        v_path_clean = model.seqdist.viterbi(clean_output.to(dtype=torch.float32)) #  Only float32 supported
       
        alphabet = {1: 'A', 2: 'C', 3: 'G', 4: 'T'}

        ### Add bases for clean string ###
        v_path_clean_list = v_path_clean.flatten().tolist()
        str_clean = ""
        last_char = None
        
        for state in v_path_clean_list: 
            if state == 0:
                last_char = None
                continue
            
            char = alphabet[state]
            if char != last_char:
                str_clean += char
                last_char = char
    
        # Save clean data
        requests_output_df.at[read_line, 'clean_data'] = input
        requests_output_df.at[read_line, 'clean_str'] = str_clean
        requests_output_df.at[read_line, 'clean_str_decode'] = bonito_model.decode(clean_output[:, 0, :])
        
        # Do both a shorten run (1) and extend run (1) for each read request
        for extend in range(0, 2): 

            # Make a corrupt signal that pastes the token over last homopolymer or extends the second to last homopolymer token out
            corrupt_input = input.detach().clone()
            
            # pad = 12 # to fit sampling instead of previous <- last_hmer_tok_start - second_to_last_hmer_tok_start

            if extend: # Turn CGAAA*AA*TTC into CGAAAA*AA*TC
                section = input[0, 0, second_to_last_hmer_tok_start : next_tok_start]
                space_left = corrupt_input.shape[-1] - last_hmer_tok_start
                paste_len = min(len(section), space_left)
                corrupt_input[0, 0, last_hmer_tok_start : last_hmer_tok_start + paste_len] = section[:paste_len]
            else: # Shorten string - Turn CGAAA*A*ATTC to Turn CGAAA**ATTTC
                second_last_tok_len = last_hmer_tok_start - second_to_last_hmer_tok_start
                last_hmer_tok_len = next_tok_start - last_hmer_tok_start
                section_add_len = 12 - (max(second_last_tok_len, last_hmer_tok_len) % 12) # Whatever it takes to get to a multiple of 12
                section = input[0, 0, last_hmer_tok_start : next_tok_start + section_add_len] # Round it out to 12 and make sure it covers the token I guess
                
                space_left = corrupt_input.shape[-1] - second_to_last_hmer_tok_start # in case we are at the end of the signal
                paste_len = min(len(section), space_left)
                
                corrupt_input[0, 0, second_to_last_hmer_tok_start : second_to_last_hmer_tok_start + paste_len] = section[:paste_len]


            # # Cross-fade smoothing
            # fade_len = 3
            # if fade_len < shift_dist:
            #     alpha = torch.linspace(0, 1, fade_len, device=input.device)
            #     splice_point = last_hmer_tok_start if extend else second_to_last_hmer_tok_start
                
            #     left_edge = input[0, 0, splice_point - fade_len : splice_point]
            #     right_edge = corrupt_input[0, 0, splice_point]
            #     corrupt_input[0, 0, splice_point - fade_len : splice_point] = (1 - alpha) * left_edge + alpha * right_edge

            # Check signal
            plt.figure()
            plt.plot(input.cpu()[0, 0, :], label="Clean signal", color='blue')
            plt.plot(corrupt_input.cpu()[0, 0, :], label="Corrupted signal", color='red')
            plt.title('Read ' + str(num_read) + ' request ' + str(chop_request))
            plt.savefig(f"plots/read_{num_read} line {read_line} extend={extend}.png")
            plt.close()

            # Run model and get string for corrupt signal      
            with torch.no_grad():
                corrupt_output = bonito_model(corrupt_input)

            v_path_corrupt = model.seqdist.viterbi(corrupt_output.to(dtype=torch.float32))

            # Save string to inspect later
            v_path_corrupt_list = v_path_corrupt.flatten().tolist()
            str_corrupt = ""
            last_char = None
            
            for state in v_path_corrupt_list: 
                if state == 0:
                    last_char = None
                    continue
                
                char = alphabet[state]
                if char != last_char:
                    str_corrupt += char
                    last_char = char

            # Save string and signal data to the dataframe
            if extend:
                requests_output_df.at[read_line, 'corrupt_extended_data'] = corrupt_input
                requests_output_df.at[read_line, 'corrupt_str_extended'] = str_corrupt
                requests_output_df.at[read_line, 'corrupt_str_extended_decode'] = bonito_model.decode(corrupt_output[:, 0, :])
            else:
                requests_output_df.at[read_line, 'corrupt_shortened_data'] = corrupt_input
                requests_output_df.at[read_line, 'corrupt_str_shortened'] = str_corrupt
                requests_output_df.at[read_line, 'corrupt_str_shortened_decode'] = bonito_model.decode(corrupt_output[:, 0, :])

        # Move to the next read request
        read_line += 1
        
        if read_line >= len(reads_df):
            break

        chop_request = reads_df.iloc[read_line]

# Save the dataframe to a csv (for reading) and pickle file (for loading)
if not requests_output_df.empty:
    
    # Make and save dataframe for csv (exclude signal data)
    requests_output_df_csv = requests_output_df.copy()
    requests_output_df_csv = requests_output_df_csv.drop(columns=['clean_data', 'corrupt_extended_data', 'corrupt_shortened_data'])
    requests_output_df_csv.to_csv(read_requests_name + ".csv", index=False)
    
    # Save the full dataframe as a .pkl
    requests_output_df.to_pickle(read_requests_name + ".pkl")
    print(f"saved to {read_requests_name}.csv and .pkl")
else:
    print("Failed - reads list is empty")
