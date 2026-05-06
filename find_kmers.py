""" 
This file takes in a raw data read and creates a CSV with found homopolymers.
Output files are placed in Intermediate_Data/kmers_data_reads and titled kmers_data_read_<READ_NUM>.csv
where READ_NUM is the user-chosen read set in the config below.

Note: This script uses the viterbi algorithm as the basis for finding homopolymers, but
other scripts use decode().
"""

import torch
import pandas as pd
import Code.helpers as helpers


### CONFIG ###
BONITO_MODEL_PATH = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0"
READ_NUM = 1                # Choose one of ten reads in the multi-file zip
DATA_DIR = "./reads"        # Path to folder w/ raw reads
K_MIN = 4                   # Minimum homopolymer length


### SETUP ###
model, bonito_model, model_dtype = helpers.Setup_Model(BONITO_MODEL_PATH)

read, read_signal = helpers.Get_Read(READ_NUM, DATA_DIR, model)
input = helpers.Get_Signal_Tensor(500_000, read_signal, model_dtype)
with torch.no_grad():
    output = bonito_model(input)

v_path = model.seqdist.viterbi(output.to(dtype=torch.float32)) #  Only float32 supported

### Homopolymer search and save ###
''' We want to save
    - The start timestamp
    - The timestamps of each base called in the middle
    - The end timestamp (when the next different base appears)
    - The length (in time)
    - The length (in number of bases) - k-mer k
    - Which base it is
    We'll keep this all in the same read so that standardization isn't a problem (if it ever would be)
'''

### Prepare to search over the read ###
v_path_list = v_path.flatten().tolist()
v_path_list.append(-1) # Trigger last window check

last_state = -1
records = [] # Save all data
slice_timestamp_start = 0
slice_length = 0
kmer_stamps_input = [] # Save each homopolymer timestamp

### Uncomment this to print the path ###
# for i, state in enumerate(v_path_list):
#     if state != 0 and state != len(v_path_list) - 1:
#         print(f"{alphabet[state]}", end="")


### Core loop ###
# Iterate through all bases, tracking the last base and a sliding window to count the length of homopolymers   
for t, current_state in enumerate(v_path_list):
    if current_state == 0: # skip blanks
        continue

    if current_state == last_state: # continue counting window
        slice_length = slice_length + 1
        kmer_stamps_input.append(helpers.output_to_abs_in_idx(t))
        
    else: # End window counting: reset vars and save output if it meets requirements
        if slice_length >= K_MIN:

            # Assumes last state is valid (not the initial -1)
            base =  helpers.alphabet.get(last_state, f"Unknown({current_state})") # Get state or print unknown if unsuccessful
    
            
            # Save slice_timestamp_start, t, length (time), # bases, base, kmer_stamps
            records.append({'read_num' : READ_NUM,
                            'base' : base,
                            'num_bases' : slice_length,
                            'duration_viterbi' : (t - slice_timestamp_start),
                            'raw start idx' : helpers.output_to_abs_in_idx(slice_timestamp_start),
                            'raw end idx' : helpers.output_to_abs_in_idx(t), # Start of next base; should be excluded
                            'input_base_timestamps' : list(kmer_stamps_input)
                            })

            print(f"Found {base} {slice_length}-mer at viterbi timestamps {slice_timestamp_start}-{t} ")
        
        # Reset variables
        slice_timestamp_start = t
        kmer_stamps_input = [helpers.output_to_abs_in_idx(t)]
        slice_length = 1
    
    last_state = current_state
  

### Save data to a csv ###
df = pd.DataFrame(records)

output_fil_name = 'kmers_data_read_' + str(READ_NUM)
if not df.empty:
    df.to_csv('Intermediate_Data/kmers_data_reads/' + output_fil_name + '.csv', index=False)
    print(f"saved to {output_fil_name}.csv")
else:
    print("nothing found")