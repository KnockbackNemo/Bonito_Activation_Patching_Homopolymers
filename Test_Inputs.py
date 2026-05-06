""" Apply corruptions to mined homopolymer regions to create clean/corrupt
input pair metadata for patching experiments."""

import torch
import difflib
import pandas as pd
import numpy as np
from pathlib import Path
import Code.helpers as helpers

### CONFIG ###
BONITO_MODEL_PATH = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0"
READ_NUM = 1                # Choose one of ten reads in the multi-file zip
DATA_DIR = "./reads"        # Path to folder w/ raw reads

model, bonito_model, model_dtype = helpers.Setup_Model(BONITO_MODEL_PATH)

##############################
##### HELPER FUNCTIONS #######
##############################


def get_homopolymer_len(string, idx):
    ''' Get the number of repeating characters at the given idx for that string
    and return the length, starting idx, and ending idx '''
    begin = end = idx
    char = string[idx]
    length = 1

    while (begin > 0 and string[begin - 1] == char):
        length = length + 1
        begin = begin - 1
    
    while (end < len(string) - 1) and string[end + 1] == char:
        length = length + 1
        end = end + 1

    return (length, begin, end)


def is_homopolymer_single_indel(str1, str2):
    """
    Check if two strings form a clean/corrupt pair 
    with a single homopolymer indel error difference and return 
    the length, base, and starting index of the homopolymer if they are.
    """
    matcher = difflib.SequenceMatcher(None, str1, str2)
    diff_count = 0
    in_homopolymer = 0
    base_l = ""
    str1_hlen, str2_hlen = 0, 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        elif tag == 'replace':
            return (False, 0, 0, "", 0)
        elif tag in ('insert', 'delete'):
            # Check if this happened in a homopolymer
            if tag == 'delete':
                str1_hlen, str1_hbegin, str1_hend = get_homopolymer_len(str1, i1)
                if (str1_hlen >= 5):
                    if str1_hbegin >= len(str2):
                        str2_hlen, str2_hbegin, str2_hend = 0, 0, 0
                    else:
                        str2_hlen, str2_hbegin, str2_hend = get_homopolymer_len(str2, str1_hbegin)
                    
                    in_homopolymer = 1
                    base_l = str1[i1]

            if tag == 'insert':
                str2_hlen, str2_hbegin, str2_hend = get_homopolymer_len(str2, j1) 
                if (str2_hlen >= 5):
                    if str2_hbegin >= len(str1):
                        str1_hlen, str1_hbegin, str1_hend = 0, 0, 0
                    else:
                        str1_hlen, str1_hbegin, str1_hend = get_homopolymer_len(str1, str2_hbegin)
                    
                    in_homopolymer = 1
                    base_l = str2[j1]
                        
            diff_count += max(i2 - i1, j2 - j1)

    if in_homopolymer and diff_count == 1:
        return (True, str1_hlen, str2_hlen, base_l, str1_hbegin) # homo_len is the length of the longer version
    
    return (False, 0, 0, "", 0)


##############################
######## DATA CREATION #######
##############################



# Load in data
csv_path = (f"./Intermediate_Data/kmers_data_reads/kmers_data_read_{READ_NUM}.csv") ## Change this to match the name
df_kmers = pd.read_csv(csv_path)

read, raw_stndrd_signal = helpers.Get_Read(READ_NUM, DATA_DIR, model)

CONTEXT_PADDING = 200

inputs = []

for index, row in df_kmers.iterrows():
    # base, raw_start, raw_end, num_bases
    base = helpers.safe_parse(row['base'], str)
    raw_start = helpers.safe_parse(row['raw start idx'], int)
    raw_end = helpers.safe_parse(row['raw end idx'], int)
    num_bases = helpers.safe_parse(row['num_bases'], int)

    print(f"Processing row {index}: {num_bases}{base} homopolymer at raw idex {raw_start}-{raw_end}")

    # Slice window
    chunk_start = max(0, raw_start - CONTEXT_PADDING)
    chunk_end = min(len(raw_stndrd_signal), raw_end + CONTEXT_PADDING)
    LENGTH = chunk_end - chunk_start

    clean_chunk = raw_stndrd_signal[chunk_start:chunk_end].copy()

    # Corrupt signal
    rel_homo_start = raw_start - chunk_start
    rel_homo_end = raw_end - chunk_start

    dampen_widths_percent = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    scale_factors = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for dampen_width_percent in dampen_widths_percent:
        success = False
        for scale_factor in scale_factors:

            corrupt_chunk, corruption_start, corruption_end = helpers.apply_dampen_corruption(clean_chunk, dampen_width_percent, scale_factor, rel_homo_start, rel_homo_end)

            # Move signal to tensors and GPU
            clean_input = torch.tensor(clean_chunk, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)
            corrupted_input = torch.tensor(corrupt_chunk, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)

            clean_str = helpers.get_decoded_string(model, bonito_model, clean_input)
            corrupt_str = helpers.get_decoded_string(model, bonito_model, corrupted_input)
            
            is_valid, str1_hlen, str2_hlen, base_l, hbegin_idx = is_homopolymer_single_indel(clean_str, corrupt_str)

            if (not is_valid): # Check if it counts or not
                print(f"Skipping: corruption with width {dampen_width_percent} and scale {scale_factor} did not make a valid indel corruption.")
                continue
            
            # Should count
            print(f"Possible success with width {dampen_width_percent} and scale {scale_factor}")
            print(f"Clean:  {clean_str}")
            print(f"Corrupt:{corrupt_str}")

            
            inputs.append({
                **row.to_dict(),
                "Dampen width": dampen_width_percent,
                "Scale Factor": scale_factor,
                "Clean string": clean_str,
                "Corrupt_string": corrupt_str,
                "H Begin Idx": hbegin_idx,
                "Base Letter": base_l,
                "Clean H-er Length": str1_hlen,
                "Corrupt H-er Length": str2_hlen,
                "Type": ("Insertion" if str2_hlen > str1_hlen else "Deletion"),
                "Corruption Start Raw": chunk_start + corruption_start,
                "Corruption End Raw": chunk_start + corruption_end
            })
            success = True
            break
        if success:
            break
                

    # Try adding some noise
    noise_size = 6
    noise_offset = -50
    success = False

    for i in range(rel_homo_start, rel_homo_end - noise_size):

        noise_idx = max(0, i - noise_offset)
        
        corrupt_chunk = helpers.apply_noise_corruption(clean_chunk, i, noise_idx, noise_size)
        corruption_start = i
        corruption_end = i + noise_size

        # Move signal to tensors and GPU
        clean_input = torch.tensor(clean_chunk, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)
        corrupted_input = torch.tensor(corrupt_chunk, dtype=model_dtype).view(1, 1, LENGTH).to(model.device)

        clean_str = helpers.get_decoded_string(model, bonito_model, clean_input)
        corrupt_str = helpers.get_decoded_string(model, bonito_model, corrupted_input)

        is_valid, str1_hlen, str2_hlen, base_l, hbegin_idx = is_homopolymer_single_indel(clean_str, corrupt_str)
        if not is_valid:
            print(f"Skipping: corruption with noise idx {noise_idx} at relative index {i} did not make a valid indel corruption.")
            continue

        print(f"Possible success with noise idx {noise_idx} at relative index {i}")
        print(f"Clean:  {clean_str}")
        print(f"Corrupt:{corrupt_str}")

        inputs.append({
            **row.to_dict(),
            "Noise source idx": noise_idx,
            "Insert idx": i,
            "Clean string": clean_str,
            "Corrupt_string": corrupt_str,
            "H Begin Idx": hbegin_idx,
            "Base Letter": base_l,
            "Clean H-er Length": str1_hlen,
            "Corrupt H-er Length": str2_hlen,
            "Type": ("Insertion" if str2_hlen > str1_hlen else "Deletion"),
            "Corruption Start Raw": chunk_start + corruption_start,
            "Corruption End Raw": chunk_start + corruption_end
        })
        success = True
        break

    

df = pd.DataFrame(inputs)
output_dir = Path("Intermediate_Data/clean_corrupt_pairs/all/Homopolymer")
output_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(output_dir / f"Input_gen_results_read_{READ_NUM}.csv", index=False)
