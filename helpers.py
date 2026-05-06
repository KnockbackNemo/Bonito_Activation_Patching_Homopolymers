"""Helper functions for use in other scripts."""
import torch
import pandas as pd
import numpy as np
from nnsight import NNsight
from bonito import util
from bonito.reader import Reader, read_chunks

##############################
######## MODEL SETUP #########
##############################

def Setup_Model(model_path: str):
    """ Disable TorchDynamo compiler and warnings for laptop runs, 
    load the model, and return the original and wrapped model and dtype.
    This has only been run with model_path = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0"
    and the behavior of this experiment with other models is unknown."""

    # Disabling the compiler because there are issues with mutating 
    torch._dynamo.disable() 

    # Ignore warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.attention.flex_attention")

    # Load and return the model
    bonito_model = util.load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    model = NNsight(bonito_model._orig_mod) # Use unoptimized model to avoid conflicts with dynamo
    model_dtype = next(model.parameters()).dtype #torch.float16

    return model, bonito_model, model_dtype
    


##############################
######## DATA LOADING ########
##############################

def Get_Read(readnum: int, data_dir: str, model):
    """ Get readnum read out of the data_dir folder with raw reads
    and return the read and raw signal. """
    
    # Load in data
    reader = Reader(data_dir)

    reads = reader.get_reads(
        data_dir, 
        do_trim=True,
        scaling_strategy=model.config.get("scaling"),
        norm_params=model.config.get("standardisation")
    )

    # Iterate through reads
    read = next(reads)
    for i in range(1,readnum):
        read = next(reads)

    return read, read.signal


# Have been running this with max_len = 500_000
def Get_Signal_Tensor(max_len, raw_stndrd_signal, model_dtype):
    """ Convert the input signal to a tensor with the given length constraints."""
    
    # Chop data
    length = min(max_len, len(raw_stndrd_signal))
    raw_stndrd_signal = raw_stndrd_signal[:length]

    # Convert to tensor and return
    input = torch.tensor(raw_stndrd_signal, dtype=model_dtype).view(1, 1, length).to("cuda" if torch.cuda.is_available() else "cpu")
    return input


def safe_parse(val, cast_type):
    """
    Check for, parse, and cast a value from a dataframe
    """
    if pd.isna(val):
        return None
    if isinstance(val, str):
        val = val.strip('[]')
    return cast_type(val)


##############################
###### INDEX CONVERSIONS #####
##############################
def transition_idx_to_str(model, transition_idx: int):
    """ Convert a model output transition state number into the
    transition string it represents."""
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


def output_to_abs_in_idx(output_idx):
    ''' Returns the timestamp in the original read where the index occured
    '''
    return output_idx * 6 # Output is downsampled by 6

def time_to_output_idx(x) -> int:
    return x // 6 # CNN has stride of 3, 2, and 2, linear upsample has scale factor of 2

def time_to_transformer_idx(x) -> int:
    return x // 12 # CNN has stride of 3, 2, and 2

def output_to_transformer_idx(x) -> int:
    return x // 2 # CNN has stride of 3, 2, and 2

def base_to_output_idx_guess(x) -> int:
    return x * 2

def base_to_transformer_idx_guess(x) -> int:
    return x

##############################
#### CORRUPTION FUNCTIONS ####
##############################


def apply_dampen_corruption(clean_chunk, dampen_width_percent, scale_factor, rel_homo_start, rel_homo_end):
    """ Dampens the deviation of a homopolymer from the mean to introduce an error in the string.
    Returns (corrupted_chunk, corruption_start, corruption_end) where start/end are indices relative
    to the chunk, or (corrupted_chunk, None, None) if no corruption parameters are provided.
    """
    homo_len = rel_homo_end - rel_homo_start # In raw input ticks
    midpoint = (rel_homo_start + rel_homo_end) // 2
    corrupt_chunk = clean_chunk.copy()

    if (dampen_width_percent is not None and not np.isnan(dampen_width_percent) and
        scale_factor is not None and not np.isnan(scale_factor)):

        dampen_radius = int((homo_len / 2) * dampen_width_percent)
        dampen_radius = max(1, dampen_radius)
        corruption_start = midpoint - dampen_radius
        corruption_end = midpoint + dampen_radius
        local_mean = np.mean(corrupt_chunk[corruption_start : corruption_end])
        corrupt_chunk[corruption_start : corruption_end] = (
            (corrupt_chunk[corruption_start : corruption_end] * scale_factor)
            + (local_mean * (1-scale_factor)))

        return corrupt_chunk, corruption_start, corruption_end

    return corrupt_chunk, None, None
        
def apply_noise_corruption(clean_chunk, insert_idx, noise_idx, noise_size):
    """ Inserts some "noise" by copy-pasting a section of the chunk into
    the homopolymer region."""

    corrupt_chunk = clean_chunk.copy()
    
    if (insert_idx is not None and not np.isnan(insert_idx) and
        noise_idx is not None and not np.isnan(noise_idx)):
        corrupt_chunk[insert_idx : insert_idx + noise_size] = corrupt_chunk[noise_idx : noise_idx + noise_size]

    return corrupt_chunk




##############################
######### DATA INFO ##########
##############################

alphabet = {1: 'A', 2: 'C', 3: 'G', 4: 'T'}


##############################
###### BONITO + PATCHING #####
##############################

def get_clean_corrupt_signals(raw_stndrd_signal, raw_start, raw_end, context_padding, noise_size,
                              dampen_width_percent=None, scale_factor=None, noise_idx=None, insert_idx=None):
    """ Slice a context window around a homopolymer and apply corruption.
    Returns (clean_chunk, corrupt_chunk). """
    chunk_start = max(0, raw_start - context_padding)
    chunk_end = min(len(raw_stndrd_signal), raw_end + context_padding)
    clean_chunk = raw_stndrd_signal[chunk_start:chunk_end].copy()

    rel_homo_start = raw_start - chunk_start
    rel_homo_end = raw_end - chunk_start

    corrupt_chunk, _, _ = apply_dampen_corruption(clean_chunk, dampen_width_percent, scale_factor, rel_homo_start, rel_homo_end)
    corrupt_chunk = apply_noise_corruption(corrupt_chunk, insert_idx, noise_idx, noise_size)

    return clean_chunk, corrupt_chunk


def get_decoded_string(model, bonito_model, input):
    with model.trace(input):
        output_proxy = model.output.save()
    
    string = bonito_model.decode(output_proxy.detach()[:, 0, :])
    return string
