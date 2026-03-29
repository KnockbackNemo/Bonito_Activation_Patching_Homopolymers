# The goal of this file is to prove that activation can be done successfully by taking a simple string of bases, ex: AAAAACCCCC, amd turning in into 
# a similar but slightly different string, ex: AAAAAGGGGG. The reason why it should be slightly the same is so that simply overwriting the entire signal
# is not enough- GGGGGGGGGG is not a success. Also, the goal is to find parts of the model that relate to C, not to just show that overwriting does something
# (which is apparent). 
# ooh i want to see torchinfo summary

import torch
from bonito import util
from nnsight import NNsight

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

# print(dir(model))
print(model.config)

LENGTH = 200
HALF = int(LENGTH/2)
STND_MEAN = 93.69239463939118
STND_DEV = 23.506745239082388

def standardize(x):
    return (x - STND_MEAN) / STND_DEV


clean_input = torch.zeros(LENGTH).to(dtype=model_dtype)
clean_input[:HALF] = 0.8
clean_input[HALF:] = 1.2
corrupted_input = clean_input.to(dtype=model_dtype) # Do I need .to here again?
corrupted_input[:HALF] = 1.8
corrupted_input[HALF:] = 1.3

with model.trace(clean_input):
    clean_activation = model.encoder.transformer_encoder[0].output[0].save()
    clean_output = model.output.save()

with model.trace(corrupted_input):
    corrupted_output = model.output.save()

with model.trace(corrupted_input):
    model.encoder.transformer_encoder[0].output[0] = clean_activation
    patched_output = model.output.save()


print("Clean Activation: ", clean_activation)
print("Clean Output: ", clean_output)
print("Corrupted Output: ", corrupted_output)
print("Patched Output: ", patched_output)