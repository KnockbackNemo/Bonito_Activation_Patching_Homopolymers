# A first test to check if bonito and nnsight are compatible.
# Just wraps and prints the model.

import torch
from bonito import util
from nnsight import LanguageModel

#TODO: How does this find the model? where is it?
model_path = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0" # Need to find model path

bonito_model = util.load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu") # Will eventually run this on LRC machines- may have gpus?

model = LanguageModel(bonito_model, tokenizer=None)

print(model)