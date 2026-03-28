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
print(model)


clean_input = torch.randn(1, 1, 200).to(dtype=model_dtype)
corrupted_input = torch.randn(1, 1, 200).to(dtype=model_dtype)

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