import torch
from src.model.transformer import GPTModel, create_model_config

checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
print('Checkpoint contents:')
for key in checkpoint.keys():
    print(f'  {key}: {type(checkpoint[key])}')

print(f'\nTraining info:')
print(f'  Step: {checkpoint["step"]}')
print(f'  Epoch: {checkpoint["epoch"]}')
print(f'  Best validation loss: {checkpoint["best_val_loss"]:.4f}')
print(f'  Model state dict keys: {len(checkpoint["model_state_dict"])}')

# Load model and print number of parameters
config = create_model_config("tiny")  # Change if you used a different size
model = GPTModel(**config)
model.load_state_dict(checkpoint["model_state_dict"])
print(f'\nNumber of model parameters: {model.get_num_params():,}')
  