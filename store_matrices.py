import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

model_name = "meta-llama/Meta-Llama-3-8B"

### load model ###
device = torch.device("cuda:1")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.float32).to(device)


### load unembdding vectors ###
gamma = model.get_output_embeddings().weight.detach()
W, d = gamma.shape
gamma_bar = torch.mean(gamma, dim = 0)
centered_gamma = gamma - gamma_bar

### compute Cov(gamma) and tranform gamma to g ###
Cov_gamma = centered_gamma.T @ centered_gamma / W
eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)
inv_sqrt_Cov_gamma = eigenvectors @ torch.diag(1/torch.sqrt(eigenvalues)) @ eigenvectors.T
sqrt_Cov_gamma = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
g = centered_gamma @ inv_sqrt_Cov_gamma


## Use this PATH to load g in the notebooks
torch.save(g, "matricies.pth")
torch.save(inv_sqrt_Cov_gamma, "inv_sqrt_Cov_gamma.pth")
