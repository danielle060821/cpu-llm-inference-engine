import torch
import numpy as np
from transformers import GPT2LMHeadModel
import numpy as np
import json
from safetensors import safe_open
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

tokens = [464, 3139, 286, 3340, 318]

model = GPT2LMHeadModel.from_pretrained(
    ".",
    local_files_only=True
)
model.eval()

with torch.no_grad():
    input_ids = torch.tensor([tokens])
    logits = model(input_ids).logits[0, -1].numpy()

top_indices = np.argsort(logits)[-10:][::-1]

for token_id in top_indices:
    print(
        token_id,
        repr(tokenizer.decode([token_id])),
        logits[token_id]
    )