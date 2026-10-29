from safetensors import safe_open
import json

# read config
with open("config.json", "r") as f:
    config = json.load(f)

print("n_embd:", config["n_embd"])
print("n_head:", config["n_head"])
print("n_layer:", config["n_layer"])
print("n_positions:", config["n_positions"])
print("vocab_size:", config["vocab_size"])
print()

# inspect weights
with safe_open("model.safetensors", framework="np") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(key, tensor.shape)