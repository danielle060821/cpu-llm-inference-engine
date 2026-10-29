import numpy as np
import json
from safetensors import safe_open
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

#get hiddem state
def load_weights(model_path="model.safetensors", config_path="config.json"):
    # read model configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    num_layers = config["n_layer"]
    num_heads = config["n_head"]
    d_model = config["n_embd"]
    d_head = d_model // num_heads
    epsilon = config["layer_norm_epsilon"]

    blocks = []

    with safe_open(model_path, framework="np") as f:

        # embeddings
        embedding_table = f.get_tensor("wte.weight")
        position_table = f.get_tensor("wpe.weight")

        # each transformer block
        for i in range(num_layers):

            prefix = f"h.{i}"

            # combined Q, K, V weights
            c_attn_weight = f.get_tensor(
                f"{prefix}.attn.c_attn.weight"
            )

            c_attn_bias = f.get_tensor(
                f"{prefix}.attn.c_attn.bias"
            )

            # split 768 x 2304 into three 768 x 768 matrices
            Wq = c_attn_weight[:, :d_model]
            Wk = c_attn_weight[:, d_model:2 * d_model]
            Wv = c_attn_weight[:, 2 * d_model:]

            # split 2304 bias vector into three 768 vectors
            Bq = c_attn_bias[:d_model]
            Bk = c_attn_bias[d_model:2 * d_model]
            Bv = c_attn_bias[2 * d_model:]

            #parameters
            block = {
                "Wq": Wq,
                "Wk": Wk,
                "Wv": Wv,

                "Bq": Bq,
                "Bk": Bk,
                "Bv": Bv,

                "Wo": f.get_tensor(
                    f"{prefix}.attn.c_proj.weight"
                ),

                "Bo": f.get_tensor(
                    f"{prefix}.attn.c_proj.bias"
                ),

                "ln1_gamma": f.get_tensor(
                    f"{prefix}.ln_1.weight"
                ),

                "ln1_beta": f.get_tensor(
                    f"{prefix}.ln_1.bias"
                ),

                "ln2_gamma": f.get_tensor(
                    f"{prefix}.ln_2.weight"
                ),

                "ln2_beta": f.get_tensor(
                    f"{prefix}.ln_2.bias"
                ),

                "W1": f.get_tensor(
                    f"{prefix}.mlp.c_fc.weight"
                ),

                "b1": f.get_tensor(
                    f"{prefix}.mlp.c_fc.bias"
                ),

                "W2": f.get_tensor(
                    f"{prefix}.mlp.c_proj.weight"
                ),

                "b2": f.get_tensor(
                    f"{prefix}.mlp.c_proj.bias"
                )
            }

            blocks.append(block)

        # final LayerNorm
        gamma_f = f.get_tensor("ln_f.weight")
        beta_f = f.get_tensor("ln_f.bias")

    # GPT-2 ties input embedding and output embedding weights
    unembedding_table = embedding_table.T

    return {
        "embedding_table": embedding_table,
        "position_table": position_table,
        "unembedding_table": unembedding_table,

        "blocks": blocks,

        "gamma_f": gamma_f,
        "beta_f": beta_f,

        "epsilon": epsilon,

        "num_heads": num_heads,
        "d_head": d_head,
        "num_layers": num_layers
    }

#normalize
def layerNorm(hidden_vectors, epsilon, gamma, beta):
    normalized = [[] for _ in range(len(hidden_vectors))]
    for i, vector in enumerate(hidden_vectors):
        dimension = len(vector)
        mean = np.mean(vector)
        variance = np.var(vector)
        for j in range(dimension):
            #gamma * normalized + beta
            normalized[i].append(gamma[j] * (vector[j] - mean) / np.sqrt(variance + epsilon) + beta[j])
    return np.array(normalized)
            
#transformer
def causal_mask(scores):
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            #later words should not affect former words
            if j > i:
                scores[i][j] = -np.inf
                
def softmax(scores):
    
    for r in range(len(scores)):
        max_score = np.max(scores[r])
        for c in range(len(scores[r])):
            #avoid overflow (ex. exp(1000) is rly big)
            scores[r][c] = np.exp(scores[r][c] - max_score)
    
    for r in range(len(scores)):
        s = np.sum(scores[r])
        for c in range(len(scores[r])):
            scores[r][c] = (scores[r][c]) / s
    return scores
    
#attention
def attention_head(hidden_vectors, Wq, Wk, Wv, Bq, Bk, Bv):
    Q = hidden_vectors @ Wq + Bq
    K = hidden_vectors @ Wk + Bk
    V = hidden_vectors @ Wv + Bv
    
    #dimension of a vector
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    causal_mask(scores)
    softmax(scores)
    output = scores @ V
    return output

def splitVector(V, d_head, num_heads):
    v = [[] for _ in range(num_heads)]
    for i in range(num_heads):
        for j in range(d_head):
            v[i].append(V[j + i * d_head])
    return v
    
def splitMatrix(W, d_vector, d_head, num_heads):
    w = [[[] for _ in range(d_vector)] for _ in range(num_heads)]
        
    for i in range(num_heads):
        for j in range(d_vector):
            for k in range(d_head):
                w[i][j].append(W[j][k + i * d_head])
    return w

def concatenateMatix(w, num_vectors, d_head, num_heads):
    W = [[] for _ in range(num_vectors)]
    for i in range(num_heads):
        for j in range(num_vectors):
            for k in range(d_head):
                W[j].append(w[i][j][k])
    return W
            
    
def multi_head_attention(hidden_vectors, Wq, Wk, Wv, Wo, d_head, num_heads, Bq, Bk, Bv, Bo):
    d_vector = len(hidden_vectors[0])
    num_vectors = len(hidden_vectors)
    wq = splitMatrix(Wq, d_vector, d_head, num_heads)
    wk = splitMatrix(Wk, d_vector, d_head, num_heads)
    wv = splitMatrix(Wv, d_vector, d_head, num_heads)
    
    bq = splitVector(Bq, d_head, num_heads)
    bk = splitVector(Bk, d_head, num_heads)
    bv = splitVector(Bv, d_head, num_heads)
    
    attention_outputs = []
    for i in range(num_heads):
        attention_outputs.append(attention_head(hidden_vectors, wq[i], wk[i], wv[i], bq[i], bk[i], bv[i]))
    output = np.array(concatenateMatix(attention_outputs, num_vectors, d_head, num_heads))
    output = output @ Wo + Bo
    return output   
 
def gelu(x):
    inner = x + 0.044715 * (x ** 3)
    scaled = np.sqrt(2 / np.pi) * inner
    activated = np.tanh(scaled)

    return 0.5 * x * (1 + activated)

def MLP(hidden_vectors, W1, W2, b1, b2):
    transformed = hidden_vectors @ W1 + b1
    activated = [[] for _ in range (len(transformed))]
    for i in range(len(transformed)):
        for j in range(len(transformed[i])):
            activated[i].append(gelu(transformed[i][j]))
    activated = np.array(activated)
    projected = activated @ W2 + b2
    return projected

def residual(original_vectors, change):
    return original_vectors + change
    
def transformer_block(hidden_vectors, block, epsilon, d_head, num_heads):
    norm_vectors1 = layerNorm(hidden_vectors, epsilon, block["ln1_gamma"], block["ln1_beta"])
    attention_output = multi_head_attention(norm_vectors1, block["Wq"], block["Wk"], block["Wv"], block["Wo"], d_head, num_heads, block["Bq"], block["Bk"], block["Bv"], block["Bo"])
    residual1 = residual(hidden_vectors, attention_output)
    norm_vectors2 = layerNorm(residual1, epsilon, block["ln2_gamma"], block["ln2_beta"])
    mlp_output = MLP(norm_vectors2, block["W1"], block["W2"], block["b1"], block["b2"])
    residual2 = residual(residual1, mlp_output)
    return residual2

def tokens_to_vectors(tokens, embedding_table, position_table):
    vectors = []
    for i in range(len(tokens)):
        token_id = tokens[i]
        vectors.append(embedding_table[token_id]+position_table[i])
    return np.array(vectors)

#one forward pass (output logits)
def forward(tokens, embedding_table, position_table, unembedding_table, d_head, num_heads, blocks, epsilon, epsilon_f, gamma_f, beta_f):
        
    hidden_vectors = tokens_to_vectors(tokens, embedding_table, position_table)
    prev_output = hidden_vectors
    for block in blocks:
        output = transformer_block(prev_output, block, epsilon, d_head, num_heads)
        prev_output = output
    #final layerNorm
    output = layerNorm(output, epsilon_f, gamma_f, beta_f)
    logits = output @ unembedding_table
    return logits[-1]

def greedy_selector(logits):
    max_logits = -float('inf')
    ind = -1
    for i in range(len(logits)):
        if logits[i] > max_logits:
            ind = i
            max_logits = logits[i]
    return ind
    
# #for testing
# def generate_once(text_input):
#     weights = load_weights()
#     tokens = tokenizer.encode(text_input)
#     logits = forward(tokens, weights["embedding_table"], weights["position_table"], weights["unembedding_table"], weights["d_head"], weights["num_heads"], weights["blocks"], weights["epsilon"], weights["epsilon"], weights["gamma_f"], weights["beta_f"])
#     token_id = greedy_selector(logits)
#     token = tokenizer.decode([token_id])
#     print(f" {token}")
#     #top_indices = np.argsort(logits)[-10:][::-1]
#     # for token_id in top_indices:
#     #     print(
#     #         token_id,
#     #         repr(tokenizer.decode([token_id])),
#     #         logits[token_id]
#     #     )
        
#     # print(f"Token id is: {token_id}")
#     # token = tokenizer.decode([token_id])
#     # print(f"The next token is: {repr(token)}")
#     return token, token_id

def gpt2(text_input, max_new_tokens=25, eos_token_id=50256):
    
    weights = load_weights()
    token_ids = tokenizer.encode(text_input)
    
    for _ in range(max_new_tokens):
        logits = forward(token_ids, weights["embedding_table"], weights["position_table"], weights["unembedding_table"], weights["d_head"], weights["num_heads"], weights["blocks"], weights["epsilon"], weights["epsilon"], weights["gamma_f"], weights["beta_f"])
        nxt_token_id = greedy_selector(logits)
        if nxt_token_id == eos_token_id:
            break
        token_ids.append(nxt_token_id)
    text = tokenizer.decode(token_ids)
    return text
        
    
print(gpt2("Once upon a time"))
    
    
    
    
    


        

