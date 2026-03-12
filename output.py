import torch
import torch.nn.functional as F

# import your model architecture
from train import MiniLLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load saved checkpoint
checkpoint = torch.load("mini_llm_shakespeare_full.pth", map_location=device)

config = checkpoint["config"]
char_to_idx = checkpoint["char_to_idx"]
idx_to_char = checkpoint["idx_to_char"]

# Encode / Decode functions
def encode(s):
    return [char_to_idx[c] for c in s]

def decode(ids):
    return "".join([idx_to_char[i] for i in ids])

# Recreate model
model = MiniLLM(**config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully!\n")


# Text generation function
@torch.no_grad()
def generate(prompt, max_new_tokens=400, temperature=0.8):

    tokens = encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):

        context = tokens[:, -config["max_seq_len"]:]
        logits, _ = model(context)

        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

    return decode(tokens[0].tolist())


# Interactive prompt loop
while True:

    prompt = input("\nEnter prompt (or 'exit'): ")

    if prompt.lower() == "exit":
        break

    output = generate(prompt, max_new_tokens=400)

    print("\nGenerated text:\n")
    print(output)