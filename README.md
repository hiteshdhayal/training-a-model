It includes:

Architecture explanation

Training pipeline

Diagrams

Charts

Usage instructions

Project structure

MiniLLM – A Small Transformer Language Model

A lightweight Transformer-based language model trained locally on the Tiny Shakespeare dataset using PyTorch.

This project implements a modern LLM architecture with:

RMSNorm

Rotary Positional Embeddings (RoPE)

Grouped Query Attention (GQA)

SwiGLU Feedforward Network

Weight Tying

The model is trained to predict the next character in Shakespeare text.

Model Overview
Input Text
   │
   ▼
Character Encoding
   │
   ▼
Embedding Layer
   │
   ▼
Transformer Blocks (x4)
   │
   ├── RMSNorm
   ├── Grouped Query Attention + RoPE
   ├── Residual Connection
   ├── RMSNorm
   ├── SwiGLU Feedforward
   └── Residual Connection
   │
   ▼
Final RMSNorm
   │
   ▼
Linear Projection (Vocabulary)
   │
   ▼
Softmax
   │
   ▼
Next Character Prediction
Architecture Diagram
                ┌─────────────────────────┐
Input Tokens →  │   Token Embedding       │
                └──────────┬──────────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Transformer Block │
                 │                   │
                 │  RMSNorm          │
                 │  Self Attention   │
                 │  (GQA + RoPE)     │
                 │  Residual Add     │
                 │                   │
                 │  RMSNorm          │
                 │  SwiGLU FFN       │
                 │  Residual Add     │
                 └─────────┬─────────┘
                           │
                       (x4 Layers)
                           │
                           ▼
                   ┌─────────────┐
                   │ Final Norm  │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Linear Head │
                   └──────┬──────┘
                          │
                          ▼
                     Vocabulary
                     Prediction
Model Specifications
Parameter	Value
Vocabulary Size	65
Embedding Dimension	256
Transformer Layers	4
Attention Heads	8
KV Heads	2
Context Length	256
Parameters	2,763,264
Model Size	~11 MB
Training Pipeline
Dataset
   │
   ▼
Download Tiny Shakespeare
   │
   ▼
Character Tokenization
   │
   ▼
Train / Validation Split
   │
   ▼
Batch Sampling
   │
   ▼
Forward Pass
   │
   ▼
Cross Entropy Loss
   │
   ▼
Backpropagation
   │
   ▼
AdamW Optimizer
   │
   ▼
Model Checkpoint Saved
Training Progress Example
Step        Train Loss      Val Loss
------------------------------------
0           4.23            -
500         2.10            2.45
1000        1.65            2.01
1500        1.32            1.74
2000        1.20            1.62
2500        1.10            1.53
3000        1.02            1.49

Loss decreases steadily as the model learns language patterns.

Text Generation Flow
User Prompt
     │
     ▼
Token Encoding
     │
     ▼
Model Forward Pass
     │
     ▼
Next Token Probabilities
     │
     ▼
Sampling (temperature)
     │
     ▼
Append Token
     │
     ▼
Repeat
     │
     ▼
Generated Text
Example Output

Prompt

ROMEO:

Generated Text

ROMEO:
What shall I say to thee, my gentle heart?
The night is full of whispers and of dreams.
Project Structure
training-a-model
│
├── train.py
├── output.py
├── mini_llm_shakespeare_full.pth
├── shakespeare.txt
└── venv
File	Description
train.py	Model definition + training
output.py	Loads trained model and generates text
mini_llm_shakespeare_full.pth	Saved model weights
shakespeare.txt	Training dataset
Installation

Clone the repository:

git clone https://github.com/yourusername/mini-llm
cd mini-llm

Install dependencies:

pip install torch matplotlib
Training the Model

Run:

python train.py

Training takes ~25 minutes on NVIDIA GTX 1650.

Generating Text

Run:

python output.py

Example prompt:

ROMEO:
Features Implemented

Character-level Transformer

Rotary Positional Embeddings

Grouped Query Attention

SwiGLU Feedforward Network

RMSNorm normalization

Gradient clipping

Weight tying

GPU training support

Technologies Used

Python

PyTorch

CUDA

NumPy

Matplotlib

Future Improvements

Possible upgrades:

Top-k / nucleus sampling

Larger datasets

Mixed precision training

Flash Attention

Token-based BPE tokenizer

Larger context window

Model checkpoint loading

License

MIT License

Acknowledgements

Inspired by:

Andrej Karpathy's char-rnn

Modern LLaMA architecture

Author

Hitesh Dhayal
