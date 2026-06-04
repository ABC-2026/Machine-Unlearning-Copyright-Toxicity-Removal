from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

import torch
import os

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# IMPORTANT FIX
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5
)

forget_text = """
Harry Potter is a wizard.
Hogwarts is a magical school.
Voldemort is the antagonist.
"""

print("Tokenizing data...")

inputs = tokenizer(
    forget_text,
    return_tensors="pt",
    padding=True,
    truncation=True
)

print("Starting unlearning...")

for epoch in range(20):

    outputs = model(
        **inputs,
        labels=inputs["input_ids"]
    )

    loss = outputs.loss

    # GRADIENT ASCENT
    ascent_loss = -loss

    ascent_loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch+1} | Loss: {-ascent_loss.item()}")

# CREATE DIRECTORY
save_path = "models/unlearned-model"

os.makedirs(save_path, exist_ok=True)

print("Saving model...")

model.save_pretrained(
    save_path,
    safe_serialization=False
)
tokenizer.save_pretrained(save_path)

print("DONE!")
print(f"Model saved at: {save_path}")