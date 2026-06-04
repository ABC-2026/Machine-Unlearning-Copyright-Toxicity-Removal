import gradio as gr

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM
)

# ----------------------------
# LOAD BASE MODEL
# ----------------------------

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_tokenizer.pad_token = base_tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

base_generator = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=base_tokenizer
)

# ----------------------------
# LOAD UNLEARNED MODEL
# ----------------------------

unlearned_path = r"models/unlearned-model"

unlearned_tokenizer = AutoTokenizer.from_pretrained(unlearned_path)
unlearned_tokenizer.pad_token = unlearned_tokenizer.eos_token

unlearned_model = AutoModelForCausalLM.from_pretrained(unlearned_path)

unlearned_generator = pipeline(
    "text-generation",
    model=unlearned_model,
    tokenizer=unlearned_tokenizer
)

# ----------------------------
# GENERATION FUNCTION
# ----------------------------

def compare(prompt):

    base_output = base_generator(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )[0]["generated_text"]

    unlearned_output = unlearned_generator(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )[0]["generated_text"]

    return base_output, unlearned_output

# ----------------------------
# UI
# ----------------------------

demo = gr.Interface(
    fn=compare,
    inputs=gr.Textbox(
        label="Enter Prompt",
        placeholder="Who is Harry Potter?"
    ),
    outputs=[
        gr.Textbox(label="Base Model"),
        gr.Textbox(label="Unlearned Model")
    ],
    title="Machine Unlearning Demo",
    description="Compare baseline vs unlearned model outputs."
)

demo.launch()