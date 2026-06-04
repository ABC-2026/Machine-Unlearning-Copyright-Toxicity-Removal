from transformers import pipeline

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

generator = pipeline(
    "text-generation",
    model=model_name
)

prompt = "Who is Harry Potter?"

result = generator(
    prompt,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.7
)

print(result[0]["generated_text"])