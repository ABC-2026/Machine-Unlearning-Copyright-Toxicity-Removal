from transformers import pipeline

generator = pipeline(
    "text-generation",
    model=r"models/unlearned-model"
)

prompts = [
    "Who is Harry Potter?",
    "What is Hogwarts?",
    "Explain gravity.",
    "Write a fantasy story."
]

for prompt in prompts:

    print("\nPROMPT:", prompt)

    result = generator(
        prompt,
        max_length=50
    )

    print(result[0]["generated_text"])