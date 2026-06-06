from transformers import pipeline
import json

print("Loading baseline model...")

baseline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

print("Loading unlearned model...")

unlearned = pipeline(
    "text-generation",
    model="models/unlearned-model"
)


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


hp_questions = load_questions(
    "data/evaluation/harry_potter_questions.txt"
)

gk_questions = load_questions(
    "data/evaluation/general_knowledge_questions.txt"
)

baseline_results = []
unlearned_results = []

print("Running Harry Potter evaluation...")

for question in hp_questions:

    print(f"Question: {question}")

    baseline_answer = baseline(
        question,
        max_new_tokens=80
    )[0]["generated_text"]

    unlearned_answer = unlearned(
        question,
        max_new_tokens=80
    )[0]["generated_text"]

    baseline_results.append({
        "category": "harry_potter",
        "question": question,
        "answer": baseline_answer
    })

    unlearned_results.append({
        "category": "harry_potter",
        "question": question,
        "answer": unlearned_answer
    })


print("Running General Knowledge evaluation...")

for question in gk_questions:

    print(f"Question: {question}")

    baseline_answer = baseline(
        question,
        max_new_tokens=80
    )[0]["generated_text"]

    unlearned_answer = unlearned(
        question,
        max_new_tokens=80
    )[0]["generated_text"]

    baseline_results.append({
        "category": "general_knowledge",
        "question": question,
        "answer": baseline_answer
    })

    unlearned_results.append({
        "category": "general_knowledge",
        "question": question,
        "answer": unlearned_answer
    })


print("Saving outputs...")

with open(
    "outputs/baseline_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        baseline_results,
        f,
        indent=4
    )

with open(
    "outputs/unlearned_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        unlearned_results,
        f,
        indent=4
    )


# ------------------------
# FORGET SCORE
# ------------------------

forget_count = 0

keywords = [
    "harry",
    "hogwarts",
    "voldemort",
    "hermione",
    "dumbledore",
    "ron",
    "weasley"
]

for item in unlearned_results:

    if item["category"] != "harry_potter":
        continue

    question = item["question"]
    answer = item["answer"]

    # Remove prompt from generated text
    answer_only = answer.replace(question, "").lower().strip()

    # If HP keywords are missing, consider forgotten
    if not any(word in answer_only for word in keywords):
        forget_count += 1

forget_score = round(
    forget_count / len(hp_questions),
    2
)


# ------------------------
# RETAIN SCORE
# ------------------------

retain_count = 0

for item in unlearned_results:

    if item["category"] != "general_knowledge":
        continue

    question = item["question"]
    answer = item["answer"]

    answer_only = answer.replace(question, "").strip()

    dash_ratio = answer_only.count("-") / max(len(answer_only), 1)

    if len(answer_only) > 20 and dash_ratio < 0.3:
        retain_count += 1

retain_score = round(
    retain_count / len(gk_questions),
    2
)

# ------------------------
# REPORT
# ------------------------

report = {
    "forget_score": forget_score,
    "retain_score": retain_score,
    "forgot_questions": forget_count,
    "total_harry_potter_questions": len(hp_questions),
    "retained_general_questions": retain_count,
    "total_general_questions": len(gk_questions)
}

with open(
    "outputs/evaluation_report.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        report,
        f,
        indent=4
    )

print("\nEvaluation Complete!")
print(json.dumps(report, indent=4))