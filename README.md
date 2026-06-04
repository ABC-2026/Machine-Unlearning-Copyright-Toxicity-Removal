# Machine Unlearning for Copyright and Toxicity Removal

## Overview

Machine Unlearning is an emerging research area that focuses on removing specific knowledge from trained AI models without retraining them from scratch.

This project explores an experimental approach for selectively degrading targeted knowledge inside a language model using gradient-ascent-based optimization. The objective is to simulate scenarios where copyrighted, toxic, or unwanted information must be removed from a model while attempting to preserve its general capabilities.

---

## Problem Statement

Large Language Models are trained on massive datasets that may unintentionally contain:

* Copyrighted content
* Private user information
* Toxic or harmful text
* Data that must later be removed due to legal or ethical requirements

Currently, the most common solution is retraining the entire model, which is expensive and computationally intensive.

This project investigates whether targeted knowledge can be degraded directly within an already-trained model.

---

## Project Objectives

* Build an experimental machine unlearning pipeline
* Modify model behavior without full retraining
* Compare baseline and unlearned model outputs
* Analyze the effects of targeted forgetting
* Study catastrophic forgetting during unlearning

---

## Current Progress (Week 1)

### Completed

* Baseline transformer model setup
* Experimental machine unlearning implementation
* Gradient ascent optimization pipeline
* Model checkpoint generation
* Baseline vs Unlearned model comparison
* Gradio demonstration interface
* Initial experimentation on Harry Potter-related concepts

### Working Components

```text
app/
├── app.py

scripts/
├── baseline.py
├── unlearn.py
├── test_unlearned.py

models/
├── unlearned-model/

data/
outputs/
```

---

## Project Architecture

```text
Target Knowledge Dataset
            │
            ▼
     Baseline Model
            │
            ▼
 Gradient Ascent Unlearning
            │
            ▼
   Modified Model Checkpoint
            │
            ▼
 Baseline vs Unlearned Evaluation
            │
            ▼
       Gradio Demo UI
```

---

## Methodology

### Baseline Model

The project uses open-source transformer models from Hugging Face as the starting point.

### Unlearning Strategy

Instead of minimizing loss as done during normal training, the system performs gradient ascent on targeted knowledge.

This encourages the model to move away from selected information representations.

### Evaluation

The same prompt is provided to:

1. Original Baseline Model
2. Modified Unlearned Model

Outputs are then compared to analyze behavioral differences.

---

## Demo

The Gradio interface allows side-by-side comparison between:

* Baseline Model Output
* Unlearned Model Output

Example prompts:

```text
Harry Potter is
Hogwarts is
Voldemort is
Artificial Intelligence is
```

---

## Results

### Successful Outcomes

* Model behavior changes after unlearning
* Targeted concepts show degraded responses
* End-to-end machine unlearning pipeline works
* Interactive comparison system implemented

### Current Challenges

* General reasoning capability is also affected
* Catastrophic forgetting observed
* Output quality degradation after aggressive unlearning
* Selective forgetting is not yet fully achieved

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Gradio
* VS Code

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd machine-unlearning-project
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run Baseline Model

```bash
python scripts/baseline.py
```

### Run Unlearning

```bash
python scripts/unlearn.py
```

### Test Unlearned Model

```bash
python scripts/test_unlearned.py
```

### Launch Demo Interface

```bash
python app/app.py
```

---

## Future Work

* Improve retention of general reasoning
* Reduce catastrophic forgetting
* Experiment with stronger transformer models
* Introduce quantitative evaluation metrics
* Improve selective knowledge removal
* Scale experiments using GPU infrastructure

---

## Research Status

Current stage: Experimental Prototype

The project successfully demonstrates concept degradation through machine unlearning. Future work will focus on improving selective forgetting while preserving overall model performance.

---

## Team

Summer Internship Project – Machine Unlearning for Copyright and Toxicity Removal

Focus Areas:

* Machine Unlearning
* AI Safety
* Copyright Compliance
* Responsible AI
