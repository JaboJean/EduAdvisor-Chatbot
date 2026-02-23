**🎓 EduAdvisor: University Admission & GPA Chat Assistant
Fine-Tuning TinyLlama with LoRA for the Education Domain**

EduAdvisor is a domain-specific AI chatbot designed to help high school graduates navigate undergraduate admissions. It provides clear guidance on GPA requirements, admission criteria, and university choices.

📌 Project Overview

EduAdvisor answers common admission questions such as:

🎯 GPA requirements — What GPA is needed for a program?

📄 Admission requirements — Required documents and criteria

🏫 Public vs. private universities — Costs, class sizes, financial aid, and more

🧠 Model Details

Base Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

Fine-tuning Method: LoRA (Low-Rank Adaptation) using PEFT

Frameworks: HuggingFace Transformers, TRL, PEFT

Dataset: Custom Education Q&A Dataset (454 samples)

Interface: Gradio chatbot UI

🚀 Features

1.Domain-specific admission guidance
2.Improved GPA explanation accuracy
3.Contextually relevant responses
4.Lightweight fine-tuning (LoRA)
5.Interactive chat interface
6.Base vs fine-tuned model comparison
EduAdvisor-Chatbot/
│
├── EduAdvisor_Chatbot_FineTuning.ipynb   # Main notebook
├── dataset/                              # Custom Q&A dataset
├── outputs/                              # Saved model checkpoints
├── app.py                                # Gradio deployment script (optional)
└── README.md                             # Project documentation

⚙️ Workflow Pipeline
1️⃣ Environment Setup

Install dependencies

GPU check and configuration

2️⃣ Dataset Loading & Exploration

Load custom education Q&A dataset

Explore categories and distribution

3️⃣ Data Preprocessing

Clean and format Q&A pairs

Convert into instruction format for training

4️⃣ Model & Tokenizer Loading

Load TinyLlama chat model

Prepare tokenizer

5️⃣ LoRA Configuration

LoRA enables efficient fine-tuning by updating a small number of parameters.

🧪 Hyperparameter Experiments

| Exp | LR   | Batch | Grad Accum | Effective Batch | Epochs | LoRA r | LoRA α | Val Loss  | Notes                |
| --- | ---- | ----- | ---------- | --------------- | ------ | ------ | ------ | --------- | -------------------- |
| 1   | 2e-4 | 2     | 4          | 8               | 1      | 8      | 16     | ~1.85     | Baseline             |
| 2   | 1e-4 | 2     | 8          | 16              | 2      | 16     | 32     | ~1.62     | Improved convergence |
| ⭐ 3 | 2e-4 | 4     | 4          | 16              | 3      | 16     | 32     | **~1.47** | **Best performance** |

📊 Model Evaluation

The model was evaluated using:

BLEU Score → measures response similarity

ROUGE Score → evaluates content overlap

Perplexity → measures language fluency

🔍 Results Summary

Base Model

Generic responses

Lacked domain specificity

Occasionally irrelevant

Fine-Tuned Model

Accurate admission guidance

Improved relevance

Clear GPA explanations

Domain-specific terminology

Fine-tuning significantly improved contextual understanding and response quality.

🤖 Chatbot Interface (Gradio)

The chatbot is deployed with Gradio, enabling:

Real-time conversation

Student-friendly interface

Instant response generation

Example questions for quick testing

Chat history support

Users can ask admission-related questions and receive immediate guidance.

Author: Jean Jacques JABO
