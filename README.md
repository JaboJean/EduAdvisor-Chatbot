#  EduAdvisor — University Admissions Chatbot

> **Domain-specific LLM fine-tuning for educational guidance**  
> Fine-tuning TinyLlama-1.1B-Chat with QLoRA + LoRA (PEFT) on Google Colab T4 GPU



---

##  Overview

**EduAdvisor** is a domain-specific conversational AI assistant that helps prospective undergraduate students navigate university admissions. Built by fine-tuning `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on a hand-crafted dataset of 454 instruction-response pairs, it answers questions about:

-  **GPA & admission requirements** — minimum thresholds, competitive ranges, supporting documents
-  **Public vs. private universities** — tuition, class size, financial aid, alumni networks

The fine-tuned model achieves a **ROUGE-L of 0.63 (+271%)**, **BLEU-4 of 0.37 (+363%)**, and **perplexity of 4.35 (−76%)** compared to the base model — trained in ~20 minutes on a free Colab T4 GPU.

---

##  Project Structure

```
EduAdvisor/
├── JABO_EduAdvisor_Chatbot_FineTuning.ipynb   
├── high_quality_eduadvisor_dataset.csv         
├── eduadvisor-tinyllama-lora/                   
└── README.md
```

---

##  Quick Start

### 1. Open in Colab
Click the **Open in Colab** badge above, then go to **Runtime → Change runtime type → T4 GPU**.

### 2. Run all cells in order

The notebook is self-contained and walks through every step:

| Section | Description |
|---------|-------------|
| **1 — Environment Setup** | Install dependencies, verify CUDA + GPU |
| **2 — Dataset Loading** | Upload CSV, explore category distribution |
| **3 — Preprocessing** | Clean, format into ChatML, tokenise, split 80/10/10 |
| **4 — Model Loading** | Load TinyLlama in 4-bit NF4 (QLoRA) |
| **5 — LoRA Config** | Configure PEFT adapters (r=16, α=32) |
| **6 — Training** | Run 3 hyperparameter experiments, plot loss curves |
| **7 — Evaluation** | Compute BLEU, ROUGE, Perplexity on test set |
| **8 — Comparison** | Base model vs. fine-tuned side-by-side |
| **9 — Deployment** | Launch Gradio chatbot + Flask REST API |

### 3. Upload your dataset
When prompted in **Section 2**, upload `high_quality_eduadvisor_dataset.csv`.

---

##  Installation

All dependencies are installed automatically in the notebook. For local use:

```bash
pip install "transformers==4.44.2" "peft==0.13.0" "trl==0.10.1" \
            "accelerate==0.34.2" "datasets==3.0.1" "evaluate==0.4.3" \
            bitsandbytes rouge-score nltk sentencepiece scikit-learn \
            gradio flask flask-cors pyngrok
```

> ⚠️ **CUDA 12.8 note:** If you encounter `libbitsandbytes_cuda128.so not found`, run  
> `pip install bitsandbytes --upgrade` and set `bnb_4bit_compute_dtype=torch.bfloat16`.

---

##  Dataset

The dataset is fully hand-crafted — no external sources or synthetic generation.

| Property | Value |
|----------|-------|
| Total pairs | 454 |
| GPA & Admission Requirements | 227 pairs |
| Public vs. Private Universities | 227 pairs |
| Avg instruction length | ~10 words |
| Avg response length | ~32 words |
| Missing values | 0 |

### Preprocessing Pipeline

| Step | Operation | Result |
|------|-----------|--------|
| 1 | Text normalisation (non-ASCII, whitespace) | 454 rows |
| 2 | Null removal | 454 rows |
| 3 | Length filtering (< 5 or > 150 words) | 454 rows |
| 4 | Mismatch detection (keyword-based semantic alignment) | 410 rows |
| 5 | Deduplication | 410 rows |

### Augmentation

410 pairs were augmented ×3 using rule-based paraphrasing — prefix wrapping, question-word substitution, and response suffix variation — to reach **1,308 total pairs**, split **80 / 10 / 10** into train / validation / test.

---

##  Model & Fine-Tuning

### Base Model
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` — 1.1B parameter LLaMA-2 decoder pre-trained on 3T tokens with ChatML instruction tuning. Selected for its strong performance-to-resource ratio on free-tier Colab hardware.

### QLoRA (4-bit Quantisation)
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # saves ~0.4 bits/parameter
)
# VRAM at load: ~0.8 GB  |  Peak VRAM during training: ~5.8 GB
```

### LoRA Configuration
```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,           # rank of low-rank matrices
    lora_alpha=32,  # scaling factor (α/r = 2.0)
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
# Trainable parameters: ~8.4M  (0.5% of 1.1B total)
```

### Prompt Template (ChatML)
```
<|system|>
You are EduAdvisor, a helpful and knowledgeable university admissions assistant.
You help students understand GPA requirements, admission criteria, and differences
between public and private universities for undergraduate study in the field of education.
<|user|>
{student question}
<|assistant|>
{answer}
```

---

## 🧪 Hyperparameter Experiments

| Exp | LR | Batch | Grad Acc | Eff. Batch | Epochs | LoRA r | LoRA α | Train Loss | Val Loss | BLEU-4 | ROUGE-L | Perplexity | Time |
|-----|----|-------|----------|------------|--------|--------|--------|------------|----------|--------|---------|------------|------|
| 1 | 2e-4 | 2 | 4 | 8 | 1 | 8 | 16 | 1.91 | 1.85 | 0.21 | 0.38 | 6.37 | ~6 min |
| 2 | 1e-4 | 2 | 8 | 16 | 2 | 16 | 32 | 1.72 | 1.62 | 0.29 | 0.51 | 5.06 | ~14 min |
| **3 ★** | **2e-4** | **4** | **4** | **16** | **3** | **16** | **32** | **1.54** | **1.47** | **0.37** | **0.63** | **4.35** | **~20 min** |
| Base (no fine-tuning) | — | — | — | — | — | — | — | — | — | 0.08 | 0.17 | 18.40 | — |

**Experiment 3 is the best configuration.** Larger effective batch size (4 × 4 = 16) with 3 epochs produced the lowest validation loss and best metrics across the board.

---

##  Evaluation Results

Evaluated on 45 held-out test examples using BLEU (`nltk`), ROUGE (`evaluate` library), and perplexity.

| Metric | Base Model | EduAdvisor (Exp 3) | Improvement |
|--------|------------|-------------------|-------------|
| BLEU-1 | 0.12 | 0.52 | **+333%** |
| BLEU-4 | 0.08 | 0.37 | **+363%** |
| ROUGE-1 | 0.22 | 0.71 | **+223%** |
| ROUGE-2 | 0.09 | 0.48 | **+433%** |
| ROUGE-L | 0.17 | 0.63 | **+271%** |
| Perplexity ↓ | 18.40 | 4.35 | **−76%** |

---

##  Qualitative Examples

**GPA question:**
```
User:       What GPA do I need to study Education at a public university?

Base:       GPA requirements vary widely. I recommend checking the specific
            university's website and contacting their admissions office...

EduAdvisor: Most public universities require a minimum GPA of 2.5–3.0 for
            Education programmes. Competitive programmes may expect 3.2 or
            higher. Admissions teams also consider test scores, personal
            statements, and recommendation letters.
```

**Out-of-domain handling:**
```
User:       What's a good recipe for pasta?

EduAdvisor: I'm EduAdvisor — I specialise in university admissions advice.
            I can help with GPA requirements or public vs. private university
            comparisons. Try asking me something in those areas!
```

---

##  Deployment

### Gradio Chat Interface
Launches automatically in Colab with a public URL via `share=True`:
- Multi-turn conversation (last 3 turns injected into prompt context)
- Temperature and max-token sliders
- 6 pre-loaded example questions

### Flask REST API
Exposed publicly via `pyngrok`. Four endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check — confirms API is running |
| POST | `/chat` | Single-turn Q&A |
| POST | `/chat/history` | Multi-turn Q&A with conversation history |
| GET | `/model/info` | Model name, LoRA config, request statistics |

**Example API calls:**
```bash
# Health check
curl <PUBLIC_URL>/

# Single-turn question
curl -X POST <PUBLIC_URL>/chat \
     -H 'Content-Type: application/json' \
     -d '{"question": "What GPA do I need for a public university?",
          "max_tokens": 150, "temperature": 0.3}'

# Multi-turn with history
curl -X POST <PUBLIC_URL>/chat/history \
     -H 'Content-Type: application/json' \
     -d '{"question": "Which one is cheaper?",
          "history": [{"user": "Tell me about public universities.",
                       "assistant": "Public universities are government-funded..."}]}'
```

---

##  Technical Environment

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| PyTorch | 2.1.0+cu128 |
| transformers | 4.44.2 |
| peft | 0.13.0 |
| trl | 0.10.1 |
| accelerate | 0.34.2 |
| datasets | 3.0.1 |
| evaluate | 0.4.3 |
| bitsandbytes | Latest (CUDA 12.8 build) |
| gradio | Latest |
| CUDA | 12.8 |
| GPU | Tesla T4 — 15.6 GB VRAM |
| Platform | Google Colab |

---

##  Known Issues & Fixes

**1. CUDA / bitsandbytes compatibility**
```
Error: CUDA SETUP: Required library version not found: libbitsandbytes_cuda128.so
Fix:   pip install bitsandbytes --upgrade
       Set bnb_4bit_compute_dtype=torch.bfloat16
```

**2. Response leakage** — model appended extra Q&A pairs after its answer  
```
Fix: repetition_penalty=1.3 in model.generate()
     + clean_response() post-processing to strip leaked content
```

**3. Gradient checkpointing RuntimeError**
```
Fix: gradient_checkpointing_kwargs={'use_reentrant': False}
```

---

##  Future Work

- [ ] Expand to 5,000+ unique hand-crafted source pairs
- [ ] Add Retrieval-Augmented Generation (RAG) for real-time policy lookups
- [ ] Multilingual support for international students
- [ ] Permanent deployment on HuggingFace Spaces
- [ ] Human evaluation study with actual prospective students

---



---
Author: JABO Jean Jacques
Youtubelink: https://youtu.be/szDaoUIWHr8

Notebooklink: [JABO Jean Jacques_EduAdvisor_Chatbot_FineTuning_.ipynb](https://colab.research.google.com/drive/1N1Nsu4DPLrwjpzZCAyKMCcvj_s9pfX-a)


