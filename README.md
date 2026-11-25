# Tanglish Code-Mixed Sentiment & Sarcasm Analyzer

A multilingual Natural Language Processing (NLP) prototype built to classify sentiment in **Tamil-English (Tanglish) code-mixed social media text** using transfer learning with **XLM-RoBERTa**.

> *Developed as a preliminary research prototype for my Winter/Summer Research Internship application at **NIT Tiruchirappalli (NITT)**.*

---

## 📌 Problem Statement
Standard sentiment analysis models trained on monolingual English or pure Tamil script struggle on Indian social media, where users frequently mix Romanized Tamil vocabulary with English grammar, slang, and sarcasm (e.g., *"First half super, aana second half konjam lag"*). Translating code-mixed text into English first often strips away emotional nuance and cultural context.

## 🛠️ Methodology & Pipeline
* **Dataset:** Fine-tuned on the benchmark **DravidianCodeMix (`TamilMixSentiment`)** dataset containing real-world YouTube comments categorized into 5 classes: *Positive, Negative, Neutral, Mixed Feelings,* and *Not-Tamil*.
* **Tokenization:** Used `xlm-roberta-base` SentencePiece subword tokenization (`max_length=64`) to handle non-standardized phonetic spellings in Romanized Tamil without falling back to unknown (`<unk>`) tokens.
* **Model & Training:** Fine-tuned a pretrained **XLM-RoBERTa** sequence classification model (`lr=2e-5`, `weight_decay=0.01`, `AdamW` optimizer) on a T4 GPU using Hugging Face `Transformers` and `PyTorch`.
* **Interactive Interface:** Deployed a live inference interface using **Gradio** with Softmax probability outputs for real-time testing.

## 📊 Prototype Results
* **Validation Accuracy:** ~66.6% (baseline prototype trained on a 2,000-sample subset over 2 epochs)
* **Sample Predictions:**
  * *"Padam semma mass bro, vera level!"* $\rightarrow$ **Positive**
  * *"hi broo epdi irukinga"* $\rightarrow$ **Positive (61%)**
  * *"Waste of money, total disappointment."* $\rightarrow$ **Negative**

## 🚀 Planned Research Extensions (For Internship Scope)
1. **Full Dataset & Class Imbalance Handling:** Train across the full DravidianCodeMix corpus using class-weighted cross-entropy loss to boost **Macro F1-score** on minority classes like *Mixed Feelings*.
2. **Model Benchmarking:** Compare `xlm-roberta-base` against India-centric pretrained models such as **AI4Bharat IndicBERT** and **Google MuRIL**.
3. **Sarcasm & Negation Detection:** Incorporate a multi-task classification head to detect ironic praise and regional sarcasm.

## 💻 Tech Stack
`Python` • `PyTorch` • `Hugging Face Transformers` • `Datasets` • `Scikit-learn` • `Gradio` • `Google Colab`
