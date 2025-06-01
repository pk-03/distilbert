
# Hindi Sentiment Classification using DeBERTa

This project fine-tunes the `DistilBERT` transformer model for sentiment classification on Hindi text. The goal is to classify Hindi reviews into one of three sentiment classes: **positive**, **negative**, or **neutral**.

---
## Model

DistilBERT is a smaller, faster, and lighter version of BERT (Bidirectional Encoder Representations from Transformers) developed by Hugging Face. It was created to retain 95% of BERT’s performance while being:
- 40% smaller
- 60% faster
- Almost as accurate


## 🧠 Model Architecture

- **Model**: [DistilBERT](https://huggingface.co/distilbert)
- **Tokenizer**: AutoTokenizer from Hugging Face
- **Framework**: PyTorch with HuggingFace Transformers & Trainer API

---

## 📁 Dataset Format

The dataset should be a `.csv` or `.tsv` file with the following two columns:

- `Reviews`: Hindi review text.
- `labels`: Sentiment labels (e.g., `"positive"`, `"negative"`, `"neutral"`)

Example:

| Reviews                  | labels   |
|--------------------------|----------|
| यह फोन बहुत अच्छा है।   | positive |
| बैटरी खराब है।          | negative |
| यह एक औसत उत्पाद है।     | neutral  |

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/pk-03/distilbert.git
cd distilbert
```

### 2. Install Dependencies
Make sure Python 3.8+ is installed.

```bash
pip install transformers torch pandas scikit-learn indic-nlp-library
```

---

## 🏋️‍♀️ Training the Model

### 1. Load and preprocess your dataset
Ensure your `DataFrame` has `Reviews` and `labels` columns or you can access the datasets given in [Datasets](https://github.com/pk-03/Data-Augmentation-and-Datasets.git).

```python
data = pd.read_csv("your_dataset.csv")  # or .tsv
```


### 2. Run the training script
The script:
- Encodes text with DistilBERT tokenizer
- Prepares datasets
- Trains using HuggingFace `Trainer`
- Evaluates on a held-out test set

```python
python train_sentiment_model.py
```

> **Note:** You can customize training parameters such as `batch_size`, `epochs`, and `learning_rate` inside the script.

---

## 🧪 Evaluation

After training, the model is evaluated using the following metrics:

- Accuracy
- F1 Score (weighted)
- Precision (weighted)
- Recall (weighted)

The best model (based on F1 score) is automatically saved and restored for evaluation.

---

## 🧠 Inference

Use the `classify_text` function to predict sentiment for new Hindi inputs:

```python
sample = "यह उत्पाद बहुत बेकार है।"
predicted_label = classify_text(sample)
print(f"Predicted Sentiment: {predicted_label}")
```

---

## 💾 Saving and Loading the Model

To save the fine-tuned model and tokenizer:

```python
model.save_pretrained("./distilbert")
tokenizer.save_pretrained("./distilbert")
```

To load it again later:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./distilbert")
tokenizer = AutoTokenizer.from_pretrained("./distilbert")
```

---

## 📊 Results

| Metric    | Value |
|-----------|-------|
| Accuracy  |  84.27|
| F1 Score  |  84.31|
| Precision |  84.45|
| Recall    |  84.3|
| loss | 1.328  |
| samples_per_second | 144.418 |
|epoch| 27 |

<!-- Test Results: {'eval_loss': 1.3282955884933472, 'eval_accuracy': 0.842741935483871, 'eval_f1': 0.8431236499020262, 'eval_precision': 0.8445560638413548, 'eval_recall': 0.842741935483871, 'eval_runtime': 10.3034, 'eval_samples_per_second': 144.418, 'eval_steps_per_second': 18.052, 'epoch': 27.0} -->


---

## 📌 Future Work

- Add support for class imbalance handling.
- Incorporate validation split for hyperparameter tuning.
- Integrate IndicNLP for preprocessing (normalization, sentence splitting).

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---

## 🙏 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Microsoft DeBERTa](https://github.com/microsoft/DeBERTa)
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library)

---

## ✨ Contact

For queries, reach out to [pranitarora074@gmail.com] or create an issue.
