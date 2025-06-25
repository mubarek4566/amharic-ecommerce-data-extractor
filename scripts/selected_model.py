from transformers import Trainer, TrainingArguments
import time
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from transformers import DataCollatorForTokenClassification
from transformers.trainer_callback import TrainerCallback
import pandas as pd
import numpy as np

class MetricsCallback(TrainerCallback):
    """Custom callback to log metrics at the end of each epoch."""
    def __init__(self):
        self.metrics = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.metrics.append(logs)

class NERModelExplainability:
    def __init__(self, dataset_path, model_name, tokenizer_name):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.label2id = {}
        self.id2label = {}

    def load_conll_data(self, filepath):
        data = []
        with open(filepath, "r") as file:
            tokens, labels = [], []
            for line in file:
                if line.strip():
                    token, label = line.strip().split()
                    tokens.append(token)
                    labels.append(label)
                else:
                    if tokens:
                        data.append({"tokens": tokens, "ner_tags": labels})
                        tokens, labels = [], []
        return pd.DataFrame(data)

    def tokenize_and_align_labels(self, examples, tokenizer):
        tokenized_inputs = tokenizer(
            examples['tokens'],
            truncation=True,
            padding='max_length',
            is_split_into_words=True,
            max_length=128
        )
        labels = []

        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id.get(label[word_idx], -1))
                else:
                    label_ids.append(-100)

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs['labels'] = labels
        return tokenized_inputs

    def load_and_prepare_data(self):
        df = self.load_conll_data(self.dataset_path)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        unique_labels = set(tag for tags in dataset['train']['ner_tags'] for tag in tags)
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        tokenized_datasets = dataset.map(
            lambda examples: self.tokenize_and_align_labels(examples, tokenizer),
            batched=True
        )

        return tokenized_datasets

    def compute_metrics(self, pred):
        predictions, labels = pred
        predictions = predictions.argmax(-1)
        true_labels = [[self.id2label[l] for l in label if l != -100] for label in labels]
        pred_labels = [[self.id2label[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]

        precision, recall, f1, _ = precision_recall_fscore_support(
            [l for sublist in true_labels for l in sublist],
            [l for sublist in pred_labels for l in sublist],
            average="weighted"
        )
        return {"precision": precision, "recall": recall, "eval_f1": f1}

    def train_and_save_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )

        dataset = self.load_and_prepare_data()

        # Check dataset structure
        print("Sample training data:", dataset["train"][0])

        dataset = dataset.remove_columns(["tokens", "__index_level_0__", "ner_tags"])

        args = TrainingArguments(
            output_dir=f"./results_{self.model_name}",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=f"./logs_{self.model_name}",
            save_strategy="epoch",
            logging_steps=5
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)
        metrics_callback = MetricsCallback()

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[metrics_callback]
        )

        print(f"Training {self.model_name}...")
        try:
            trainer.train()
        except Exception as e:
            print("Error during training:", str(e))

        print("Saving model and tokenizer for explainability...")
        model.save_pretrained(f"./saved_model_{self.model_name}")
        tokenizer.save_pretrained(f"./saved_model_{self.model_name}")
        print(f"Model and tokenizer saved to ./saved_model_{self.model_name}")