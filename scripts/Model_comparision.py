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


class NERModelComparison:
    def __init__(self, dataset_path, model_names, tokenizer_names):
        self.dataset_path = dataset_path
        self.model_names = model_names
        self.tokenizer_names = tokenizer_names
        self.label2id = {}
        self.id2label = {}
        self.best_model = None
        self.best_metrics = {}

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

    def tokenize_and_align_labels(self, examples, tokenizer, model_name):
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
                    if model_name == "xlm-roberta-base":
                        label_ids.append(self.label2id.get(label[word_idx], -1))
                    elif model_name in ["distilbert-base-uncased", "bert-base-multilingual-cased"]:
                        label_ids.append(self.label2id.get(label[word_idx], -1))
                    else:
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

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_names[0])

        unique_labels = set(tag for tags in dataset['train']['ner_tags'] for tag in tags)
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        return dataset

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
    
    def train_and_evaluate(self, model_name, tokenizer_name, tokenized_datasets):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True  # Suppress the weight initialization warning
        )

        # Remove unused columns to avoid warnings
        tokenized_datasets = tokenized_datasets.remove_columns(["tokens", "__index_level_0__", "ner_tags"])
        
        args = TrainingArguments(
            output_dir=f"./results_{model_name}",
            eval_strategy="epoch",  # Updated for deprecation warning
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=f"./logs_{model_name}",
            save_strategy="epoch",
            logging_steps=5,  # Ensure logging happens every 5 steps
            log_level="info",   # Ensure logging is enabled
            report_to="none"   # Avoid reporting to any external system (optional)
        )

        # Use DataCollatorForTokenClassification for batching
        data_collator = DataCollatorForTokenClassification(tokenizer)

        metrics_callback = MetricsCallback()
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,  # Replace `tokenizer` with `data_collator`
            compute_metrics=self.compute_metrics,
            callbacks=[metrics_callback]
        )

        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        eval_results = trainer.evaluate()
        return eval_results, training_time, metrics_callback.metrics

    def compare_models(self):
        best_model = None
        best_f1 = 0
        best_model_name = None
        best_training_time = float("inf")

        for model_name, tokenizer_name in zip(self.model_names, self.tokenizer_names):
            print(f"Training and evaluating {model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            dataset = self.load_and_prepare_data()
            tokenized_datasets = dataset.map(
                lambda examples: self.tokenize_and_align_labels(
                    examples, tokenizer, model_name=model_name
                ),
                batched=True
            )

            eval_results, training_time, epoch_metrics = self.train_and_evaluate(
                model_name, tokenizer_name, tokenized_datasets
            )

            print(f"{model_name} Evaluation Results: {eval_results}")
            print(f"{model_name} Training Time: {training_time} seconds")
            print(f"{model_name} Epoch Metrics:")
            for epoch, metrics in enumerate(epoch_metrics, 1):
                print(f"Epoch {epoch}: {metrics}")

            if eval_results['eval_f1'] > best_f1 and training_time < best_training_time:
                best_f1 = eval_results['eval_f1']
                best_training_time = training_time
                best_model = model_name

        self.best_model = best_model
        self.best_metrics = {'eval_f1': best_f1, 'training_time': best_training_time}
        print(f"Best Model: {self.best_model}")
        print(f"Best Metrics: {self.best_metrics}")