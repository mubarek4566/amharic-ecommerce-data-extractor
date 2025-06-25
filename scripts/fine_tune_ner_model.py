import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import torch  
from sklearn.metrics import precision_score, recall_score, f1_score

class NERFineTuner:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label2id = None  # label2id is not initialized here

    def load_conll_data(self, filepath: str):
        """Loads CoNLL-formatted data into a Pandas DataFrame."""
        sentences, labels = [], []
        sentence, label = [], []

        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip() == '':
                    if sentence:
                        sentences.append(sentence)
                        labels.append(label)
                        sentence, label = [], []
                else:
                    token, tag = line.strip().split()
                    sentence.append(token)
                    label.append(tag)

        return pd.DataFrame({"tokens": sentences, "ner_tags": labels})

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples['tokens'], truncation=True, padding='max_length', is_split_into_words=True, max_length=128)
        labels = []

        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id.get(label[word_idx], -1))  # Ensure a valid label
                else:
                    label_ids.append(-100)  # Ignore subsequent tokens of the same word
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    
    def prepare_data(self, filepath: str):
        """Loads and tokenizes the dataset."""
        df = self.load_conll_data(filepath)

        # Create label mappings from unique tags in the dataset
        unique_labels = set(tag for tags in df['ner_tags'] for tag in tags)
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        # Split the data into train and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        # Convert to Hugging Face datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

        # Tokenize and align labels
        tokenized_datasets = dataset.map(self.tokenize_and_align_labels, batched=True)

        # Initialize model with the correct number of labels
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=len(self.label2id))
        self.model.config.label2id = self.label2id
        self.model.config.id2label = self.id2label

        return tokenized_datasets  

    def train(self, tokenized_datasets, output_dir: str):
        """Fine-tunes the model using the tokenized dataset."""
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_strategy="epoch",
            save_total_limit=2,
            no_cuda=False  # Enable CUDA for actual training
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def compute_metrics(self, p):
        """Computes metrics for evaluation."""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Filter out the -100 labels from the ground truth and predictions
        true_labels = [
            [self.id2label[l] for l in label if l != -100] for label in labels
        ]
        true_predictions = [
            [self.id2label[pred] for (pred, label) in zip(prediction, label) if label != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Flatten the lists of true labels and predictions for evaluation
        flat_true_labels = [item for sublist in true_labels for item in sublist]
        flat_true_predictions = [item for sublist in true_predictions for item in sublist]

        return {
            "precision": precision_score(flat_true_labels, flat_true_predictions, average="weighted"),
            "recall": recall_score(flat_true_labels, flat_true_predictions, average="weighted"),
            "f1": f1_score(flat_true_labels, flat_true_predictions, average="weighted")
        }