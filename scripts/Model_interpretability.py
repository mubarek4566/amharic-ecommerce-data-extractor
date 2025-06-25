import lime
from lime import lime_text
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import pandas as pd
import torch.nn.functional as F

class NERModelExplainabilityLIME:
    def __init__(self, model_name, tokenizer_name):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)

    def explain_with_lime(self, input_text):
        # Define a prediction function compatible with LIME
        def predict_fn(texts):
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: (batch_size, seq_len, num_classes)

            # Aggregate logits to create a single prediction per sample
            logits_mean = logits.mean(dim=1)  # Shape: (batch_size, num_classes)
            probs = F.softmax(logits_mean, dim=1)  # Convert to probabilities
            return probs.detach().numpy()

        # Create a LIME text explainer
        explainer = lime_text.LimeTextExplainer(class_names=["O", "B-PER", "I-PER"])  # Example class names for NER

        # Explain a prediction
        explanation = explainer.explain_instance(
            input_text,
            predict_fn,
            num_features=5,
            num_samples=500
        )

        return explanation.as_list()

    def identify_difficult_cases(self, dataset):
        difficult_cases = []

        for input_text in dataset:
            # Ensure input_text is a string
            input_text = str(input_text)

            # Tokenize the input
            inputs = self.tokenizer(input_text, truncation=True, padding=True, return_tensors="pt")
            outputs = self.model(**inputs)
            predicted_labels = outputs.logits.argmax(-1).squeeze().cpu().numpy()

            # Simulated ground truth labels (example)
            true_labels = [0] * len(predicted_labels)

            if not np.array_equal(predicted_labels[:len(true_labels)], true_labels):
                difficult_cases.append({
                    "input": input_text,
                    "true_labels": true_labels,
                    "predicted_labels": predicted_labels[:len(true_labels)]
                })

        return difficult_cases


    def generate_report(self, difficult_cases):
        """
        Generate a report for the identified difficult cases using LIME explanations.
        """
        report = []

        for case in difficult_cases:
            # Get LIME explanation for the input text
            lime_explanation = self.explain_with_lime(case["input"])

            report.append({
                "input": case["input"],
                "true_labels": case["true_labels"],
                "predicted_labels": case["predicted_labels"],
                "lime_explanation": lime_explanation
            })

        # Save report to a CSV file
        pd.DataFrame(report).to_csv("lime_difficult_cases_report.csv", index=False)

        return report