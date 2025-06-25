import pandas as pd
import re
from typing import List, Tuple

class CoNLLLabeler:
    def __init__(self, dataset: pd.DataFrame, labeled_data: pd.DataFrame):
        """
        Initializes the CoNLLLabeler with the dataset and labeled tokens in DataFrame format.
        """
        self.dataset = dataset  # Load dataset
        self.labeled_data = labeled_data  # Labeled tokens in DataFrame format
        self.messages = self.dataset["Filtered_Content"].dropna().tolist()  # Extract non-null messages

        # Convert labeled_data DataFrame to a dictionary for efficient lookup
        self.token_label_map = dict(zip(self.labeled_data["Token"], self.labeled_data["Label"]))

    def tokenize_message(self, message: str) -> List[str]:
        """
        Tokenizes a message into individual words/tokens.
        """
        # Split message into tokens using whitespace and punctuation
        tokens = re.findall(r'\w+|[^\s\w]', message)
        return tokens

    def label_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Labels tokens in a message based on predefined labeled tokens.
        """
        labeled_output = []
        for token in tokens:
            # Get the label for the token, default to "O" if not found
            label = self.token_label_map.get(token, "O")
            labeled_output.append((token, label))
        return labeled_output

    def process_messages(self, num_messages: int) -> List[str]:
        """
        Processes a subset of messages, tokenizes and labels them in CoNLL format.
        """
        labeled_messages = []
        for message in self.messages[:num_messages]:  # Process the specified number of messages
            tokens = self.tokenize_message(message)
            labeled_tokens = self.label_tokens(tokens)

            # Format the labeled tokens in CoNLL format
            labeled_message = "\n".join([f"{token} {label}" for token, label in labeled_tokens])
            labeled_messages.append(labeled_message)
        return labeled_messages

    def save_conll_format(self, labeled_messages: List[str], output_path: str):
        """
        Saves the labeled messages in CoNLL format to a text file.
        """
        with open(output_path, "w", encoding="utf-8") as file:
            # Separate messages with a blank line
            file.write("\n\n".join(labeled_messages))