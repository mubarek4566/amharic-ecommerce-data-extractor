from transformers import AutoTokenizer
import tensorflow as tf
import pandas as pd
import logging
import sqlite3
import re
import unicodedata
import os
class Preprocess:
    def __init__(self):
        self.df = {}
        # Load a transformer tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def normalize_amharic_text(self, text):
        """
        Normalize Amharic text by handling diacritics, punctuation, and special marks.
        """
        if not text:
            return None

        # Remove diacritics and normalize similar characters
        text = unicodedata.normalize("NFD", text)  # Decompose Unicode
        text = re.sub(r'[\u135D-\u135F]', '', text)  # Remove diacritics (e.g., ፝, ፞, ፟)
        text = unicodedata.normalize("NFC", text)  # Recompose to canonical form

        # Replace Ethiopian punctuation with standard punctuation
        text = text.replace('፡', ' ').replace('።', '.').replace('፣', ',')

        # Remove non-Amharic characters (optional)
        text = re.sub(r'[^\u1200-\u137F\s.,!?]', '', text)  # Keep Amharic and basic punctuation

        # Lowercase and strip leading/trailing whitespace
        text = text.strip().lower()

        return text

    def preprocess_text(self, text):
        """
        Preprocess Amharic text using normalization and transformer-based tokenization.
        Includes positional tokenization.
        """
        if not text:
            return None

        # Normalize Amharic text
        text = self.normalize_amharic_text(text)

        # Tokenize with the transformer tokenizer
        tokenized = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="tf",  # TensorFlow tensors
        )

        # Generate positional encodings (position indices for each token)
        # Positional IDs range from 0 to the length of tokens - 1
        position_ids = tf.range(start=0, limit=tf.shape(tokenized["input_ids"])[1])

        # Convert tensors to numpy arrays and then to lists
        tokenized = {
            key: value.numpy().tolist() if isinstance(value, tf.Tensor) else value
            for key, value in tokenized.items()
        }
        tokenized["position_ids"] = position_ids.numpy().tolist()  # Add position IDs as a list

        return tokenized

    def tokenize_dataframe(self, df, message_column="Message"):
        """
        Tokenize the messages in a DataFrame and add tokenization outputs as new columns.
        Includes positional tokenization.
        """
        input_ids = []
        attention_masks = []
        token_type_ids = []
        position_ids = []

        for text in df[message_column]:
            if pd.notnull(text):  # Skip null messages
                tokenized_output = self.preprocess_text(text)
                if tokenized_output:
                    input_ids.append(tokenized_output["input_ids"])
                    attention_masks.append(tokenized_output["attention_mask"])
                    token_type_ids.append(tokenized_output.get("token_type_ids", [[None]]))
                    position_ids.append(tokenized_output["position_ids"])
                else:
                    input_ids.append(None)
                    attention_masks.append(None)
                    token_type_ids.append(None)
                    position_ids.append(None)
            else:
                input_ids.append(None)
                attention_masks.append(None)
                token_type_ids.append(None)
                position_ids.append(None)

        # Add tokenization outputs as new columns
        df["input_ids"] = input_ids
        df["attention_mask"] = attention_masks
        df["token_type_ids"] = token_type_ids
        df["position_ids"] = position_ids  # Add position IDs
        return df

    def clean_structure(self, df):
        """
        Restructure messages into a clean DataFrame format.
        """
        structured_data = []
        # Convert DataFrame to a list of dictionaries
        if isinstance(df, pd.DataFrame):
            df = df.to_dict(orient='records')

        for message in df:
            structured_data.append({
                'Channel Title': message['Channel Title'],
                'Channel Username': message['Channel Username'],
                'ID': message['ID'],
                'Date': message['Date'],
                'Media Path': message['Media Path'],
                'Content': message.get('Message', ''),
                'input_ids': message.get('input_ids', []),
                'attention_mask': message.get('attention_mask', []),
                'token_type_ids': message.get('token_type_ids', []),
                'position_ids': message.get('position_ids', []),
            })

        return pd.DataFrame(structured_data)

    def store_preprocessed_data(self, df):
        """
        Store preprocessed data into a SQLite database, ensuring compatibility with SQLite data types.
        If the table already exists, it is dropped and recreated to store new data.
        """
        # Replace NaN with None
        df = df.where(pd.notnull(df), None)

        # Ensure all required columns are properly formatted
        for column in ['Media Path', 'Content', 'input_ids', 'attention_mask', 'token_type_ids','position_ids']:
            if column in df.columns:
                df[column] = df[column].astype(str)

        # Format the 'Date' column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')

        # Create the database directory if it doesn't exist
        database_dir = os.path.join(os.getcwd(), '../../10_X_data', 'database')
        os.makedirs(database_dir, exist_ok=True)
        db_path = os.path.join(database_dir, 'telegram_data.db')

        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Drop the table if it exists
        cursor.execute('DROP TABLE IF EXISTS telegram_messages')

        # Recreate the table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS telegram_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            MESSAGE_ID INTEGER,
            Channel_Title TEXT,
            Channel_Username TEXT,
            Date TEXT,
            Media_Path TEXT,
            Content TEXT,
            input_ids TEXT,
            attention_mask TEXT,
            token_type_ids TEXT,
            position_ids TEXT
        )
        ''')

        # Insert the new data
        for _, row in df.iterrows():
            cursor.execute('''
            INSERT INTO telegram_messages (MESSAGE_ID, Channel_Title, Channel_Username, Date, Media_Path, Content, input_ids, attention_mask, token_type_ids, position_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,?)
            ''', (
                row['ID'],
                row['Channel Title'],
                row['Channel Username'],
                row['Date'],
                row['Media Path'],
                row['Content'],
                row['input_ids'],
                row['attention_mask'],
                row['token_type_ids'],
                row['position_ids']
            ))

        # Commit the changes and close the connection
        conn.commit()
        conn.close()


    def ReadSavedDate(self, db_name, table_name):
        """
        Read saved data from the SQLite database.
        """
        database_dir = os.path.join(os.getcwd(), '../../10_X_data', 'database')
        db_path = os.path.join(database_dir, 'telegram_data.db')
        conn = sqlite3.connect(db_path)

        query = f'''
        SELECT * FROM {table_name}
        '''

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df
    
    # Define the filter function
    def filter_amharic_text(self, content):
        if not content:  # Check for None or empty values
            return None
        # Regex to match Amharic characters and numbers
        amharic_pattern = r'[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\u1E00-\u1EFF0-9]+'
        matches = re.findall(amharic_pattern, content)
        return " ".join(matches) if matches else None

    # ===== Data Cleaning =====
    # Handle missing values
    def handlling_missing_values(self, df):
        print("\n--- Handling Missing Values ---")
        # Drop rows with all NaN values
        df_cleaned = df.dropna(how = 'all')
        # Replace missing values in specific columns with placeholders
        df_cleaned['Media_Path'] = df_cleaned['Media_Path'].fillna('No path')
        df_cleaned['Content'] = df_cleaned['Content'].fillna("No Content")
        df_cleaned['input_ids'] = df_cleaned['input_ids'].fillna("None")
        df_cleaned['attention_mask'] = df_cleaned['attention_mask'].fillna("None")
        df_cleaned['token_type_ids'] = df_cleaned['token_type_ids'].fillna("None")
        df_cleaned['position_ids'] = df_cleaned['position_ids'].fillna("None")
        return df_cleaned
    # Handle duplicate data
    def check_and_handlling_duplicate_values(self, df):
        print("\n--- Checking Duplicates ---")
        duplicates = df.duplicated()
        print(f"Number of duplicate rows: {duplicates.sum()}")
        if duplicates.sum() > 0:
            df = df.drop_duplicates()
        else:
            df = df
        return df 