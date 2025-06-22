import os
import sys
import pandas as pd
sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('../Scripts'))
class DataLoad:
    def __init__(self):
        self.df = {}
    
    def read_csv_txt_data(self):
        # Get the directory of the current script (data_loader.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(script_dir, "../src/")  # Path to the src folder

        # Define file paths
        file_paths = [
            os.path.join(src_dir, 'zemen.csv'),
            os.path.join(src_dir, 'Shewabrand.csv'),
            os.path.join(src_dir, 'gebeyaadama.csv'),
            os.path.join(src_dir, 'AwasMart.csv'),
            os.path.join(src_dir,'helloomarketethiopia.csv'),
            os.path.join(src_dir, 'Leyueqa.csv'),
            os.path.join(src_dir,'MerttEka.csv')
        ]

        # Load data
        dataframes = []
        for file_path in file_paths:
            if file_path.endswith('.txt'):  # Handle the .txt file
                df = pd.read_csv(file_path, delimiter=',', encoding='utf-8')  # Adjust delimiter if necessary
            else:  # Handle .csv files
                df = pd.read_csv(file_path, encoding='utf-8')
            dataframes.append(df)

        # Merge all dataframes
        result = pd.concat(dataframes, ignore_index=True)
        return result
    
    def lalebed_file(self):
        """
        Reads the x.txt file from the src folder and processes it as per CoNLL format (entity tagging).
        Returns the data as a pandas DataFrame.
        """
        # Get the directory of the current script (data_loader.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(script_dir, "../src/")  # Path to the src folder

        # Define the file path for labeled_telegram_product_price_location.txt
        x_txt_path = os.path.join(src_dir, 'labeled_telegram_product_price_location.txt')

        # Check if the file exists
        if not os.path.exists(x_txt_path):
            raise FileNotFoundError(f"The file labeled_telegram_product_price_location.txt does not exist in the src folder: {x_txt_path}")

        # Read the x.txt file
        # Assuming each line contains tokens and labels separated by whitespace or tabs
        data = []
        with open(x_txt_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:  # Ignore blank lines
                    tokens = line.split()  # Split by whitespace
                    if len(tokens) == 2:  # Ensure each line contains a token and a label
                        token, label = tokens
                        data.append({"Token": token, "Label": label})
                    else:
                        raise ValueError(f"Invalid line format: {line}")

        # Convert the processed data into a DataFrame
        df = pd.DataFrame(data)
        return df