import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

class Visualization:
    def __init__(self):
        self.df = {}
    # Visualize missing values
    def Visualize_missing_values(self, df):
        msno.matrix(df)
        plt.title("Missing Values Matrix")
        plt.show()
    # Message count per username
    def message_count_per_username(self, df_cleaned):
        plt.figure(figsize=(8, 5))
        sns.countplot(y='Channel_Username', data=df_cleaned, order=df_cleaned['Channel_Username'].value_counts().index)
        plt.title("Message Count by Channel Username")
        plt.xlabel("Count")
        plt.ylabel("Channel Username")
        plt.show()
    # Messages over time (trend)
    def messages_over_time_trend(self,df_cleaned):
        plt.figure(figsize=(12, 6))
        df_cleaned['Date'].value_counts().sort_index().plot(kind='line', marker='o', color='skyblue')
        plt.title("Messages Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Messages")
        plt.grid()
        plt.show()
    # Distribution of Message IDs
    def Distribution_of_Message_IDs(self, df_cleaned):
        plt.figure(figsize=(10, 5))
        sns.histplot(df_cleaned['MESSAGE_ID'], kde=True, bins=10, color='green')
        plt.title("Distribution of Message IDs")
        plt.xlabel("Message ID")
        plt.ylabel("Frequency")
        plt.show()
    # Distribution of Message
    def Distribution_of_Content(self, df_cleaned):
        plt.figure(figsize=(10, 5))
        sns.histplot(df_cleaned['Content'], kde=True, bins=10, color='green')
        plt.title("Distribution of Message")
        plt.xlabel("Message")
        plt.ylabel("Frequency")
        plt.show()
    
    # Word count in 'Content' (if available)
    def word_count_in_content(self,df_cleaned):
        df_cleaned['Content_Word_Count'] = df_cleaned['Filtered_Content'].apply(lambda x: len(str(x).split()))
        plt.figure(figsize=(10, 5))
        sns.histplot(df_cleaned['Content_Word_Count'], bins=10, kde=True, color='purple')
        plt.title("Content Word Count Distribution")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        plt.show()