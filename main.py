import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the train data
train_df = pd.read_csv('dataset/train.csv')

# Remove irrelevant columns
relevant_columns = ['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_df = train_df[relevant_columns]

# Remove duplicate rows
train_df = train_df.drop_duplicates()

# Remove null values
train_df = train_df.dropna()

# Convert all text to lowercase
train_df['comment_text'] = train_df['comment_text'].str.lower()

# Remove URLs, usernames, and email addresses
def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+|@\S+', '', text)
train_df['comment_text'] = train_df['comment_text'].apply(remove_urls)

# Remove HTML tags
def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)
train_df['comment_text'] = train_df['comment_text'].apply(remove_html_tags)

# Remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))
train_df['comment_text'] = train_df['comment_text'].apply(remove_punctuation)

# Remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    return ' '.join(word for word in text.split() if word not in stop_words)
train_df['comment_text'] = train_df['comment_text'].apply(remove_stopwords)

# Lemmatize text
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())
train_df['comment_text'] = train_df['comment_text'].apply(lemmatize)

# Save cleaned data to a new file
train_df.to_csv('cleaned_train_data.csv', index=False)