import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def data_cleaning(comments_df):

    # Remove irrelevant columns

    print("Removing irrelevant columns...")
    irrelevant_columns = ["id"]
    comments_df = comments_df.drop(irrelevant_columns, axis=1)
    print(type(comments_df))

    # Remove duplicate rows

    print("Removing duplicate rows...")
    comments_df = comments_df.drop_duplicates()
    print(comments_df.shape)

    # Remove null values

    print("Removing null values...")
    comments_df = comments_df.dropna()
    print(comments_df.shape)

    # Remove URLs, usernames, and email addresses

    print("Removing URLs, usernames, and email addresses...")
    def remove_urls(text):
        return re.sub(r'http\S+|www\.\S+|@\S+', '', text)
    comments_df['comment_text'] = comments_df['comment_text'].apply(remove_urls)
    print(comments_df.head)

    # Remove HTML tags

    print("Removing HTML tags...")
    def remove_html_tags(text):
        return re.sub(r'<.*?>', '', text)
    comments_df['comment_text'] = comments_df['comment_text'].apply(remove_html_tags)
    print(comments_df.head)

    # Remove punctuation

    print("Removing punctuation...")
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    comments_df['comment_text'] = comments_df['comment_text'].apply(remove_punctuation)
    print(comments_df.head)

    # Remove stopwords

    print("Removing stopwords...")
    stopwords_data = set(stopwords.words('english'))
    def remove_stopwords(text):
        return ' '.join(word for word in text.split() if word not in stopwords_data)
    comments_df['comment_text'] = comments_df['comment_text'].apply(remove_stopwords)
    print(comments_df.head)

    # Lemmatizing text

    print("Lemmatizing text...")
    lemmatizer = WordNetLemmatizer()
    def lemmatize(text):
        return ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    comments_df['comment_text'] = comments_df['comment_text'].apply(lemmatize)
    print(comments_df.head)

    # Remove Special characters
    print("Removing Special characters...")
    def remove_special_characters(text):
        return re.sub(r'[^\w\s]', '', text)
    comments_df['comment_text'] = comments_df['comment_text'].apply(remove_special_characters)
    print(comments_df.head)

    # Remove digits

    print("Removing digits...")
    def remove_digits(text):
        return re.sub(r'[0-9]', '', text)
    comments_df['comment_text'] = comments_df['comment_text'].apply(remove_digits)
    print(comments_df.head)

    # Convert all text to lowercase
    
    print("Converting all text to lowercase...")
    comments_df['comment_text'] = comments_df['comment_text'].str.lower()
    print(comments_df.head)

    # Save cleaned data to a new file
    comments_df.to_csv('cleaned_data.csv', index=False)

    return pd.read_csv('cleaned_data.csv')
