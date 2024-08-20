import openai
import pandas as pd
import sys
import re
from telethon import TelegramClient
import os

OPENAI_API_KEY = ''
client = openai.OpenAI(api_key=OPENAI_API_KEY)

emoji_pattern = re.compile(
    u'[\U0001F600-\U0001F64F'  # emoticons
    u'\U0001F300-\U0001F5FF'  # symbols & pictographs
    u'\U0001F680-\U0001F6FF'  # transport & map symbols
    u'\U0001F700-\U0001F77F'  # alchemical symbols
    u'\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
    u'\U0001F800-\U0001F8FF'  # Supplemental Arrows-C
    u'\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
    u'\U0001FA00-\U0001FA6F'  # Chess Symbols
    u'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
    u'\U00002702-\U000027B0'  # Dingbats
    u'\U000024C2-\U0001F251'  # Enclosed Characters
    ']+', re.UNICODE)

url_pattern = re.compile(r'https?://\S+|www\.\S+|@\w+', re.UNICODE)

def clean_text(text):
    # Remove emojis
    text = emoji_pattern.sub(r'', text)
    # Remove URLs and @mentions
    text = url_pattern.sub(r'', text)
    # Remove any remaining special characters or excessive whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())  # Remove extra spaces
    return text

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"  # Choose the model you want to use
    )
    return response.data[0].embedding

