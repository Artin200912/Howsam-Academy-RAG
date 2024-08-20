import os
import pandas as pd
from utils import get_embedding
import openai


OPENAI_API_KEY = ''
client = openai.OpenAI(api_key=OPENAI_API_KEY)

df = pd.read_csv('new.csv')
df['embedding'] = df['conv'].apply(lambda x: get_embedding(x))
