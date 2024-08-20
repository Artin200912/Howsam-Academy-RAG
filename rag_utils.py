import os
import openai
import pandas as pd
from scipy import spatial
import tiktoken

OPENAI_API_KEY = ''
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

embedding_model = 'text-embedding-ada-002'
gpt_model = 'gpt-4o-mini'

def strings_ranked_by_relatedness(query: str, df: pd.DataFrame, relatedness_function = lambda x, y: 1 - spatial.distance.cosine(x, y), top_n: int = 3):

  """
  This function returns a list of strings and their corresponding relatedness scores. sorted from the most relevant to the least
  """

  q_embedding = openai_client.embeddings.create(
      input = query,
      model = embedding_model
  ).data[0].embedding

  strings_and_relatedness = [
      (row['conv'], relatedness_function(q_embedding, row['embedding']))
      for i, row in df.iterrows()
  ]

  strings_and_relatedness.sort(key = lambda x: x[1], reverse = True)

  strings, relatednesses = zip(*strings_and_relatedness)

  return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = gpt_model):
  """
  This function returns the number of tokens in a text.
  """
  encoding = tiktoken.encoding_for_model(model)
  num_tokens = len(encoding.encode(text))
  return num_tokens

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
):
  """
  This returns a message ready for GPT. 
  """
  strings, relatedness = strings_ranked_by_relatedness(
      query, 
      df 
  )

  introduction = 'Use the information in the conversations below that are extracted from the chats of a group so that you can answer the question that is asked to you. Note that the dates of the conversations are different, so answer very carefully. split your answer to 3 parts. 1. the final answer 2. why did you decide to produce this answer(based on which conversation.) 3. the conversation you produced your answer based on it. title the 3 parts with 1, 2, 3. please write them in a format that is readable by a human  .If the answer cannot be found in the text below, write "I could not find an answer.'
  
  question = f'\n\nQuestion: {query}'

  for string in strings:
    next_conv = f'\n\nConversation: """{string}"""'

    if (
        num_tokens(introduction + next_conv + question, model=model)
        > token_budget
    ):
      break
    else:
      introduction += next_conv

  return introduction + question


def ask(
    query: str,
    df: pd.DataFrame,
    model: str = gpt_model,
    token_budget: int = 5000,
    print_message: bool = False,
):
  """
  Answers a query using GPT and a df of relevant text and embeddings
  """

  message = query_message(query, df, model, token_budget)

  if print_message:
    print(message)

  messages = [
      {'role':'system', 'content': 'You answer question in farsi about The Howsam Academy, An AI Academy with multiple courses. you will use the info you will be given to answer questions.'},
      {'role':'user', 'content': message},
  ]

  response = openai_client.chat.completions.create(
    model = model,
    messages = messages
  ).choices[0].message.content

  return response