import openai
import pandas as pd
import sys
from telethon import TelegramClient
import asyncio
import os
from utils import clean_text


TELETHON_API_ID = '' 
TELETHON_HASH_ID = ''
TELETHON_PHONE_NUMBER = '+'
TELETHON_GROUP_ID = ''
OPENAI_API_KEY = ''

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
telethon_client = TelegramClient('session', TELETHON_API_ID, TELETHON_HASH_ID)

async def fetch_messages():
    try:
        await telethon_client.start(TELETHON_PHONE_NUMBER)
        print('Client started.')

        group = await telethon_client.get_entity(int(TELETHON_GROUP_ID))
        conversations = []

        print("Fetching group messages.")
        message_count = 0

        # Iterate through all messages in the group
        async for message in telethon_client.iter_messages(group, limit=50):
            try:
                # Ensure the message has text before processing
                if message.text:
                    message_count += 1

                    # Print and update the message count dynamically
                    sys.stdout.write(f"\rFetching message {message_count}...")
                    sys.stdout.flush()

                    replies = []

                    # Ensure message has replies and the message ID is valid
                    if message.replies and message.id:
                        try:
                            async for reply in telethon_client.iter_messages(group, reply_to=message.id):
                                try:
                                    cleaned_reply = clean_text(reply.text.strip())

                                    replies.append({
                                        'date': reply.date,
                                        'text': cleaned_reply
                                    })
                                except Exception as e:
                                    print(f"\nError processing reply: {e}")
                                    continue  # Skip this reply and continue with the next

                            if replies:
                                cleaned_message = clean_text(message.text.strip())
                                conversation = [f'-> {cleaned_message}']

                                replies.sort(key=lambda r: r['date'])

                                for reply in replies:
                                    conversation.append(f"-> {reply['text']}")

                                conversations.append("\n".join(conversation))

                        except Exception as e:
                            print(f"\nError processing replies: {e}")
                            continue  # Skip to the next message if there's an error fetching replies

            except Exception as e:
                print(f"\nError processing message: {e}")
                continue  # Skip to the next message if there's an error processing the current message

        # Once done, move to the next line in the console
        print("\nAll messages fetched.")

        # Handle the DataFrame creation and file saving
        try:
            df = pd.DataFrame({'conv': conversations})
            df.to_csv('conversations.csv', index=False)
            print('Done')
        except Exception as e:
            print(f"\nError saving conversations to CSV: {e}")

    except Exception as e:
        print(f"\nError starting the client or fetching group: {e}")
        
if __name__ == "__main__":
    # Run the fetch_messages function using asyncio
    asyncio.run(fetch_messages())