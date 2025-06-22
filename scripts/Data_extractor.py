from telethon.sync import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
import os, json
from datetime import datetime

# Telegram API credentials
api_id = YOUR_API_ID
api_hash = 'YOUR_API_HASH'
phone = 'YOUR_PHONE_NUMBER'  # Needed only for first login

# Output folder
output_dir = 'data/raw'
os.makedirs(output_dir, exist_ok=True)

# List of target channels
channels = ['shageronlinestore', 'ethiodeals', 'betochecommerce', 'addisfashionshop', 'tech4ethio']

# Create Telegram client
client = TelegramClient('ethio_ecomm_session', api_id, api_hash)

async def fetch_messages():
    await client.start()
    for channel in channels:
        print(f"Fetching from @{channel}")
        messages = []
        async for message in client.iter_messages(channel, limit=500):
            if message.message:
                data = {
                    'channel': channel,
                    'id': message.id,
                    'text': message.message,
                    'date': message.date.isoformat(),
                    'sender_id': getattr(message.sender_id, 'user_id', None),
                    'has_media': message.media is not None
                }
                messages.append(data)
        with open(f"{output_dir}/{channel}_messages.json", "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

with client:
    client.loop.run_until_complete(fetch_messages())
