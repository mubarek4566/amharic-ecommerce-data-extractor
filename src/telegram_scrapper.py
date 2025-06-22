from telethon import TelegramClient
import csv
import os

class DataCrawling:
    def __init__(self):
        # Load environment variables
        # load_dotenv('.env')
        # self.api_id = os.getenv('TG_API_ID')
        # self.api_hash = os.getenv('TG_API_HASH')
        # self.phone = os.getenv('phone')

        if not self.api_id or not self.api_hash or not self.phone:
            raise ValueError("API ID, API Hash, or phone is missing!")

        # Initialize the Telegram client
        self.client = TelegramClient('scraping_session', self.api_id, self.api_hash)

    async def scrape_channel(self, channel_username, writer, media_dir):
        """Scrape messages from a single Telegram channel."""
        entity = await self.client.get_entity(channel_username)
        channel_title = entity.title  # Extract the channel's title
        
        async for message in self.client.iter_messages(entity, limit=10000):
            media_path = None
            if message.media:
                # Save media (photo or file) to the specified directory
                filename = f"{channel_username}_{message.id}"
                media_path = os.path.join(media_dir, filename)
                await self.client.download_media(message.media, media_path)

            # Write the scraped data to the CSV file
            writer.writerow([channel_title, channel_username, message.id, message.message, message.date, media_path])

    async def scrape_telegram_data(self, channels):
        """Scrape data from multiple Telegram channels."""
        # Ensure the client is running
        await self.client.start()

        # Create a directory for media files
        media_dir = 'photos'
        os.makedirs(media_dir, exist_ok=True)

        # Open the CSV file and write the header
        with open('telegram_data.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])  # CSV header

            for channel in channels:
                await self.scrape_channel(channel, writer, media_dir)
                print(f"Scraped data from {channel}")

    async def run(self, channels):
        """Run the scraper."""
        async with self.client:  # Use async context manager for the client
            await self.scrape_telegram_data(channels)
