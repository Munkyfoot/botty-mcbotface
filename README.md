# Customizable Discord Bot with OpenAI Integration
This is an easily customizable Discord bot that uses OpenAI's APIs to generate text and images. This is a work in progress, and I will be adding and refining features as I go. Current features include:
- Chat capabilities powered by OpenAI's Chat Completion API
- Separate, locally stored chat history for each channel/DM 
- DM Whitelist, allowing you to specify which users can use the bot in DMs
- Commands/Functions:
    - Image generation powered by OpenAI's DALL-E 3 API 
    - Wikipedia API integration, allowing you to search Wikipedia from Discord
    - Available via slash commands as well as autonomous usage by the bot
- Customizable:
    - Chat model, response length, and max history length
    - Bot name, system prompt, introduction message
    - Timezone for displaying dates/times
    - Random messages (bot will autonomously send a message if a channel has been inactive)

## Before You Begin 
1. Create a Discord bot and get your token. You can find a guide [here](https://discordpy.readthedocs.io/en/stable/discord.html).
2. Create an OpenAI account and get your API key. You can find a guide [here](https://platform.openai.com/docs/quickstart/account-setup).

## Setup
1. Clone the repository.
```bash
git clone https://github.com/Munkyfoot/botty-mcbotface
```
2. Install the required packages. A virtual environment is recommended, but not required. 
```bash 
pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and fill in the required information.
```bash
cp .env.example .env
```

4. Copy `settings.json.example` to `settings.json`. Update the settings as desired.
```bash
cp settings.json.example settings.json
```
Default Settings:
```json
{
    "model": "gpt-4-turbo-preview", // OpenAI model to use for chatting
    "vision_model": "gpt-4-vision-preview", // OpenAI model to use for image analysis
    "max_output_token_count": 512, // Max tokens in response 
    "max_input_token_count_base": 4096, // Max total tokens in history - each model has a different upper limit you should not exceed
    "bot_name": "Botty McBotface", // How the bot will refer to itself
    // This is the message the bot will use to introduce itself. It also allows you to have a bit more control over the bot's personality.
    "bot_introduction": "Hi! I'm Botty McBotface, a bot powered by OpenAI's API. I'm still learning, so please be patient with me. I'm not perfect, but I'm trying my best!", 
    // This serves as the base of the system message and is used to tell the bot who it is and how it should respond.
    "system_message_base": "You are Botty McBotface, a bot powered by OpenAI's API. You are a friendly, helpful bot that is always willing to chat and help out. You are not perfect, but you are trying your best.",
    "suppress_emojis": false, // Whether or not to suppress emojis in the bot's responses
    "timezone": "US/Pacific", // Timezone for displaying dates/times
    "random_messages": true // Whether or not the bot should send random messages when a channel is inactive
}
```

## Usage
1. Run the bot.
```bash
python bot.py  
```
2. Invite the bot to your Discord server. Make sure it has the necessary permissions:
   - Send Messages 
   - Read Message History
   - Use Slash Commands

## Commands
In the future, commands will be more customizable. For now, here are the built-in commands:
- `/search [query] [limit]` - Searches Wikipedia for the given query.
- `/read [result index]` - Reads the Wikipedia article at the given index.
- `/image [prompt] [detailed] [wide] [realism]` - Generates an image from a prompt using the DALL-E API.
- `/forget` - Forgets everything. 
- `/sleep` - Puts the bot to sleep.
- `/wake` - Wakes the bot up.
- `/help` - Shows the available commands.
