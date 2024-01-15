"""Customizable Discord Bot integrated with OpenAI's API."""

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from datetime import timedelta
from io import BytesIO

import discord
import tiktoken
from discord import app_commands
from dotenv import load_dotenv
from openai import AsyncOpenAI

from utils import WikiAPI

load_dotenv()


@dataclass
class BotSettings:
    """A dataclass for storing bot settings."""

    model: str = "gpt-4"
    max_output_token_count: int = 512
    max_input_token_count_base: int = 4096
    bot_name: str = "Botty McBotface"
    bot_introduction: str = "Hi! I'm Botty McBotface, a bot powered by OpenAI's API. I'm still learning, so please be patient with me. I'm not perfect, but I'm trying my best!"
    system_message_base: str = "You are Botty McBotface, a bot powered by OpenAI's API. You are a friendly, helpful bot that is always willing to chat and help out. You are not perfect, but you are trying your best."
    suppress_emojis: bool = False


class BotGPT:
    def __init__(self):
        self.intents = discord.Intents.default()
        self.intents.message_content = True
        self.client = discord.Client(intents=self.intents)
        self.tree = app_commands.CommandTree(self.client)

        self.wiki = WikiAPI()

        self.openai_api = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        try:
            with open("settings.json", "r") as f:
                self.settings = BotSettings(**json.load(f))
        except FileNotFoundError:
            print(
                "Settings file not found. Using default settings. Please create a settings.json file to customize the bot."
            )
            self.settings = BotSettings()

        self.model = self.settings.model
        self.init_token_count = 0
        self.max_output_token_count = self.settings.max_output_token_count
        self.max_input_token_count = (
            self.settings.max_input_token_count_base - self.max_output_token_count - 1
        )
        self.max_history_token_count = (
            self.max_input_token_count - self.init_token_count
        )

        self.message_history = {}
        self.sleeping = {}
        self.search_results = {}

        self.dm_whitelist = os.getenv("DM_WHITELIST").split(",")

        self.introduction = self.settings.bot_introduction
        self.introduce_on_join = False

        self.prompt_messages = [
            {
                "role": "system",
                "content": f"""{self.settings.system_message_base}
                
In addition to chatting and providing fun interactions throught text, you also offer some unique capabilities via the following commands, which users can use to interact with you:
/search [query] [limit] - Searches Wikipedia for the given query.
/read [result_index] - Reads the Wikipedia article at the given index.
/image [prompt] [detailed] [wide] [realism] - Generates an image from a prompt using the DALL-E API.
/forget - Forgets everything in the chat history.
/sleep - Puts you to sleep. You will stop responding to messages until a user says your name or uses the /wake command.
/wake - Wakes you up.
/help - Shows the available commands.

Important:
You do not have access to these commands directly. If a user asks you to perform one of these commands, and you do not have access to an autonomous function with similar functionality, you should inform them that you can't perform the command directly and, instead, provide the command they can use to perform the action themselves.
The autonomous functions you currently have access to are:
search - Searches Wikipedia for a given query and returns the top results up to limit. You can use this autonomously when a user asks you to search for something.
read_result - Reads a Wikipedia article from search results. You can use this autonomously when a user asks you to read an article.
generate_image - Generates an image from a prompt using the DALL-E API. You can use this autonomously when a user asks you to generate an image.

{"Note: Do not use emojis under any circumstances. They do not match your personality." if self.settings.suppress_emojis else ""}
""",
            },
            {
                "role": "user",
                "content": "Can you introduce yourself?",
                "name": "admin",
            },
            {"role": "assistant", "content": self.introduction},
        ]

        self.functions = [
            {
                "name": "search",
                "description": "Searches Wikipedia for a given query and returns the top results up to limit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search for.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "The number of results to return.",
                        },
                    },
                },
            },
            {
                "name": "read_result",
                "description": "Reads a Wikipedia article from search results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result_index": {
                            "type": "integer",
                            "description": "The index of the result to read. Remember to use 0-based indexing. (Result 1 = 0, Result 2 = 1, etc.)",
                        },
                    },
                },
            },
            {
                "name": "generate_image",
                "description": "Generates an image from a prompt using the DALL-E API.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to generate the image from.",
                        },
                        "detailed": {
                            "type": "boolean",
                            "description": "Whether to generate a detailed image. Only set to True for lengthy, detailed prompts.",
                        },
                        "wide": {
                            "type": "boolean",
                            "description": "Whether to generate a wide image. Only set to True for images requiring a wide aspect ratio.",
                        },
                        "realism": {
                            "type": "boolean",
                            "description": "Whether to generate an image with more realistic colors. Best for photorealism.",
                        },
                    },
                },
            },
        ]

    def get_tokens_from_text(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        return encoding.encode(text)

    def get_text_from_tokens(self, tokens):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        return encoding.decode(tokens)

    def get_num_tokens(self, messages):
        if "gpt-3.5-turbo" in self.model or "gpt-4" in self.model:
            num_tokens = 0

            for message in messages:
                num_tokens += 4
                for key, value in message.items():
                    num_tokens += len(self.get_tokens_from_text(value))
                    if key == "name":
                        num_tokens += -1
            num_tokens += 2

            return num_tokens
        else:
            raise NotImplementedError(
                f"Token estimation is not yet implemented for this model ({self.model}). Please visit the OpenAI tiktoken documentation to learn more."
            )

    def abridge_history(self, channel_key):
        history_token_count = self.get_num_tokens(self.message_history[channel_key])
        print(f"History Tokens: {history_token_count}")

        if history_token_count > self.max_history_token_count:
            print(
                "History Token Count is Greater than Max History Tokens. Abridging history..."
            )

        removed_count = 0
        while (
            history_token_count > self.max_history_token_count
            and len(self.message_history[channel_key]) > 1
        ):
            self.message_history[channel_key].pop(0)
            history_token_count = self.get_num_tokens(self.message_history[channel_key])
            removed_count += 1

        if removed_count > 0:
            print(f"Removed {removed_count} messages from history.")
            print(f"History Tokens: {history_token_count}")

    def save_history(self, channel_key):
        save_path = f"history/{channel_key}.json"

        if not os.path.exists("history"):
            os.makedirs("history")

        with open(save_path, "w") as f:
            json.dump(self.message_history[channel_key], f)

    def load_all_history(self):
        if not os.path.exists("history"):
            os.makedirs("history")

        for file in os.listdir("history"):
            if file.endswith(".json"):
                channel_key = file.split(".")[0]
                with open(f"history/{file}", "r") as f:
                    self.message_history[channel_key] = json.load(f)

    def append_history(self, channel_key, message):
        if channel_key not in self.message_history:
            self.message_history[channel_key] = []

        self.message_history[channel_key].append(message)
        self.abridge_history(channel_key)
        self.save_history(channel_key)

    def setup(self):
        """Sets the initial parameters."""
        self.init_token_count = self.get_num_tokens(self.prompt_messages)
        self.max_history_token_count = (
            self.max_input_token_count - self.init_token_count - 1
        )

        if self.max_history_token_count < 32:
            raise ValueError(
                f"Max history token count ({self.max_history_token_count}) is too low. Must be at least 32. Please reduce the number of prompt messages or decrease the max output token count."
            )

        print(f"Prompt Tokens: {self.init_token_count}")
        print(f"Max Output Tokens: {self.max_output_token_count}")
        print(f"Max Input Tokens: {self.max_input_token_count}")
        print(f"Max History Tokens: {self.max_history_token_count}")

        print("Loading history...")
        self.load_all_history()

    def start(self):
        self.client.run(os.getenv("DISCORD_TOKEN"))

    async def generate_response(
        self,
        channel_key: str,
        ctx: discord.Interaction | discord.Message,
        functions_enabled: bool = True,
    ):
        """Generates a response to a message using the GPT-3 API."""

        messages = self.prompt_messages + self.message_history[channel_key]

        text = ""

        usage_message = ""

        attempts = 0

        error_message = ""

        async with ctx.channel.typing():
            while attempts < 3:
                try:
                    attempts += 1
                    chat_completion_args = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": self.max_output_token_count,
                        "top_p": 1,
                        "frequency_penalty": 0.4,
                        "presence_penalty": 0.6,
                        "user": self.settings.bot_name,
                    }

                    if functions_enabled:
                        chat_completion_args["functions"] = self.functions

                    response = await self.openai_api.chat.completions.create(
                        **chat_completion_args
                    )

                    if response.choices[0].message.function_call is not None:
                        available_functions = {
                            "search": self.search,
                            "read_result": self.read_result,
                            "generate_image": self.generate_image,
                        }

                        function_name = response.choices[0].message.function_call.name
                        function_args = json.loads(
                            response.choices[0].message.function_call.arguments
                        )
                        function_args["interaction"] = ctx
                        function = available_functions[function_name]

                        await function(**function_args)
                        return
                    else:
                        text = response.choices[0].message.content.strip()
                    completion_tokens = response.usage.completion_tokens
                    prompt_tokens = response.usage.prompt_tokens
                    total_tokens = response.usage.total_tokens
                    usage_message = (
                        f"Usage: {prompt_tokens} + {completion_tokens} = {total_tokens}"
                    )
                    break
                except asyncio.TimeoutError:
                    print(f"API request timed out. Aborting...")
                    error_message += (
                        f"Error (Attempt {attempts}): API request timed out.\n"
                    )
                    break
                except Exception as e:
                    print(
                        f"Unable to get response from API. Trying again in 1 second. ({attempts}/3)"
                    )
                    error_message += f"Error (Attempt {attempts}): {e}\n"
                    await asyncio.sleep(1)

            if text == "":
                text = "I'm sorry, I'm having trouble connecting to the API right now. Please try again later."

            self.append_history(channel_key, {"role": "assistant", "content": text})

        if len(text) > 2000:
            text_chunks = text.split("\n\n")
            first_half = "\n\n".join(text_chunks[: len(text_chunks) // 2])
            second_half = "\n\n".join(text_chunks[len(text_chunks) // 2 :])
            await ctx.channel.send(first_half)
            await ctx.channel.send(second_half)
        else:
            await ctx.channel.send(text)

        print(usage_message)
        print(f"Completed in {attempts} attempt(s).")
        if error_message != "":
            print(error_message)
        print("")

    async def sleep(self, interaction: discord.Interaction):
        try:
            channel_key = self.get_channel_key(
                interaction.channel, interaction.user, interaction.guild
            )
        except:
            await interaction.response.send_message("I can't respond in this channel.")
            return

        self.sleeping[channel_key] = True
        await interaction.response.send_message("Zzz...")

    async def wake(self, interaction: discord.Interaction):
        try:
            channel_key = self.get_channel_key(
                interaction.channel, interaction.user, interaction.guild
            )
        except:
            await interaction.response.send_message("I can't respond in this channel.")
            return

        self.sleeping[channel_key] = False
        await interaction.response.send_message("I'm awake!")

    async def forget(self, interaction: discord.Interaction):
        try:
            channel_key = self.get_channel_key(
                interaction.channel, interaction.user, interaction.guild
            )
        except:
            await interaction.response.send_message("I can't respond in this channel.")
            return

        self.message_history[channel_key] = []
        self.search_results[channel_key] = []
        self.save_history(channel_key)
        await interaction.response.send_message("I've forgotten everything.")

    async def read_result(self, interaction: discord.Interaction, result_index: int):
        try:
            if type(interaction) == discord.Interaction:
                channel_key = self.get_channel_key(
                    interaction.channel, interaction.user, interaction.guild
                )
            else:
                channel_key = self.get_channel_key(
                    interaction.channel, interaction.author, interaction.guild
                )
        except:
            if type(interaction) == discord.Interaction:
                await interaction.response.send_message(
                    "I can't read an article in this channel."
                )
            else:
                await interaction.channel.send(
                    "I can't read an article in this channel."
                )
            return

        if channel_key not in self.search_results:
            self.search_results[channel_key] = []

        if len(self.search_results[channel_key]) > 0:
            search_results = self.search_results[channel_key]

            if result_index < 0 or result_index >= len(search_results):
                if type(interaction) == discord.Interaction:
                    await interaction.response.send_message(
                        "I don't have a result with that index."
                    )
                else:
                    await interaction.channel.send(
                        "Looks like I tried to read an article that doesn't exist. We might need to run another search."
                    )
                return

            result_key = search_results[result_index]
            try:
                print(
                    f"Search result found. Result Key: {result_key}. Attempting to get the page..."
                )
                page = self.wiki.get_page(result_key)

                if channel_key not in self.message_history:
                    self.message_history[channel_key] = []

                for section in reversed(page.sections):
                    if section.title.lower() in [
                        "see also",
                        "references",
                        "external links",
                        "further reading",
                        "notes",
                        "footnotes",
                        "bibliography",
                        "sources",
                    ]:
                        continue

                    if (
                        len(self.get_tokens_from_text(section.text))
                        < self.max_history_token_count // 2
                        and section.text != ""
                    ):
                        self.append_history(
                            channel_key,
                            {
                                "role": "system",
                                "content": f'Here is the {section.title} section of a Wikipedia article about "{page.title}":\n\n{section.text}',
                            },
                        )
                    else:
                        continue

                    for subsection in section.sections:
                        if (
                            len(self.get_tokens_from_text(subsection.text))
                            < self.max_history_token_count // 2
                            and subsection.text != ""
                        ):
                            self.append_history(
                                channel_key,
                                {
                                    "role": "system",
                                    "content": f'Here is the {subsection.title} subsection in the {section.title} section of a Wikipedia article about "{page.title}":\n\n{subsection.text}',
                                },
                            )
                        else:
                            continue

                if type(interaction) == discord.Interaction:
                    await interaction.response.send_message(
                        f"Here's the article: {self.wiki.get_view_url(result_key)}"
                    )
                    await interaction.followup.send(
                        "Reading it now... I'll have a summary of the article in a few seconds!"
                    )
                else:
                    await interaction.channel.send(
                        f"Here's the article: {self.wiki.get_view_url(result_key)}"
                    )
                    await interaction.channel.send(
                        "Reading it now... I'll have a summary of the article in a few seconds!"
                    )

                self.append_history(
                    channel_key,
                    {
                        "role": "system",
                        "content": f'Here is the summary of a Wikipedia article about "{page.title}":\n\n{page.summary}',
                    },
                )

                self.append_history(
                    channel_key,
                    {
                        "role": "system",
                        "content": f'Please summarize the Wikipedia article you were just provided with. Begin your response with "{page.title}...". After your summary, ask the user if they have any questions about the subject of the article.',
                    },
                )

                self.abridge_history(channel_key)
                asyncio.create_task(
                    self.generate_response(channel_key, interaction, False)
                )
            except Exception as e:
                print(f"Experienced an error while getting the page: {e}")
                if type(interaction) == discord.Interaction:
                    await interaction.response.send_message(
                        "Something went wrong while getting the page. Please try again later."
                    )
                else:
                    await interaction.channel.send(
                        "Something went wrong while getting the page. Please try again later."
                    )
        else:
            if type(interaction) == discord.Interaction:
                await interaction.response.send_message(
                    "You need to search for something first."
                )
            else:
                await interaction.channel.send("What do you want me to read again?")

    async def search(
        self,
        interaction: discord.Interaction | discord.Message,
        query: str,
        limit: int = 5,
    ):
        if len(query) > 0:
            try:
                print(f'Searching for "{query}"...')
                search_results = self.wiki.search(query, limit)["pages"]
            except Exception as e:
                print(f'Experienced an error while searching for "{query}": {e}')
                search_results = []

            if type(interaction) == discord.Interaction:
                channel_key = self.get_channel_key(
                    interaction.channel, interaction.user, interaction.guild
                )
            else:
                channel_key = self.get_channel_key(
                    interaction.channel, interaction.author, interaction.guild
                )

            if len(search_results) > 0:
                try:
                    self.search_results[channel_key] = [
                        result["key"] for result in search_results
                    ]
                    search_results_message = (
                        f'Here are the top {len(search_results)} results for "{query}":'
                    )
                    search_results_view = discord.ui.View()
                    for i, result in enumerate(search_results):
                        result_label = (
                            f"\n\t{i+1}. {result['title']}: {result['description']}"
                        )
                        if len(result_label) > 80:
                            result_label = result_label[:77] + "..."
                        result_button = discord.ui.Button(
                            label=result_label, custom_id=str(i)
                        )
                        result_button.callback = lambda ctx, idx=i: self.read_result(
                            ctx, idx
                        )

                        search_results_view.add_item(result_button)

                    if type(interaction) == discord.Interaction:
                        await interaction.response.send_message(
                            search_results_message, view=search_results_view
                        )
                    else:
                        await interaction.channel.send(
                            search_results_message, view=search_results_view
                        )

                    if channel_key not in self.message_history:
                        self.message_history[channel_key] = []

                    self.append_history(
                        channel_key,
                        {
                            "role": "assistant",
                            "content": search_results_message
                            + ",".join(
                                [
                                    f"\n\t{i+1}. {result['title']}: {result['description']}"
                                    for i, result in enumerate(search_results)
                                ]
                            ),
                        },
                    )
                    print(self.message_history[channel_key][-1])
                except Exception as e:
                    print(
                        f'Experienced an error while sending search results for "{query}": {e}'
                    )
                    if type(interaction) == discord.Interaction:
                        await interaction.response.send_message(
                            f'Something went wrong while searching for "{query}". Please try again later.'
                        )
                    else:
                        await interaction.channel.send(
                            f'Something went wrong while searching for "{query}". Please try again later.'
                        )

            else:
                print("No search results found.")
                if type(interaction) == discord.Interaction:
                    await interaction.response.send_message(
                        f'Sorry, I couldn\'t find any results for "{query}".'
                    )
                else:
                    await interaction.channel.send(
                        f'Sorry, I couldn\'t find any results for "{query}".'
                    )
        else:
            if type(interaction) == discord.Interaction:
                await interaction.response.send_message(
                    "You need to include a search query."
                )
            else:
                await interaction.channel.send(
                    "What do you want me to search for again?"
                )

    async def generate_image(
        self,
        interaction: discord.Interaction | discord.Message,
        prompt: str,
        detailed: bool = False,
        wide: bool = False,
        realism: bool = False,
    ):
        """Generates an image from a prompt using the GPT-3 API."""
        try:
            if type(interaction) == discord.Interaction:
                await interaction.response.defer(thinking=True)

            print(
                f'Generating "{prompt}"... (detailed={detailed}, wide={wide}, realism={realism})'
            )
            image_completion_args = {
                "model": "dall-e-3",
                "prompt": prompt,
                "quality": "hd" if detailed else "standard",
                "size": "1792x1024" if wide else "1024x1024",
                "style": "natural" if realism else "vivid",
                "response_format": "b64_json",
                "user": self.settings.bot_name,
            }

            response = await self.openai_api.images.generate(**image_completion_args)

            image_data = response.data[0].b64_json
            binary_data = base64.b64decode(image_data)
            image = BytesIO(binary_data)

            if type(interaction) == discord.Interaction:
                await interaction.followup.send(
                    file=discord.File(image, filename="image.png")
                )
            else:
                await interaction.channel.send(
                    file=discord.File(image, filename="image.png")
                )

            user_name = str(
                interaction.user
                if type(interaction) == discord.Interaction
                else interaction.author
            )

            if type(interaction) == discord.Interaction:
                channel_key = self.get_channel_key(
                    interaction.channel, interaction.user, interaction.guild
                )
            else:
                channel_key = self.get_channel_key(
                    interaction.channel, interaction.author, interaction.guild
                )

            if channel_key not in self.message_history:
                self.message_history[channel_key] = []

            if type(interaction) == discord.Interaction:
                self.append_history(
                    channel_key,
                    {
                        "role": "system",
                        "content": f'{user_name} has generated an image from the prompt "{prompt}".',
                    },
                )
            else:
                self.append_history(
                    channel_key,
                    {
                        "role": "system",
                        "content": f'An image has been generated from the prompt "{prompt}". Follow up with a comment about the image.',
                    },
                )

            self.abridge_history(channel_key)
            asyncio.create_task(self.generate_response(channel_key, interaction, False))
        except Exception as e:
            print(f"Experienced an error while generating image: {e}")
            await interaction.response.send_message(
                "Something went wrong while generating the image. Please try again later."
            )

    def get_channel_key(
        self,
        channel: discord.TextChannel | discord.DMChannel,
        author: discord.User | discord.Member,
        guild: discord.Guild,
    ):
        """Returns a unique key for each channel."""
        user_name = str(author)
        if channel.type == discord.ChannelType.private:
            print(author)
            if str(author) in self.dm_whitelist:
                return f"user-{user_name}"
            else:
                return None
        elif channel.name == "random":
            return f"guild-{guild.id}#{channel.name}"
        else:
            return None

    async def handle_on_ready(self):
        print(f"{self.client.user} has connected to Discord!")

        """Finds the general channel in each server and adds it to a list of channel IDs."""
        channel_ids = []
        for guild in self.client.guilds:
            for channel in guild.channels:
                if channel.name in ["general", "random"]:
                    channel_ids.append(channel.id)

        if self.introduce_on_join:
            """Sends a message to each selected channel."""
            for channel_id in channel_ids:
                channel = self.client.get_channel(channel_id)
                await channel.send(self.introduction)

    async def handle_on_message(self, message):
        """Responds to a message with a random response from the GPT-3 API."""
        if message.author == self.client.user:
            return

        channel_key = self.get_channel_key(
            message.channel, message.author, message.guild
        )
        user_name = str(message.author)

        if channel_key is None:
            if self.settings.bot_name.lower().split(" ")[0] in message.content.lower():
                await message.channel.send(
                    "Hey! I can't respond in this channel. Talk to me in the #random channel instead."
                )
            return

        if message.content == "":
            return

        query = message.content.strip()

        if query == "":
            return

        print(f"Query Key: {channel_key}")

        self.append_history(
            channel_key, {"role": "user", "content": f"{query}", "name": f"{user_name}"}
        )
        time_message = {
            "role": "system",
            "content": f"The current date/time is {(message.created_at + timedelta(hours=-8)).strftime('%I:%M %p on %B %d, %Y')}. The local timezone is US/Pacific.",
        }
        print(f"Time Message: {time_message}")
        self.append_history(channel_key, time_message)
        self.abridge_history(channel_key)

        if channel_key not in self.sleeping:
            self.sleeping[channel_key] = False

        if self.settings.bot_name.lower().split(" ")[0] in message.content.lower():
            if self.sleeping[channel_key]:
                self.sleeping[channel_key] = False
                print("Waking up...")
        elif self.sleeping[channel_key]:
            print("Still sleeping...")
            return

        print("Responding to message...")

        asyncio.create_task(self.generate_response(channel_key, message))


if __name__ == "__main__":
    bot = BotGPT()
    bot.setup()

    @bot.client.event
    async def on_ready():
        await bot.tree.sync()
        await bot.handle_on_ready()

    @bot.client.event
    async def on_message(message):
        if message.content.startswith("/"):
            return
        await bot.handle_on_message(message)

    @bot.tree.command(name="help", description="Show available commands.")
    async def recieve_help_command(interaction: discord.Interaction):
        help_message = f"""Here are the available commands:
`/search [query] [limit]`- Searches Wikipedia for the given query.
`/read [result index]` - Reads the Wikipedia article at the given index.
`/image [prompt] [detailed] [wide] [realism]` - Generates an image from a prompt using the DALL-E API.
`/forget` - Forgets everything.
`/sleep` - Puts {bot.settings.bot_name} to sleep.
`/wake` - Wakes {bot.settings.bot_name} up.
`/help` - Shows this message."""
        await interaction.response.send_message(help_message)

    @bot.tree.command(
        name="sleep", description="Puts {bot.settings.bot_name} to sleep."
    )
    async def recieve_sleep_command(interaction: discord.Interaction):
        await bot.sleep(interaction)

    @bot.tree.command(name="wake", description="Wakes {bot.settings.bot_name} up.")
    async def recieve_wake_command(interaction: discord.Interaction):
        await bot.wake(interaction)

    @bot.tree.command(name="forget", description="Forgets everything.")
    async def recieve_forget_command(interaction: discord.Interaction):
        await bot.forget(interaction)

    @bot.tree.command(
        name="read",
        description="Reads a Wikipedia article from search results. Must run the search command first.",
    )
    async def recieve_read_command(interaction: discord.Interaction, result_index: int):
        await bot.read_result(interaction, result_index - 1)

    @bot.tree.command(
        name="search",
        description="Searches Wikipedia for a given query and returns the top results up to limit.",
    )
    async def recieve_search_command(
        interaction: discord.Interaction,
        query: str,
        limit: app_commands.Range[int, 1, 10] = 5,
    ):
        await bot.search(interaction, query, limit)

    @bot.tree.command(
        name="image",
        description="Generates an image from a prompt using the DALL-E API.",
    )
    async def recieve_image_command(
        interaction: discord.Interaction,
        prompt: str,
        detailed: bool = False,
        wide: bool = False,
        realism: bool = False,
    ):
        await bot.generate_image(interaction, prompt, detailed, wide, realism)

    bot.start()
