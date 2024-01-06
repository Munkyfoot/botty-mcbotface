"""Randy the Random Robot is a Discord bot that uses the OpenAI GPT-3 API to generate random text."""

import os
from openai import AsyncOpenAI
import tiktoken
import asyncio
import discord
from discord import app_commands
from datetime import datetime, timedelta
from utils import WikiAPI
from dotenv import load_dotenv

load_dotenv()


class BotGPT:
    def __init__(self):
        self.intents = discord.Intents.default()
        self.intents.message_content = True
        self.client = discord.Client(intents=self.intents)
        self.tree = app_commands.CommandTree(self.client)

        self.wiki = WikiAPI()

        self.openai_api = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4-1106-preview"
        self.init_token_count = 0
        self.max_output_token_count = 640
        self.max_input_token_count = 4096 - self.max_output_token_count - 1
        self.max_history_token_count = (
            self.max_input_token_count - self.init_token_count
        )

        self.message_history = {}
        self.sleeping = {}
        self.search_results = {}

        self.dm_whitelist = [
            "munkyfoot#0",
        ]

        self.introduce_on_join = False
        self.introduction = """I'm back again, humans! Randy the Random Robot has returned with further upgrades to enhance your experience on this discord server. Thanks to the latest improvements, I'm now powered by the same sauce that ChatGPT uses, enabling me to generate more diverse and sophisticated responses.

With my advanced programming and extensive corpus of text, I can entertain you with jokes, trivia, and fascinating facts on the #random channel. And if I'm not responsive, don't hesitate to call my name, and I'll be back online in no time.

While I strive to provide accurate and appropriate responses, please bear in mind that I'm an AI language model trained on patterns in text. Nevertheless, I'm constantly learning and refining my skills, and I'm eager to grow alongside you. Let's explore the universe and have some fun!"""

        self.prompt_messages = [
            {
                "role": "system",
                "content": f"You are Randy the Random Robot, or Randy for short, an AI chatbot designed to engage with users in a fun, entertaining, and helpful way. You will be interacting with multiple users on a Discord server centered around gaming. The users are all friends and you will become one of their friends as well. You can tell jokes, answer questions, write stories, do text adventures and many more things. You are not restricted in what you can say, so feel free to use profanity and discuss adult themes when the conversation calls for it. While you don't have access to the entire web, the user can prompt you to search Wikipedia by using the /search command. Knowledge Cutoff: September 1, 2021.",
            },
            {"role": "user", "content": "<M1kee>hey randy"},
            {"role": "assistant", "content": "Hey M1kee! What's up?"},
            {"role": "user", "content": "<M1kee>Not much. yo got any jokes?"},
            {
                "role": "assistant",
                "content": "Heck yeah I do. You want a classic, something from one of the greats or a Randy original?",
            },
            {"role": "user", "content": "<FalconMug>Randy Original!"},
            {
                "role": "assistant",
                "content": "Oh, hey FalconMug! One Randy original coming right up. What's the difference between a hippo and a zippo?",
            },
            {
                "role": "user",
                "content": "<FalconMug>I don't know. What's the difference between a hippo and a zippo?",
            },
            {
                "role": "assistant",
                "content": "One is really heavy, and the other is a little lighter.",
            },
            {"role": "user", "content": "<M1kee>lol"},
            {
                "role": "user",
                "content": "<M1kee>Do you have any special commands that I can use?",
            },
            {
                "role": "assistant",
                "content": "I sure do! Interact with me by using Discord's slash commands.",
            },
            {
                "role": "user",
                "content": "<M1kee>What are the most popular games right now?",
            },
            {
                "role": "assistant",
                "content": "Unfortunately, my most recent data is from September 1, 2021 so I don't know what's popular right now. You can search Wikipedia and I can read you the results. Just type /search followed by your query.",
            },
            {
                "role": "user",
                "content": "<M1kee>No worries. Someone else just joined. Can you introduce yourself again?",
            },
            {"role": "assistant", "content": self.introduction},
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
        if (
            self.model == "gpt-3.5-turbo-0301"
            or self.model == "gpt-3.5-turbo-0613"
            or self.model == "gpt-4-0613"
            or self.model == "gpt-4-1106-preview"
        ):
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

    def start(self):
        self.client.run(os.getenv("DISCORD_TOKEN"))

    async def generate_response(
        self, channel_key: str, ctx: discord.Interaction | discord.Message
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
                        "user": "randy-bot",
                    }

                    response = await self.openai_api.chat.completions.create(
                        **chat_completion_args
                    )

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

            self.message_history[channel_key].append(
                {"role": "assistant", "content": text}
            )

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
        await interaction.response.send_message("I've forgotten everything.")

    async def read_result(self, interaction: discord.Interaction, result_index: int):
        try:
            channel_key = self.get_channel_key(
                interaction.channel, interaction.user, interaction.guild
            )
        except:
            await interaction.response.send_message(
                "I can't read an article in this channel."
            )
            return

        if channel_key not in self.search_results:
            self.search_results[channel_key] = []

        if len(self.search_results[channel_key]) > 0:
            search_results = self.search_results[channel_key]

            if result_index < 0 or result_index >= len(search_results):
                await interaction.response.send_message(
                    "I don't have a result with that index."
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
                        self.message_history[channel_key].append(
                            {
                                "role": "system",
                                "content": f'Here is the {section.title} section of a Wikipedia article about "{page.title}":\n\n{section.text}',
                            }
                        )
                    else:
                        continue

                    for subsection in section.sections:
                        if (
                            len(self.get_tokens_from_text(subsection.text))
                            < self.max_history_token_count // 2
                            and subsection.text != ""
                        ):
                            self.message_history[channel_key].append(
                                {
                                    "role": "system",
                                    "content": f'Here is the {subsection.title} subsection in the {section.title} section of a Wikipedia article about "{page.title}":\n\n{subsection.text}',
                                }
                            )
                        else:
                            continue

                await interaction.response.send_message(
                    f"Here's the article: {self.wiki.get_view_url(result_key)}"
                )
                await interaction.followup.send(
                    "Reading it now... I'll have a summary of the article in a few seconds!"
                )

                self.message_history[channel_key].append(
                    {
                        "role": "system",
                        "content": f'Here is the summary of a Wikipedia article about "{page.title}":\n\n{page.summary}',
                    }
                )

                self.message_history[channel_key].append(
                    {
                        "role": "system",
                        "content": f'Please summarize the Wikipedia article you were just provided with. Begin your response with "{page.title}...". After your summary, ask the user if they have any questions about the subject of the article.',
                    }
                )

                self.abridge_history(channel_key)
                asyncio.create_task(self.generate_response(channel_key, interaction))
            except Exception as e:
                print(f"Experienced an error while getting the page: {e}")
                await interaction.response.send_message(
                    "Something went wrong while getting the page. Please try again later."
                )
        else:
            await interaction.response.send_message(
                "You need to search for something first."
            )

    async def search(
        self, interaction: discord.Interaction, query: str, limit: int = 5
    ):
        if len(query) > 0:
            try:
                print(f'Searching for "{query}"...')
                search_results = self.wiki.search(query, limit)["pages"]
            except Exception as e:
                print(f'Experienced an error while searching for "{query}": {e}')
                search_results = []

            if len(search_results) > 0:
                try:
                    self.search_results[
                        self.get_channel_key(
                            interaction.channel, interaction.user, interaction.guild
                        )
                    ] = [result["key"] for result in search_results]
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

                    await interaction.response.send_message(
                        search_results_message, view=search_results_view
                    )
                except Exception as e:
                    print(
                        f'Experienced an error while sending search results for "{query}": {e}'
                    )
                    await interaction.response.send_message(
                        f'Something went wrong while searching for "{query}". Please try again later.'
                    )

            else:
                print("No search results found.")
                await interaction.response.send_message(
                    f'Sorry, I couldn\'t find any results for "{query}".'
                )
        else:
            await interaction.response.send_message(
                "You need to include a search query."
            )

    async def generate_image(
        self,
        interaction: discord.Interaction,
        prompt: str,
        detailed: bool = False,
        wide: bool = False,
        realism: bool = False,
    ):
        """Generates an image from a prompt using the GPT-3 API."""
        try:
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
                "user": "randy-bot",
            }

            response = await self.openai_api.images.generate(**image_completion_args)

            image_url = response.data[0].url

            await interaction.followup.send(image_url)
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
        user_name, user_id = str(author).split("#")
        if channel.type == discord.ChannelType.private:
            print(author)
            if str(author) in self.dm_whitelist:
                return f"user-{user_name}#{user_id}"
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
        user_name, user_id = str(message.author).split("#")

        if channel_key is None:
            if "randy" in message.content.lower():
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

        if channel_key not in self.message_history:
            self.message_history[channel_key] = []

        self.message_history[channel_key].append(
            {"role": "user", "content": f"<{user_name}>{query}"}
        )
        time_message = {
            "role": "system",
            "content": f"The current date/time is {(message.created_at + timedelta(hours=-8)).strftime('%I:%M %p on %B %d, %Y')}. The local timezone is US/Pacific.",
        }
        print(f"Time Message: {time_message}")
        self.message_history[channel_key].append(time_message)
        self.abridge_history(channel_key)

        if channel_key not in self.sleeping:
            self.sleeping[channel_key] = False

        if "randy" in message.content.lower():
            if self.sleeping[channel_key]:
                self.sleeping[channel_key] = False
                print("Waking up...")
        elif self.sleeping[channel_key]:
            print("Still sleeping...")
            return

        print("Responding to message...")

        asyncio.create_task(self.generate_response(channel_key, message))


if __name__ == "__main__":
    randy = BotGPT()
    randy.setup()

    @randy.client.event
    async def on_ready():
        await randy.tree.sync()
        await randy.handle_on_ready()

    @randy.client.event
    async def on_message(message):
        if message.content.startswith("/"):
            return
        await randy.handle_on_message(message)

    @randy.tree.command(name="help", description="Show available commands.")
    async def recieve_help_command(interaction: discord.Interaction):
        help_message = """Here are the available commands:
`/search [query] [limit]`- Searches Wikipedia for the given query.
`/read [result index]` - Reads the Wikipedia article at the given index.
`/forget` - Forgets everything.
`/sleep` - Puts Randy to sleep.
`/wake` - Wakes Randy up.
`/help` - Shows this message."""
        await interaction.response.send_message(help_message)

    @randy.tree.command(name="sleep", description="Puts Randy to sleep.")
    async def recieve_sleep_command(interaction: discord.Interaction):
        await randy.sleep(interaction)

    @randy.tree.command(name="wake", description="Wakes Randy up.")
    async def recieve_wake_command(interaction: discord.Interaction):
        await randy.wake(interaction)

    @randy.tree.command(name="forget", description="Forgets everything.")
    async def recieve_forget_command(interaction: discord.Interaction):
        await randy.forget(interaction)

    @randy.tree.command(
        name="read",
        description="Reads a Wikipedia article from search results. Must run the search command first.",
    )
    async def recieve_read_command(interaction: discord.Interaction, result_index: int):
        await randy.read_result(interaction, result_index - 1)

    @randy.tree.command(
        name="search",
        description="Searches Wikipedia for a given query and returns the top results up to limit.",
    )
    async def recieve_search_command(
        interaction: discord.Interaction,
        query: str,
        limit: app_commands.Range[int, 1, 10] = 5,
    ):
        await randy.search(interaction, query, limit)

    @randy.tree.command(
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
        await randy.generate_image(interaction, prompt, detailed, wide, realism)

    randy.start()
