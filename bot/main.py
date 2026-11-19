"""Adaption community support bot for Discord.

- /ticket-panel posts an "Open a ticket" button in the current channel.
- Each ticket is a private thread with the user; the bot answers there in the
  user's language, grounded in Adaption's docs (see answer.py).
- Under every answer: helpful / not helpful, and Talk to a human (pings the
  support role and pauses the bot in that ticket; staff resume it with /resume-bot).
- Messages containing an Adaption API key are deleted with a warning to rotate it.

    python bot/main.py
"""
import asyncio
import json
import re
import time

import discord
from discord import app_commands

from answer import Answerer
from settings import DATA, load_settings

API_KEY = re.compile(r"\bpt_(live|test)_[A-Za-z0-9_-]{8,}")
HISTORY_LIMIT = 12
WELCOME = (
    "Hi {user}! Ask me anything about Adaption: Adaptive Data, AutoScientist, Invent, "
    "Inference Stacks, the API and SDK, pricing and credits, or the community programs "
    "(ambassadors, hackathons, grants, startups).\n"
    "Write in any language: English, हिन्दी, اردو, Tagalog, Kiswahili, or a mix like Hinglish or Taglish.\n"
    "Never paste API keys or private data here. Use **Talk to a human** any time to reach the team."
)
PANEL = (
    "Questions about Adaption's products, your account, credits, or our community programs?\n"
    "Open a private ticket and our assistant answers right away, in your language. "
    "The team can join the ticket whenever you need a human."
)
BOT_PERMISSIONS = discord.Permissions(
    view_channel=True, send_messages=True, send_messages_in_threads=True, create_private_threads=True,
    manage_threads=True, manage_messages=True, embed_links=True, read_message_history=True,
)


class Tickets:
    """Ticket state in data/bot/tickets.json: thread id -> owner, handoff flag, closed flag."""

    def __init__(self):
        self.path = DATA / "tickets.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def save(self):
        DATA.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")

    def get(self, thread_id):
        return self.data.get(str(thread_id))

    def open_for(self, user_id):
        return next((int(t) for t, v in self.data.items() if v["user_id"] == user_id and not v["closed"]), None)

    def add(self, thread_id, user_id):
        self.data[str(thread_id)] = {"user_id": user_id, "handoff": False, "closed": False, "created": time.time()}
        self.save()

    def set(self, thread_id, **kw):
        self.data[str(thread_id)].update(kw)
        self.save()


def log_feedback(record: dict):
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_staff(member) -> bool:
    perms = getattr(member, "guild_permissions", None)
    return bool(perms and (perms.manage_threads or perms.manage_guild))


class SupportBot(discord.Client):
    def __init__(self, settings):
        intents = discord.Intents.default()
        intents.message_content = True  # privileged: enable it in the Developer Portal
        super().__init__(intents=intents)
        self.settings = settings
        self.tree = app_commands.CommandTree(self)
        self.tickets = Tickets()
        self.answerer = Answerer(settings.model, settings.retrieval, settings.kb_dataset_id)
        self.locks: dict[int, asyncio.Lock] = {}

    async def setup_hook(self):
        # Persistent views: buttons keep working after a restart.
        self.add_view(PanelView(self))
        self.add_view(TicketView(self))
        self.add_view(AnswerView(self))
        register_commands(self)
        if self.settings.guild_id:  # instant command sync in one server
            guild = discord.Object(id=self.settings.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()

    async def on_ready(self):
        print(f"Logged in as {self.user}. Model {self.settings.model}, retrieval {self.answerer.retrieval}.")
        print("Invite link: " + discord.utils.oauth_url(self.user.id, permissions=BOT_PERMISSIONS,
                                                        scopes=("bot", "applications.commands")))

    # --- tickets --------------------------------------------------------------
    async def open_ticket(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        user = interaction.user
        existing = self.tickets.open_for(user.id)
        if existing and (thread := self.get_channel(existing)):
            return await interaction.followup.send(f"You already have an open ticket: {thread.mention}", ephemeral=True)
        thread = await interaction.channel.create_thread(
            name=f"ticket-{user.display_name}"[:90], type=discord.ChannelType.private_thread,
            invitable=False, auto_archive_duration=1440)
        await thread.add_user(user)
        self.tickets.add(thread.id, user.id)
        await thread.send(WELCOME.format(user=user.mention), view=TicketView(self),
                          allowed_mentions=discord.AllowedMentions(users=True))
        await interaction.followup.send(f"Your ticket is open: {thread.mention}", ephemeral=True)

    async def handoff(self, interaction: discord.Interaction):
        t = self.tickets.get(interaction.channel_id)
        if not t:
            return await interaction.response.send_message("This only works inside a ticket.", ephemeral=True)
        if t["handoff"]:
            return await interaction.response.send_message("The team has already been notified.", ephemeral=True)
        self.tickets.set(interaction.channel_id, handoff=True)
        role = f"<@&{self.settings.support_role_id}>" if self.settings.support_role_id else "Support team"
        await interaction.response.send_message(
            f"{role}: {interaction.user.mention} asked for a human. I'll stay quiet in this ticket until "
            "the team runs `/resume-bot`.",
            allowed_mentions=discord.AllowedMentions(roles=True, users=False))
        try:
            await interaction.channel.edit(name=("🙋 " + interaction.channel.name)[:100])
        except discord.HTTPException:
            pass  # renaming is cosmetic; thread edits are rate limited

    async def close_ticket(self, interaction: discord.Interaction):
        t = self.tickets.get(interaction.channel_id)
        if not t:
            return await interaction.response.send_message("This only works inside a ticket.", ephemeral=True)
        if interaction.user.id != t["user_id"] and not is_staff(interaction.user):
            return await interaction.response.send_message("Only the ticket owner or staff can close it.", ephemeral=True)
        self.tickets.set(interaction.channel_id, closed=True)
        await interaction.response.send_message(f"Ticket closed by {interaction.user.mention}. Thanks for reaching out!")
        await interaction.channel.edit(archived=True, locked=True)

    async def feedback(self, interaction: discord.Interaction, rating: str):
        t = self.tickets.get(interaction.channel_id)
        if not t or interaction.user.id != t["user_id"]:
            return await interaction.response.send_message("Only the ticket owner can rate answers.", ephemeral=True)
        log_feedback({"time": time.time(), "thread_id": interaction.channel_id, "message_id": interaction.message.id,
                      "rating": rating, "answer": interaction.message.content, "model": self.settings.model})
        msg = "Thanks, glad it helped!" if rating == "up" else \
            "Thanks for telling me. Try rephrasing, or use **Talk to a human** to reach the team."
        await interaction.response.send_message(msg, ephemeral=True)

    # --- messages ---------------------------------------------------------------
    async def on_message(self, message: discord.Message):
        if message.author.bot or not isinstance(message.channel, discord.Thread):
            return
        t = self.tickets.get(message.channel.id)
        if not t or t["closed"]:
            return
        if API_KEY.search(message.content):
            try:
                await message.delete()
                note = "I deleted your message because it contained an Adaption API key."
            except discord.HTTPException:
                note = "Your message contains an Adaption API key. Please delete it."
            await message.channel.send(f"{message.author.mention} {note} Treat the key as leaked: rotate it now "
                                       "under Settings > API keys in the Adaption app.")
            return
        if t["handoff"] or message.author.id != t["user_id"]:
            return  # a human is handling it, or staff are talking
        lock = self.locks.setdefault(message.channel.id, asyncio.Lock())
        async with lock, message.channel.typing():
            history = await self.history(message.channel, t["user_id"])
            text, _ = await self.answerer.reply(history)
        parts = split(text)
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            await message.channel.send(part, view=AnswerView(self) if last else None, suppress_embeds=True,
                                       allowed_mentions=discord.AllowedMentions.none())

    async def history(self, thread: discord.Thread, owner_id: int) -> list[dict]:
        turns = []
        async for m in thread.history(limit=HISTORY_LIMIT):
            if m.author.id == owner_id and m.content:
                turns.append({"role": "user", "content": m.content})
            elif m.author.id == self.user.id and m.content and "support:up" in custom_ids(m):
                turns.append({"role": "assistant", "content": m.content})  # only answers carry feedback buttons
        turns.reverse()
        while turns and turns[0]["role"] != "user":
            turns.pop(0)
        return turns


def custom_ids(message: discord.Message) -> set[str]:
    return {getattr(c, "custom_id", None) for row in message.components for c in getattr(row, "children", [])}


def split(text: str, limit: int = 1900) -> list[str]:
    """Discord caps messages at 2,000 characters; split on paragraph breaks."""
    parts, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) + 2 > limit and cur:
            parts.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
        while len(cur) > limit:
            parts.append(cur[:limit])
            cur = cur[limit:]
    return parts + ([cur] if cur else [])


# --- views (fixed custom_ids so they survive restarts) -------------------------
class PanelView(discord.ui.View):
    def __init__(self, bot):
        super().__init__(timeout=None)
        self.bot = bot

    @discord.ui.button(label="Open a ticket", emoji="🎫", style=discord.ButtonStyle.primary, custom_id="support:open")
    async def open(self, interaction, _):
        await self.bot.open_ticket(interaction)


class TicketView(discord.ui.View):
    def __init__(self, bot):
        super().__init__(timeout=None)
        self.bot = bot

    @discord.ui.button(label="Talk to a human", emoji="🙋", style=discord.ButtonStyle.secondary, custom_id="support:human")
    async def human(self, interaction, _):
        await self.bot.handoff(interaction)

    @discord.ui.button(label="Close ticket", emoji="🔒", style=discord.ButtonStyle.danger, custom_id="support:close")
    async def close(self, interaction, _):
        await self.bot.close_ticket(interaction)


class AnswerView(discord.ui.View):
    def __init__(self, bot):
        super().__init__(timeout=None)
        self.bot = bot

    @discord.ui.button(emoji="👍", style=discord.ButtonStyle.secondary, custom_id="support:up")
    async def up(self, interaction, _):
        await self.bot.feedback(interaction, "up")

    @discord.ui.button(emoji="👎", style=discord.ButtonStyle.secondary, custom_id="support:down")
    async def down(self, interaction, _):
        await self.bot.feedback(interaction, "down")

    @discord.ui.button(label="Talk to a human", emoji="🙋", style=discord.ButtonStyle.secondary, custom_id="support:human2")
    async def human(self, interaction, _):
        await self.bot.handoff(interaction)


def register_commands(bot: SupportBot):
    @bot.tree.command(name="ticket-panel", description="Post the support ticket panel in this channel")
    @app_commands.default_permissions(manage_guild=True)
    async def ticket_panel(interaction: discord.Interaction):
        if not isinstance(interaction.channel, discord.TextChannel):
            return await interaction.response.send_message("Run this in a text channel.", ephemeral=True)
        embed = discord.Embed(title="Adaption support", description=PANEL, color=0x5865F2)
        await interaction.channel.send(embed=embed, view=PanelView(bot))
        await interaction.response.send_message("Panel posted.", ephemeral=True)

    @bot.tree.command(name="resume-bot", description="Let the assistant answer in this ticket again")
    @app_commands.default_permissions(manage_threads=True)
    async def resume_bot(interaction: discord.Interaction):
        if not bot.tickets.get(interaction.channel_id):
            return await interaction.response.send_message("Run this inside a ticket.", ephemeral=True)
        bot.tickets.set(interaction.channel_id, handoff=False)
        await interaction.response.send_message("The assistant is back on in this ticket.")


def main():
    settings = load_settings()
    SupportBot(settings).run(settings.discord_token)


if __name__ == "__main__":
    main()
