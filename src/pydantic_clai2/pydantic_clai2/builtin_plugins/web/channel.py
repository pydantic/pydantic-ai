"""One live session as browsers see it: its transcript as it streams, its status, its delegated tasks, its input."""

import math
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from pydantic_ai import AgentStreamEvent
from pydantic_ai.messages import BinaryContent, EnqueuedMessagesEvent, ModelMessage, UserContent, UserPromptPart
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.ask_user import AskUserAnswer
from pydantic_clai2.builtin_plugins.web.protocol import (
    CompletionInfo,
    EventTranslator,
    ImageIds,
    ItemAdded,
    ItemLog,
    ItemMessage,
    Notice,
    QueueChanged,
    QueuedItem,
    SessionInfo,
    SessionMessage,
    Snapshot,
    StatusChanged,
    StatusInfo,
    TaskEntry,
    TextDelta,
    TitleChanged,
    TurnEnded,
    TurnStarted,
    delivered_steering,
    dump,
    history_images,
    history_items,
    queued_item,
    usage_info,
    usage_report,
    user_images,
    user_item,
)
from pydantic_clai2.builtin_plugins.web.workers import Workers, task_entry
from pydantic_clai2.commands import is_command_input
from pydantic_clai2.plugins import CommandEnd, LiveSession, TurnOutcome
from pydantic_clai2.ui.prompt.prompt_queue import NOT_QUEUED
from pydantic_clai2.ui.rendering.usage_report import session_usage

if TYPE_CHECKING:
    from pydantic_clai2.builtin_plugins.web.service import WebService

CODE_MODE = 'run.code_mode'
SPECULATION = 'run.speculative_code_mode'
Switch = Literal['run.code_mode', 'run.speculative_code_mode']
"""The settings the browser's toggles change; nothing else is set from it but through the prompt."""
USAGE_COMMANDS = frozenset({'/usage', '/cost'})
"""Commands whose notice carries the figures they report, for the page's usage card."""
STATUS_INTERVAL = 0.2
"""Seconds between status updates sent while a turn streams, so token counts tick; turn and command ends always send one."""


class SessionChannel:
    """One live session, as every browser watching it shares it.

    Everything a browser does goes through the session's prompt, as typing there would, so the
    terminal and the browser always drive the same session. What happens comes back through the
    plugin's moments and the session's runs, which `WebService` routes here.
    """

    def __init__(self, *, session: LiveSession, service: 'WebService') -> None:
        self.session = session
        self.service = service
        self.shown = session.conversation.conversation_id
        """The conversation browsers were last sent, so a retitle sends only the title and a switch a snapshot."""
        self.running = False
        """Whether a run of the held conversation streams now, as browsers see it."""
        self.workers = Workers(publish=self.publish, history=self._task_history)
        """The session's delegated tasks, which the plugin's `on_task_event` keeps up to date."""
        self._translator = EventTranslator()
        """One for the session's life, so item IDs stay unique across turns in a browser that keeps them all."""
        self._turn = ItemLog()
        self._status_sent = 0.0
        self._usage: RunUsage | None = None
        """The running turn's usage, as its run context reports it; `None` between turns."""
        self._queue_sent: list[QueuedItem] = []
        """The queued input browsers were last sent, so only a change sends it again."""
        self._steered = 0
        self._prompt_images: list[BinaryContent] = []
        """The last prompt's images, which its run may not have added to the conversation yet."""
        self._image_ids = ImageIds()

    @property
    def id(self) -> str:
        """The conversation the session holds, which browsers name it by."""
        return self.session.conversation.conversation_id

    def publish(self, message: SessionMessage) -> None:
        self.service.hub.publish_session(message, session=self.id)

    def _task_history(self, task_id: str) -> list[ModelMessage]:
        tasks = self.session.tasks
        return [] if tasks is None else tasks.history(task_id)

    # What the browser sees.

    def snapshot(self) -> Snapshot:
        conversation = self.session.conversation
        items = history_items(conversation.messages, image_ids=self._image_ids)
        self._image_ids.keep([*self._prompt_images, *history_images(conversation.messages)])
        if self.running:
            items += self._turn.items.values()
        service = self.service
        sessions = self.session.sessions
        return Snapshot(
            session=SessionInfo(
                id=conversation.conversation_id, title=conversation.title, workspace=str(self.session.workspace)
            ),
            items=items,
            running=self.running,
            status=self.status(),
            questions=service.answerer.asked(conversation.conversation_id),
            dialogs=service.dialogs.dialogs,
            theme=service.theme(),
            tasks=self.tasks(),
            task_items=self.workers.items(),
            queue=self.queue(),
            follows_workspace=sessions is not None and sessions.follows_workspace,
            version=service.version,
            update=service.update,
        )

    def image(self, image_id: str) -> BinaryContent | None:
        """The image the transcript lists as `image_id`: the last prompt's, or one of the conversation's prompts."""
        return self._image_ids.find(image_id)

    def status(self) -> StatusInfo:
        status = self.session.status
        return StatusInfo(
            model=status.model,
            activity=status.activity if self.running else 'ready',
            context_tokens=status.context_tokens,
            context_window=status.context_window,
            cost=None if status.cost is None else str(status.cost),
            cost_is_lower_bound=status.cost_is_lower_bound,
            streamed_tokens=math.ceil(status.streamed_chars / 4),
            output_tokens=status.output_tokens,
            run=None if self._usage is None else usage_info(self._usage),
            code_mode=self._switch(CODE_MODE),
            speculation=self._switch(SPECULATION),
        )

    def _switch(self, key: str) -> bool | None:
        try:
            return self.session.setting(key) is True
        except KeyError:
            return None

    def tasks(self) -> list[TaskEntry]:
        """The conversation's delegated tasks, each parent before its children; none without a task owner."""
        tasks = self.session.tasks
        return [] if tasks is None else [task_entry(task) for task in tasks.listing()]

    def queue(self) -> list[QueuedItem]:
        """The input waiting for the session's running turn or command, as its prompt holds it."""
        return [queued_item(entry) for entry in self.session.queue]

    def publish_queue(self) -> None:
        """Send watchers the queued input when it changed: entered, edited, or removed here or in the terminal, or run."""
        queue = self.queue()
        if queue != self._queue_sent:
            self._queue_sent = queue
            self.publish(QueueChanged(queue=queue))

    def publish_status(self) -> None:
        self._status_sent = time.monotonic()
        self.publish(StatusChanged(status=self.status()))

    # What happens in the session, from the plugin's moments and capability.

    def observe(self, event: AgentStreamEvent, *, usage: RunUsage, by_user: bool = False) -> None:
        """Send `event` to watchers; `by_user` marks a `/monty` cell's run, which is not a turn and has no usage to show."""
        if not by_user:
            self._usage = usage
        if isinstance(event, EnqueuedMessagesEvent):
            # Steering, from here or the terminal, shows once the run's model receives it.
            for part in delivered_steering(event):
                self._steered += 1
                # The run's history may not hold these images yet; a snapshot keeps them, as the prompt's.
                self._prompt_images += user_images(part)
                self._show(ItemAdded(item=user_item(part, id=f'steer{self._steered}', image_ids=self._image_ids)))
        for message in self._translator.translate(event):
            if by_user:
                # Only the cell itself: the scripted model's closing text is not the user's.
                if isinstance(message, TextDelta) or (isinstance(message, ItemAdded) and message.item.kind != 'tool'):
                    continue
                if isinstance(message, ItemAdded):
                    message = ItemAdded(item=replace(message.item, by_user=True))
            self._show(message)
        if time.monotonic() - self._status_sent >= STATUS_INTERVAL:
            self.publish_status()

    def _show(self, message: ItemMessage) -> None:
        self._turn.apply(message)
        self.publish(message)

    def run_started(self, prompt: str | Sequence[UserContent] | None) -> None:
        """A run of the held conversation started; a run woken by a finished task has no prompt to show."""
        self.running = True
        self._usage = None
        self._turn.clear()
        item = None
        if prompt is not None:
            part = UserPromptPart(prompt)
            self._prompt_images = user_images(part)
            item = user_item(part, id=f'prompt{time.monotonic_ns()}', image_ids=self._image_ids)
        self.publish(TurnStarted(item=item))
        self.publish_status()

    def run_ended(
        self, outcome: TurnOutcome, *, error: BaseException | None = None, usage: RunUsage | None = None
    ) -> None:
        self.running = False
        self._usage = None
        self._turn.clear()
        self.publish(
            TurnEnded(
                outcome=outcome,
                error=None if error is None else str(error) or type(error).__name__,
                usage=None if usage is None else usage_info(usage),
            )
        )
        self.publish_status()

    def conversation_changed(self, *, title: str | None) -> bool:
        """The session's conversation or its title changed; `True` when it holds another conversation now."""
        if self.id == self.shown:
            self.publish(TitleChanged(title=title))
            return False
        hub = self.service.hub
        hub.rename(self.shown, self.id)
        self.shown = self.id
        self.workers.reset()
        self._prompt_images = []
        # To the browsers following it; one that watched the new conversation before gets it from `follow`.
        hub.send([client for client in hub.connected() if self.session.id in client.unwatch], dump(self.snapshot()))
        return True

    def command_ended(self, event: CommandEnd) -> None:
        if event.error is not None:
            self.publish(Notice(text=str(event.error), level='error', command=event.text))
        elif event.output:
            reports = event.text.split(maxsplit=1)[0] in USAGE_COMMANDS
            usage = usage_report(session_usage(self.session.conversation.messages)) if reports else None
            self.publish(Notice(text=event.output, command=event.text, usage=usage))

    # What the browser does.

    def send(self, text: str, *, images: Sequence[BinaryContent] = ()) -> str | None:
        """Enter a prompt or `/command` as if typed at the prompt, unless it is a bare command that declares a menu.

        That menu opens in the page instead of the terminal: the name of its command is returned, and
        nothing is entered. Otherwise the result is `None`. `images` go with a prompt, which may then be blank.
        """
        text = _entered(text, images=images)
        if not images and (menu := self._menu(text)) is not None:
            return menu
        self.session.send_input(text, images=images)
        return None

    def _menu(self, text: str) -> str | None:
        """The command whose menu bare `text` opens in the page, if it is one."""
        if is_command_input(text) and len(text.split()) == 1 and self.service.host.command_menu(text[1:]) is not None:
            return text[1:]
        return None

    def steer(self, text: str) -> bool:
        """Send a prompt to the running turn now, as Alt+Enter at the prompt does; `False` when nothing takes it.

        A `/command`, or any input while no turn runs, never steers: send it instead, as the prompt then does.
        """
        return self.session.steer(_entered(text))

    def edit_queued(self, input_id: str, text: str) -> None:
        """Rewrite a queued input in place, as Enter on a recalled queued prompt does."""
        text = _entered(text)
        if self._menu(text) is not None:
            raise ValueError(f'{text} opens its menu in the page; send it instead of queueing it.')
        self.session.edit_queued(input_id, text)

    def remove_queued(self, input_id: str) -> None:
        self.session.remove_queued(input_id)

    def steer_queued(self, input_id: str) -> bool:
        """Send a queued prompt to the running turn now, out of the queue; `False` leaves it queued, as the prompt does."""
        entry = next((entry for entry in self.session.queue if entry.id == input_id), None)
        if entry is None:
            raise LookupError(NOT_QUEUED)
        if not self.session.steer(entry.text):
            return False
        self.session.remove_queued(input_id)
        return True

    def switch(self, key: Switch, on: bool) -> None:
        """Flip a run switch as `/set` typed at the prompt does; a running turn keeps the tools it started with."""
        self.session.send_input(f'/set {key} {"true" if on else "false"}')

    async def complete(self, text: str) -> list[CompletionInfo]:
        return [
            CompletionInfo(
                text=completion.text,
                start=-completion.start_position,
                display=completion.display or completion.text,
                meta=completion.display_meta,
            )
            for completion in await self.session.complete(text)
        ]

    def answer(self, request_id: str, answers: Sequence[AskUserAnswer] | None) -> None:
        """Answer one of the session's open questions, or decline it with `None`."""
        self.service.answerer.reply(request_id, answers, session=self.id)


def _entered(text: str, *, images: Sequence[BinaryContent] = ()) -> str:
    """`text` trimmed, as the prompt takes it; shell lines, which run only in the terminal, are refused.

    Blank text is refused too, unless `images` go with it.
    """
    text = text.strip()
    if not text and not images:
        raise ValueError('Type a prompt or a /command.')
    if text.startswith('!'):
        raise ValueError('Shell lines (!) run only in the terminal.')
    return text
