"""Image bytes survive editing, queued turns, hooks, and persisted history."""

import io
from array import array
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import anyio
import pytest
from PIL import Image
from prompt_toolkit import PromptSession
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import BinaryContent, ModelMessagesTypeAdapter, ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from rich.console import Console

from pydantic_clai2 import Session, chat, image_input
from pydantic_clai2.image_input import (
    ImageBuffer,
    ImageInput,
    clipboard_images,
    encode_image,
    pasted_paths,
    read_image,
    read_images,
)
from pydantic_clai2.prompt_surface import PromptSurface
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    path = tmp_path / 'screen shot.PNG'
    Image.new('RGB', (3, 2), color='red').save(path)
    return path


@pytest.mark.parametrize('mode', ['RGB', 'RGBA', 'P', 'L'])
def test_normalize(mode: str) -> None:
    image = encode_image(Image.new(mode, (3, 2)))
    assert image.media_type == 'image/png'
    with Image.open(io.BytesIO(image.data)) as restored:
        assert restored.size == (3, 2)
        assert restored.mode == ('RGBA' if mode == 'RGBA' else 'RGB')


def test_size_limits(image_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_input, 'MAX_PIXELS', 5)
    with pytest.raises(ValueError, match='megapixel'):
        read_image(image_path)
    monkeypatch.setattr(image_input, 'MAX_PIXELS', 6)
    monkeypatch.setattr(image_input, 'MAX_IMAGE_BYTES', 1)
    with pytest.raises(ValueError, match='file exceeds'):
        read_image(image_path)
    with pytest.raises(ValueError, match='attachment limit'):
        encode_image(Image.new('RGB', (1, 1)))


@pytest.mark.parametrize('quoting', ['bare', 'single', 'double', 'escaped', 'multiple', 'lines'])
def test_file_paste(image_path: Path, quoting: str) -> None:
    path = str(image_path)
    text = {
        'bare': path,
        'single': f"'{path}'",
        'double': f'"{path}"',
        'escaped': image_path.as_posix().replace(' ', '\\ '),
        'multiple': f'"{path}" "{path}"',
        'lines': f'"{path}"\n"{path}"',
    }[quoting]
    assert pasted_paths(text) == [image_path] * (2 if quoting in ('multiple', 'lines') else 1)


@pytest.mark.parametrize(
    'text',
    ['', 'ordinary text', "'unclosed", '/missing.png', 'a' * 40000, '\x00', 'a' * 300],
    ids=['empty', 'text', 'unclosed-quote', 'missing', 'oversized', 'nul', 'long-filename'],
)
def test_text_paste_is_not_an_attachment(text: str) -> None:
    assert pasted_paths(text) == []


def test_mixed_and_non_image_paths(image_path: Path) -> None:
    text = image_path.with_suffix('.txt')
    text.write_text('not an image')
    assert pasted_paths(str(text)) == []
    assert pasted_paths(f'"{image_path}" "{text}"') == []
    image_path.write_text('not really a PNG')
    with pytest.raises(OSError):
        read_image(image_path)


@pytest.mark.parametrize('kind', ['image', 'files', 'empty', 'empty_files', 'error'])
def test_clipboard_backend(image_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    def grab() -> Image.Image | list[str] | None:
        if kind == 'image':
            return Image.open(image_path)
        if kind == 'files':
            return [str(image_path)]
        if kind == 'empty_files':
            return []
        if kind == 'error':
            raise NotImplementedError('No clipboard backend')
        return None

    monkeypatch.setattr(image_input.ImageGrab, 'grabclipboard', grab)
    if kind in ('image', 'files'):
        assert clipboard_images() == [read_image(image_path)]
    elif kind == 'error':
        with pytest.raises(NotImplementedError):
            clipboard_images()
    else:
        with pytest.raises(ValueError, match='No image'):
            clipboard_images()


def test_attachment_lifetime_and_limits(image_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = ImageInput()
    image = read_image(image_path)
    first = images.attach([image])
    second = images.attach([image])
    assert images.resolve(f'describe {first}{second}') == ('describe', [image, image])
    images.retain([first])
    assert images.resolve(first) == ('', [image])
    with pytest.raises(ValueError, match='expired'):
        images.resolve(second)
    monkeypatch.setattr(image_input, 'MAX_PENDING_BYTES', len(image.data))
    with pytest.raises(ValueError, match='32 MiB'):
        images.attach([image])
    assert len(images.pending) == 1
    images.retain([])
    assert images.pending == {}
    assert images.resolve('text only') == ('text only', [])


@pytest.mark.parametrize('key', ['\x16', '\x1bv', 'path', 'text', 'error', 'invalid'])
async def test_editor_bindings(image_path: Path, monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    images = ImageInput()
    image = read_image(image_path)

    def grab() -> list[BinaryContent]:
        if key == 'error':
            raise OSError('clipboard unavailable')
        return [image]

    monkeypatch.setattr(image_input, 'clipboard_images', grab)
    if key == 'path':
        typed = f'\x1b[200~"{image_path}"\x1b[201~'
    elif key == 'text':
        typed = '\x1b[200~ordinary\r\ntext\x1b[201~'
    elif key == 'invalid':
        image_path.write_text('invalid PNG')
        typed = f'\x1b[200~{image_path}\x1b[201~'
    else:
        typed = '\x16' if key == 'error' else key
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        prompt = PromptSession[str](key_bindings=images.bindings())
        pipe.send_text(typed + 'caption\n')
        result = await prompt.prompt_async('> ')
    if key in ('error', 'invalid'):
        assert 'Image paste failed' in images.notice
        assert result == 'caption'
        assert not images.pending
    elif key == 'text':
        assert result == 'ordinary\ntextcaption'
        assert not images.pending
    else:
        assert images.resolve(result) == ('caption', [image])


@pytest.mark.parametrize('terminal', [False, True])
async def test_shell_image_and_text_turns(image_path: Path, tmp_path: Path, terminal: bool) -> None:
    observed: list[Sequence[str | BinaryContent] | str] = []

    class Capture(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            message = ctx.messages[-1]
            assert isinstance(message, ModelRequest)
            part = message.parts[0]
            assert isinstance(part, UserPromptPart)
            if isinstance(part.content, str):
                observed.append(part.content)
            else:
                assert all(isinstance(item, (str, BinaryContent)) for item in part.content)
                observed.append([item for item in part.content if isinstance(item, (str, BinaryContent))])
            return request_context

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(15):
        pipe.send_text(f'\x1b[200~"{image_path}"\x1b[201~describe\ntext only\n/exit\n')
        await chat(
            Agent(TestModel(), deps_type=type(None), capabilities=[Capture()]),
            deps=None,
            console=Console(file=io.StringIO(), force_terminal=terminal),
            store=SettingsStore(tmp_path / 'config.db'),
        )
    assert observed == [['describe', read_image(image_path)], 'text only']


async def test_images_persist_and_resume(image_path: Path, tmp_path: Path) -> None:
    conversations = SqliteConversationStore(database=tmp_path / 'history.db')
    agent = Agent(TestModel())
    session = Session(agent, deps=None, conversations=conversations, workspace=tmp_path)
    image = read_image(image_path)
    await session.prompt('', images=[image])
    restored = Session(agent, deps=None, conversations=conversations, workspace=tmp_path)
    await restored.resume(session.summary.id)
    assert ModelMessagesTypeAdapter.dump_json(restored.messages) == ModelMessagesTypeAdapter.dump_json(session.messages)
    first = restored.messages[0].parts[0]
    assert isinstance(first, UserPromptPart) and not isinstance(first.content, str)
    assert first.content[0] == ''
    assert isinstance(first.content[1], BinaryContent) and first.content[1].data == image.data
    await restored.prompt('follow up')
    assert ModelMessagesTypeAdapter.dump_json(restored.messages[:1]) == ModelMessagesTypeAdapter.dump_json(
        session.messages[:1]
    )


async def test_failed_image_turn_is_durable(image_path: Path, tmp_path: Path) -> None:
    session = Session(
        Agent(TestModel()),
        deps=None,
        conversations=SqliteConversationStore(database=tmp_path / 'history.db'),
        workspace=tmp_path,
    )

    async def unavailable(name: str) -> str:
        raise ValueError('provider unavailable')

    session.model = 'missing'
    session.resolve_model = unavailable
    image = read_image(image_path)
    with pytest.raises(ValueError, match='unavailable'):
        await session.prompt('caption', images=[image])
    assert session.conversations is not None
    record = await session.conversations.get(conversation_id=session.summary.id)
    first = record.messages[0].parts[0]
    assert isinstance(first, UserPromptPart) and not isinstance(first.content, str)
    assert first.content[0] == 'caption'
    assert isinstance(first.content[1], BinaryContent) and first.content[1].data == image.data


@pytest.mark.parametrize('action', ['rewrite', 'cancel', 'fail', 'expired'])
async def test_image_hooks_and_expired_history(image_path: Path, tmp_path: Path, action: str) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir()
    (store.plugins_dir / 'caption.py').write_text(
        'def activate(host):\n'
        "    @host.on('turn_start')\n"
        '    async def start(event):\n'
        "        assert event.text == 'caption'\n"
        + {
            'rewrite': "        event.text = 'rewritten'\n",
            'cancel': "        event.cancel('image rejected')\n",
            'fail': "        raise ValueError('image rejected')\n",
            'expired': '        raise AssertionError("expired images should not run hooks")\n',
        }[action]
    )
    requests: list[UserPromptPart] = []

    class Capture(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            part = ctx.messages[-1].parts[0]
            assert isinstance(part, UserPromptPart)
            requests.append(part)
            return request_context

    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(15):
        text = '[image:1234abcd]' if action == 'expired' else f'\x1b[200~{image_path}\x1b[201~'
        pipe.send_text(text + 'caption\n/exit\n')
        await chat(
            Agent(TestModel(), deps_type=type(None), capabilities=[Capture()]),
            deps=None,
            console=Console(file=output),
            store=store,
        )
    if action == 'rewrite':
        assert len(requests) == 1
        assert requests[0].content == ['rewritten', read_image(image_path)]
    else:
        assert requests == []
        assert ('expired' if action == 'expired' else 'image rejected') in output.getvalue()


async def test_cancelled_image_turn_is_durable(image_path: Path, tmp_path: Path) -> None:
    entered = anyio.Event()
    scope = anyio.CancelScope()

    class Pause(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            entered.set()
            await anyio.sleep_forever()

    session = Session(
        Agent(TestModel(), deps_type=type(None), capabilities=[Pause()]),
        deps=None,
        conversations=SqliteConversationStore(database=tmp_path / 'history.db'),
        workspace=tmp_path,
    )
    image = read_image(image_path)

    async def run() -> None:
        with scope:
            await session.prompt('caption', images=[image])

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(run)
        await entered.wait()
        scope.cancel()
    assert session.conversations is not None
    record = await session.conversations.get(conversation_id=session.summary.id)
    assert record.summary.outcome == 'cancelled'
    part = record.messages[0].parts[0]
    assert isinstance(part, UserPromptPart) and not isinstance(part.content, str)
    assert isinstance(part.content[1], BinaryContent) and part.content[1].data == image.data


def test_batch_limit_stops_before_reading_remaining_files(image_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_input, 'MAX_PENDING_BYTES', 1)
    with pytest.raises(ValueError, match='Pasted images exceed'):
        read_images([image_path, image_path.with_name('does-not-exist.png')])


def test_palette_transparency() -> None:
    image = Image.new('P', (1, 1))
    image.info['transparency'] = 0
    encoded = encode_image(image)
    with Image.open(io.BytesIO(encoded.data)) as restored:
        assert restored.mode == 'RGBA'
        assert restored.getpixel((0, 0)) == (0, 0, 0, 0)


def test_marker_collision_does_not_replace_an_image(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = iter([UUID(int=1), UUID(int=2), UUID('12345678-0000-0000-0000-000000000000')])
    monkeypatch.setattr(image_input, 'uuid4', lambda: next(ids))
    images = ImageInput()
    red = encode_image(Image.new('RGB', (1, 1), color='red'))
    blue = encode_image(Image.new('RGB', (1, 1), color='blue'))
    first = images.attach([red])
    second = images.attach([blue])
    assert first != second
    assert images.resolve(first + second) == ('', [red, blue])


@pytest.mark.parametrize('orientation', [1, 3, 6, 8])
def test_exif_orientation_is_applied_and_metadata_removed(orientation: int) -> None:
    image = Image.new('RGB', (3, 2), color='red')
    image.putpixel((0, 0), (0, 0, 255))
    exif = image.getexif()
    exif[274] = orientation
    exif[315] = 'private camera owner'
    image.info['exif'] = exif.tobytes()
    encoded = encode_image(image)
    with Image.open(io.BytesIO(encoded.data)) as restored:
        assert restored.size == ((3, 2) if orientation in (1, 3) else (2, 3))
        blue_pixel = {1: (0, 0), 3: (2, 1), 6: (1, 0), 8: (0, 2)}[orientation]
        assert restored.getpixel(blue_pixel) == (0, 0, 255)
        assert not restored.getexif()


def test_encoder_buffer_rejects_writes_before_allocating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_input, 'MAX_IMAGE_BYTES', 4)
    with ImageBuffer() as output:
        assert output.write(b'ab') == 2
        assert output.write(bytearray(b'cd')) == 2
        with pytest.raises(ValueError, match='attachment limit'):
            output.write(b'e')
        assert output.getvalue() == b'abcd'
        output.seek(0)
        # Count bytes, not elements, for non-byte buffers.
        with pytest.raises(ValueError, match='attachment limit'):
            output.write(memoryview(array('I', [1, 2])))
        assert output.getvalue() == b'abcd'


@pytest.mark.parametrize(
    'text',
    [
        r'\\attacker.example\share\image.png',
        '//attacker.example/share/image.png',
        r'\\?\C:\image.png',
        r'\\.\device\image.png',
    ],
)
def test_network_paths_are_rejected_without_io(text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_stat(path: Path, *, follow_symlinks: bool = True) -> None:
        raise AssertionError('Network paths must be rejected before filesystem access')

    monkeypatch.setattr(Path, 'stat', no_stat)
    assert pasted_paths(text) == []
    with pytest.raises(ValueError, match='UNC and device'):
        read_image(Path(text))
    monkeypatch.setattr(image_input.ImageGrab, 'grabclipboard', lambda: [text])
    with pytest.raises(ValueError, match='UNC and device'):
        clipboard_images()


def test_expanded_home_cannot_bypass_network_path_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    def network_home(path: Path) -> Path:
        return Path('//server/share/image.png')

    monkeypatch.setattr(Path, 'expanduser', network_home)
    assert pasted_paths('~/image.png') == []


def test_unknown_home_is_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_home(path: Path) -> Path:
        raise RuntimeError('Could not determine home directory')

    monkeypatch.setattr(Path, 'expanduser', no_home)
    assert pasted_paths('~unknown/image.png') == []


def test_renamed_unsupported_format_is_rejected(image_path: Path) -> None:
    Image.new('RGB', (1, 1)).save(image_path, format='PPM')
    with pytest.raises(OSError):
        read_image(image_path)


@pytest.mark.parametrize('terminal', [False, True])
async def test_image_can_be_retried_after_selecting_a_model(
    image_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: bool
) -> None:
    monkeypatch.setattr(image_input, 'uuid4', lambda: UUID('12345678-0000-0000-0000-000000000000'))
    agent = Agent()
    output = io.StringIO()
    transcript: list[str] = []

    class Surface(PromptSurface):
        def write(self, text: str) -> int:
            transcript.append(text)
            return super().write(text)

    monkeypatch.setattr('pydantic_clai2.live_prompt.PromptSurface', Surface)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(15):
        pipe.send_text(f'\x1b[200~{image_path}\x1b[201~caption\n/set model test\n[image:12345678]caption\n/exit\n')
        await chat(
            agent,
            deps=None,
            console=Console(file=output, force_terminal=terminal),
            store=SettingsStore(tmp_path / 'settings.db'),
        )
    assert 'Choose a model first' in output.getvalue()
    assert 'expired' not in output.getvalue()
    assert 'success' in (''.join(transcript) if terminal else output.getvalue())
