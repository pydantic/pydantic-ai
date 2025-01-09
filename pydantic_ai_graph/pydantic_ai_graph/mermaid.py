from __future__ import annotations as _annotations

import base64
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from annotated_types import Ge, Le
from typing_extensions import TypeAlias, TypedDict, Unpack

from .nodes import BaseNode

if TYPE_CHECKING:
    from .graph import Graph


NodeIdent: TypeAlias = 'type[BaseNode[Any, Any]] | BaseNode[Any, Any] | str'
DEFAULT_HIGHLIGHT_CSS = 'fill:#fdff32'


def generate_code(
    graph: Graph[Any, Any],
    /,
    *,
    start_node: Sequence[NodeIdent] | NodeIdent | None = None,
    highlighted_nodes: Sequence[NodeIdent] | NodeIdent | None = None,
    highlight_css: str = DEFAULT_HIGHLIGHT_CSS,
) -> str:
    """Generate Mermaid code for a graph.

    Args:
        graph: The graph to generate the image for.
        start_node: Identifiers of nodes that start the graph.
        highlighted_nodes: Identifiers of nodes to highlight.
        highlight_css: CSS to use for highlighting nodes.

    Returns: The Mermaid code for the graph.
    """
    start_node_ids = set(node_ids(start_node or ()))
    for node_id in start_node_ids:
        if node_id not in graph.node_defs:
            raise LookupError(f'Start node "{node_id}" is not in the graph.')

    node_order = {node_id: index for index, node_id in enumerate(graph.node_defs)}

    lines = ['graph TD']
    for node in graph.nodes:
        node_id = node.get_id()
        node_def = graph.node_defs[node_id]

        # we use round brackets (rounded box) for nodes other than the start and end
        mermaid_name = f'({node_id})'
        if node_id in start_node_ids:
            lines.append(f'  START --> {node_id}{mermaid_name}')
        if node_def.returns_base_node:
            for next_node_id in graph.nodes:
                lines.append(f'  {node_id}{mermaid_name} --> {next_node_id}')
        else:
            for _, next_node_id in sorted((node_order[node_id], node_id) for node_id in node_def.next_node_ids):
                lines.append(f'  {node_id}{mermaid_name} --> {next_node_id}')
        if node_def.returns_end:
            lines.append(f'  {node_id}{mermaid_name} --> END')

    if highlighted_nodes:
        lines.append('')
        lines.append(f'classDef highlighted {highlight_css}')
        for node_id in node_ids(highlighted_nodes):
            if node_id not in graph.node_defs:
                raise LookupError(f'Highlighted node "{node_id}" is not in the graph.')
            lines.append(f'class {node_id} highlighted')

    return '\n'.join(lines)


def node_ids(node_idents: Sequence[NodeIdent] | NodeIdent) -> Iterable[str]:
    """Get the node IDs from a sequence of node identifiers."""
    if isinstance(node_idents, str):
        node_iter = (node_idents,)
    elif isinstance(node_idents, Sequence):
        node_iter = node_idents
    else:
        node_iter = (node_idents,)

    for node in node_iter:
        if isinstance(node, str):
            yield node
        else:
            yield node.get_id()


class MermaidConfig(TypedDict, total=False):
    """Parameters to configure mermaid chart generation."""

    start_node: Sequence[NodeIdent] | NodeIdent
    """Identifiers of nodes that start the graph."""
    highlighted_nodes: Sequence[NodeIdent] | NodeIdent
    """Identifiers of nodes to highlight."""
    highlight_css: str
    """CSS to use for highlighting nodes."""
    image_type: Literal['jpeg', 'png', 'webp', 'svg', 'pdf']
    """The image type to generate. If unspecified, the default behavior is `'jpeg'`."""
    pdf_fit: bool
    """When using image_type='pdf', whether to fit the diagram to the PDF page."""
    pdf_landscape: bool
    """When using image_type='pdf', whether to use landscape orientation for the PDF.

    This has no effect if using `pdf_fit`.
    """
    pdf_paper: Literal['letter', 'legal', 'tabloid', 'ledger', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    """When using image_type='pdf', the paper size of the PDF."""
    background_color: str
    """The background color of the diagram.

    If None, the default transparent background is used. The color value is interpreted as a hexadecimal color
    code by default (and should not have a leading '#'), but you can also use named colors by prefixing the
    value with `'!'`. For example, valid choices include `background_color='!white'` or `background_color='FF0000'`.
    """
    theme: Literal['default', 'neutral', 'dark', 'forest']
    """The theme of the diagram. Defaults to 'default'."""
    width: int
    """The width of the diagram."""
    height: int
    """The height of the diagram."""
    scale: Annotated[float, Ge(1), Le(3)]
    """The scale of the diagram.

    The scale must be a number between 1 and 3, and you can only set a scale if one or both of width and height are set.
    """


def request_image(
    graph: Graph[Any, Any],
    /,
    **kwargs: Unpack[MermaidConfig],
) -> bytes:
    """Generate an image of a Mermaid diagram using [mermaid.ink](https://mermaid.ink).

    Args:
        graph: The graph to generate the image for.
        **kwargs: Additional parameters to configure mermaid chart generation.

    Returns: The image data.
    """
    import httpx

    code = generate_code(
        graph,
        start_node=kwargs.get('start_node'),
        highlighted_nodes=kwargs.get('highlighted_nodes'),
        highlight_css=kwargs.get('highlight_css', DEFAULT_HIGHLIGHT_CSS),
    )
    code_base64 = base64.b64encode(code.encode()).decode()

    params: dict[str, str | bool] = {}
    if kwargs.get('image_type') == 'pdf':
        url = f'https://mermaid.ink/pdf/{code_base64}'
        if kwargs.get('pdf_fit'):
            params['fit'] = True
        if kwargs.get('pdf_landscape'):
            params['landscape'] = True
        if pdf_paper := kwargs.get('pdf_paper'):
            params['paper'] = pdf_paper
    elif kwargs.get('image_type') == 'svg':
        url = f'https://mermaid.ink/svg/{code_base64}'
    else:
        url = f'https://mermaid.ink/img/{code_base64}'

        if image_type := kwargs.get('image_type'):
            params['type'] = image_type

    if background_color := kwargs.get('background_color'):
        params['bgColor'] = background_color
    if theme := kwargs.get('theme'):
        params['theme'] = theme
    if width := kwargs.get('width'):
        params['width'] = str(width)
    if height := kwargs.get('height'):
        params['height'] = str(height)
    if scale := kwargs.get('scale'):
        params['scale'] = str(scale)

    response = httpx.get(url, params=params)
    response.raise_for_status()
    return response.content


def save_image(
    path: Path | str,
    graph: Graph[Any, Any],
    /,
    **kwargs: Unpack[MermaidConfig],
) -> None:
    """Generate an image of a Mermaid diagram using [mermaid.ink](https://mermaid.ink) and save it to a local file.

    Args:
        path: The path to save the image to.
        graph: The graph to generate the image for.
        **kwargs: Additional parameters to configure mermaid chart generation.
    """
    if isinstance(path, str):
        path = Path(path)

    if 'image_type' not in kwargs:
        ext = path.suffix.lower()[1:]
        # no need to check for .jpeg/.jpg, as it is the default
        if ext in ('png', 'webp', 'svg', 'pdf'):
            kwargs['image_type'] = ext

    image_data = request_image(graph, **kwargs)
    path.write_bytes(image_data)
