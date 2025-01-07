from __future__ import annotations as _annotations

import base64
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from annotated_types import Ge, Le

if TYPE_CHECKING:
    from .graph import Graph
    from .nodes import BaseNode


NodeIdent = type[BaseNode[Any, Any]] | str
DEFAULT_HIGHLIGHT_CSS = 'fill:#f9f'


def generate_code(
    graph: Graph[Any, Any],
    start_node_ids: set[str],
    *,
    highlighted_nodes: Sequence[NodeIdent] | None = None,
    highlight_css: str = DEFAULT_HIGHLIGHT_CSS,
) -> str:
    """Generate Mermaid code for a graph.

    Args:
        graph: The graph to generate the image for.
        start_node_ids: Identifiers of nodes that start the graph.
        highlighted_nodes: Identifiers of nodes to highlight.
        highlight_css: CSS to use for highlighting nodes.

    Returns: The Mermaid code for the graph.
    """
    for node_id in start_node_ids:
        if node_id not in graph.node_defs:
            raise LookupError(f'Start node "{node_id}" is not in the graph.')

    node_order = {node_id: index for index, node_id in enumerate(graph.node_defs)}

    lines = ['graph TD']
    for node in graph.nodes:
        node_id = node.get_id()
        node_def = graph.node_defs[node_id]
        if node_id in start_node_ids:
            lines.append(f'  START --> {node_id}')
        if node_def.returns_base_node:
            for next_node_id in graph.nodes:
                lines.append(f'  {node_id} --> {next_node_id}')
        else:
            for _, next_node_id in sorted((node_order[node_id], node_id) for node_id in node_def.next_node_ids):
                lines.append(f'  {node_id} --> {next_node_id}')
        if node_def.returns_end:
            lines.append(f'  {node_id} --> END')

    if highlighted_nodes:
        lines.append('')
        lines.append(f'classDef highlighted {highlight_css}')
        for node in highlighted_nodes:
            node_id = node if isinstance(node, str) else node.get_id()
            if node_id not in graph.node_defs:
                raise LookupError(f'Highlighted node "{node_id}" is not in the graph.')
            lines.append(f'class {node_id} highlighted')

    return '\n'.join(lines)


ImageType = Literal['jpeg', 'png', 'webp', 'svg', 'pdf']
PdfPaper = Literal['letter', 'legal', 'tabloid', 'ledger', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
Theme = Literal['default', 'neutral', 'dark', 'forest']


def request_image(
    graph: Graph[Any, Any],
    start_node_ids: set[str],
    *,
    highlighted_nodes: Sequence[NodeIdent] | None = None,
    highlight_css: str = 'fill:#f9f,stroke:#333,stroke-width:4px',
    image_type: ImageType | str | None = None,
    pdf_fit: bool = False,
    pdf_landscape: bool = False,
    pdf_paper: PdfPaper | None = None,
    bg_color: str | None = None,
    theme: Theme | None = None,
    width: int | None = None,
    height: int | None = None,
    scale: Annotated[float, Ge(1), Le(3)] | None = None,
) -> bytes:
    """Generate an image of a Mermaid diagram using [mermaid.ink](https://mermaid.ink).

    Args:
        graph: The graph to generate the image for.
        start_node_ids: IDs of start nodes of the graph.
        highlighted_nodes: Identifiers of nodes to highlight.
        highlight_css: CSS to use for highlighting nodes.
        image_type: The image type to generate. If unspecified, the default behavior is `'jpeg'`.
        pdf_fit: When using image_type='pdf', whether to fit the diagram to the PDF page.
        pdf_landscape: When using image_type='pdf', whether to use landscape orientation for the PDF.
            This has no effect if using `pdf_fit`.
        pdf_paper: When using image_type='pdf', the paper size of the PDF.
        bg_color: The background color of the diagram. If None, the default transparent background is used.
            The color value is interpreted as a hexadecimal color code by default (and should not have a leading '#'),
            but you can also use named colors by prefixing the value with `'!'`.
            For example, valid choices include `bg_color='!white'` or `bg_color='FF0000'`.
        theme: The theme of the diagram. Defaults to 'default'.
        width: The width of the diagram.
        height: The height of the diagram.
        scale: The scale of the diagram. The scale must be a number between 1 and 3, and you can only set
            a scale if one or both of width and height are set.

    Returns: The image data.
    """
    import httpx

    code = generate_code(graph, start_node_ids, highlighted_nodes=highlighted_nodes, highlight_css=highlight_css)
    code_base64 = base64.b64encode(code.encode()).decode()

    params: dict[str, str] = {}
    if image_type == 'pdf':
        url = f'https://mermaid.ink/pdf/{code_base64}'
        if pdf_fit:
            params['fit'] = ''
        if pdf_landscape:
            params['landscape'] = ''
        if pdf_paper:
            params['paper'] = pdf_paper
    elif image_type == 'svg':
        url = f'https://mermaid.ink/svg/{code_base64}'
    else:
        url = f'https://mermaid.ink/img/{code_base64}'

        if image_type:
            params['type'] = image_type

    if bg_color:
        params['bgColor'] = bg_color
    if theme:
        params['theme'] = theme
    if width:
        params['width'] = str(width)
    if height:
        params['height'] = str(height)
    if scale:
        params['scale'] = str(scale)

    response = httpx.get(url, params=params)
    response.raise_for_status()
    return response.content


def save_image(
    path: Path | str,
    graph: Graph[Any, Any],
    start_node_ids: set[str],
    *,
    highlighted_nodes: Sequence[NodeIdent] | None = None,
    highlight_css: str = 'fill:#f9f,stroke:#333,stroke-width:4px',
    image_type: ImageType | None = None,
    pdf_fit: bool = False,
    pdf_landscape: bool = False,
    pdf_paper: PdfPaper | None = None,
    bg_color: str | None = None,
    theme: Theme | None = None,
    width: int | None = None,
    height: int | None = None,
    scale: Annotated[float, Ge(1), Le(3)] | None = None,
) -> None:
    """Generate an image of a Mermaid diagram using [mermaid.ink](https://mermaid.ink) and save it to a local file.

    Args:
        path: The path to save the image to.
        graph: The graph to generate the image for.
        start_node_ids: IDs of start nodes of the graph.
        highlighted_nodes: Identifiers of nodes to highlight.
        highlight_css: CSS to use for highlighting nodes.
        image_type: The image type to generate. If unspecified, the default behavior is `'jpeg'`.
        pdf_fit: When using image_type='pdf', whether to fit the diagram to the PDF page.
        pdf_landscape: When using image_type='pdf', whether to use landscape orientation for the PDF.
            This has no effect if using `pdf_fit`.
        pdf_paper: When using image_type='pdf', the paper size of the PDF.
        bg_color: The background color of the diagram. If None, the default transparent background is used.
            The color value is interpreted as a hexadecimal color code by default (and should not have a leading '#'),
            but you can also use named colors by prefixing the value with `'!'`.
            For example, valid choices include `bg_color='!white'` or `bg_color='FF0000'`.
        theme: The theme of the diagram. Defaults to 'default'.
        width: The width of the diagram.
        height: The height of the diagram.
        scale: The scale of the diagram. The scale must be a number between 1 and 3, and you can only set
            a scale if one or both of width and height are set.
    """
    if isinstance(path, str):
        path = Path(path)

    if image_type is None:
        ext = path.suffix.lower()
        # no need to check for .jpeg/.jpg, as it is the default
        if ext in {'.png', '.webp', '.svg', '.pdf'}:
            image_type = cast(ImageType, ext[1:])

    image_data = request_image(
        graph,
        start_node_ids,
        highlighted_nodes=highlighted_nodes,
        highlight_css=highlight_css,
        image_type=image_type,
        pdf_fit=pdf_fit,
        pdf_landscape=pdf_landscape,
        pdf_paper=pdf_paper,
        bg_color=bg_color,
        theme=theme,
        width=width,
        height=height,
        scale=scale,
    )
    Path(path).write_bytes(image_data)
