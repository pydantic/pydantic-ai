# pyright: reportUnknownMemberType=false
from __future__ import annotations as _annotations

import os
from typing import TypedDict, cast

from algoliasearch.search.client import SearchClientSync
from bs4 import BeautifulSoup
from mkdocs.config import Config
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page


class AlgoliaRecord(TypedDict):
    content: str
    pageID: str
    abs_url: str
    title: str
    objectID: str


records: list[AlgoliaRecord] = []
# these values should match docs/javascripts/search-worker.js.
ALGOLIA_APP_ID = 'KPPUDTIAVX'
ALGOLIA_INDEX_NAME = 'pydantic-ai-docs'
ALGOLIA_WRITE_API_KEY = os.environ.get('ALGOLIA_WRITE_API_KEY')

# Algolia has a limit of 100kb per record in the paid plan,
# leave some space for the other fields as well.
MAX_CONTENT_LENGTH = 90_000


def on_page_content(html: str, page: Page, config: Config, files: Files) -> str:
    if not ALGOLIA_WRITE_API_KEY:
        return html

    assert page.title is not None, 'Page title must not be None'
    title = cast(str, page.title)

    soup = BeautifulSoup(html, 'html.parser')

    # Clean up presentational and UI elements
    for element in soup.find_all(['autoref']):
        element.decompose()

    # this removes the large source code embeds from Github
    for element in soup.find_all('details'):
        element.decompose()

    # Cleanup code examples
    for extra in soup.find_all('div', attrs={'class': 'language-python highlight'}):
        extra.replace_with(BeautifulSoup(f'<pre>{extra.find("code").get_text()}</pre>', 'html.parser'))

    # Cleanup code examples, part 2
    for extra in soup.find_all('div', attrs={'class': 'language-python doc-signature highlight'}):
        extra.replace_with(BeautifulSoup(f'<pre>{extra.find("code").get_text()}</pre>', 'html.parser'))

    # The API reference generates HTML tables with line numbers, this strips the line numbers cell and goes back to a code block
    for extra in soup.find_all('table', attrs={'class': 'highlighttable'}):
        extra.replace_with(BeautifulSoup(f'<pre>{extra.find("code").get_text()}</pre>', 'html.parser'))

    # Find all h1 and h2 headings
    headings = soup.find_all(['h1', 'h2'])

    # Process each section
    for current_heading in headings:
        heading_id = current_heading.get('id', '')
        section_title = current_heading.get_text().replace('¶', '').strip()

        # Get content until next heading
        content: list[str] = []
        sibling = current_heading.find_next_sibling()
        while sibling and sibling.name not in {'h1', 'h2'}:
            content.append(str(sibling))
            sibling = sibling.find_next_sibling()

        section_html = ''.join(content)

        # Create anchor URL
        anchor_url: str = f'{page.abs_url}#{heading_id}' if heading_id else page.abs_url or ''

        # Create record for this section
        records.append(
            AlgoliaRecord(
                content=section_html,
                pageID=title,
                abs_url=anchor_url,
                title=f'{title} - {section_title}',
                objectID=anchor_url,
            )
        )

    return html


def on_post_build(config: Config) -> None:
    if not ALGOLIA_WRITE_API_KEY:
        return

    client = SearchClientSync(ALGOLIA_APP_ID, ALGOLIA_WRITE_API_KEY)

    for record in records:
        if len(record['content']) > MAX_CONTENT_LENGTH:
            print(
                f"Record with title '{record['title']}' has more than {MAX_CONTENT_LENGTH} characters, {len(record['content'])}."
            )
            print(record['content'])

    # Filter the records from the index if the content is bigger than 100kb, Algolia limit
    filtered_records = list(filter(lambda record: len(record['content']) < MAX_CONTENT_LENGTH, records))
    print(f'Uploading {len(filtered_records)} out of {len(records)} records to Algolia...')

    client.clear_objects(index_name=ALGOLIA_INDEX_NAME)

    client.batch(
        index_name=ALGOLIA_INDEX_NAME,
        batch_write_params={'requests': [{'action': 'addObject', 'body': record} for record in filtered_records]},
    )
