---
hide: [toc]
---

Simple chat app example build with FastAPI.

Demonstrates:

* reusing chat history
* serializing messages

This demonstrates storing chat history between requests and using it to give the model context for new responses.

Most of the complex logic here is in `chat_app.html` which includes the page layout and JavaScript to handle the chat.

## Running the Example

With [dependencies installed and environment variables set](./index.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.chat_app
```

Then open the app at [localhost:8000](http://localhost:8000).

TODO screenshot.

## Example Code

```py title="chat_app.py"
#! pydantic_ai_examples/chat_app.py
```

```html title="chat_app.html"
#! pydantic_ai_examples/chat_app.html
```
