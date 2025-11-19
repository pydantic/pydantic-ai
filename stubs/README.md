Stub files (`*.pyi`) contain type hints used only by type checkers, not at
runtime. They were introduced in
[PEP 484](https://peps.python.org/pep-0484/#stub-files). For example, the
[`typeshed`](https://github.com/python/typeshed) repository maintains a
collection of such stubs for the Python standard library and some third-party
libraries.

The `./stubs` folder contains type information only for the parts of third-party
dependencies used in the `pydantic-ai` codebase. These stubs must be manually
maintained. When a dependency's API changes, both the codebase and the stubs
need to be updated. There are two ways to update the stubs:

(1) **Manual update:** Check the dependency's source code and copy the type
information to `./stubs`. Take for example the `from_pretrained()` method of the
`Llama` class in `llama-cpp-python`. The
[source code](https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py#L2240)
contains the type information that is copied to `./stubs/llama_cpp.pyi`. This
eliminates the need for `# type: ignore` comments in the codebase.

(2) **Update with AI coding assistants:** Most dependencies maintain `llms.txt`
and `llms-full.txt` files with their documentation. This information is compiled
by [Context7](https://context7.com). For example, the `llama-cpp-python` library
is documented [here](https://github.com/abetlen/llama-cpp-python). MCP servers
such as [this one by Upstash](https://github.com/upstash/context7) provide AI
coding assistants access to Context7. AI coding assistants such as VS Code
Copilot or Cursor can reliably generate and update the stubs.
