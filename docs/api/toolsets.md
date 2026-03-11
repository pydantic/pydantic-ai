# `pydantic_ai.toolsets`

::: pydantic_ai.toolsets
    options:
        members:
        - AbstractToolset
        - CombinedToolset
        - ExternalToolset
        - ApprovalRequiredToolset
        - FilteredToolset
        - FunctionToolset
        - PrefixedToolset
        - RenamedToolset
        - PreparedToolset
        - WrapperToolset
        - ShellToolset
        - TextEditorToolset
        - ApplyPatchToolset
        - ToolsetFunc

::: pydantic_ai.toolsets.shell
    options:
        members:
        - ShellExecutor
        - ShellOutput

::: pydantic_ai.toolsets.text_editor
    options:
        members:
        - TextEditorCommand
        - TextEditorOutput
        - TextEditorExecuteFunc
        - ViewCommand
        - StrReplaceCommand
        - CreateCommand
        - InsertCommand

::: pydantic_ai.toolsets.apply_patch
    options:
        members:
        - ApplyPatchOperation
        - ApplyPatchOutput
        - ApplyPatchExecuteFunc

::: pydantic_ai.toolsets.fastmcp
