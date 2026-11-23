"""The tests no CI cell runs, and why, for `.github/scripts/never_ran.py`.

See "Tests that run in no cell" in `tests/AGENTS.md`.
"""

from __future__ import annotations as _annotations

from .manifests import Exempt


def _skipped_rule(why: str) -> Exempt:
    """A workspace conformance rule the suite skips for a backend, with the suite's reason."""
    return Exempt(f'The conformance suite skips this rule for the backend: {why}.')


_AZURE_REALTIME = Exempt("No recording: Azure's realtime credentials return 401, so the case cannot be recorded.")
_BROWSER_USE = Exempt(
    'Needs `browser_use`, which only `test-harness-browser-use` installs, and that job runs only '
    '`tests/harness/browser_use/` and the `BrowserUse` cases of `tests/harness/test_capability_contracts.py`.'
)
_DOCS_EXAMPLE = Exempt('The docs mark the example `test="skip"`, so its code is at most checked, never run.')
_GATEWAY_INFERENCE = Exempt('The test skips gateway providers.')
_GATEWAY_LIVE = Exempt('Calls the live gateway (`--run-gateway-live`); `gateway-model-health.yml` runs it each Monday.')
_INJECTED_EXTRA_ERROR = Exempt(
    'The case injects the missing-extra error, checks its install hint and skips, as for a user without the extra.'
)
_ISSUE_8824 = Exempt('Skipped while it hangs or flakes in CI: https://github.com/pydantic/pydantic-ai/issues/8824.')
_MISSING_CLASSIFIER = Exempt(
    "Imports `Classifier` from `pydantic_evals.evaluators`, which pydantic-evals doesn't define, so its import guard "
    'skips it in every cell.'
)
_NO_NATIVE_TOOL = Exempt(
    'The provider has no native implementation of the capability, and the capability has no local fallback.'
)
_NO_UPLOADED_FILE = Exempt(
    '`UPLOADED_FILE_IDS` has no uploaded file of this media type for the provider; adding one needs a live upload.'
)
_NO_UPLOADED_FILE_SUPPORT = Exempt('The provider does not support `UploadedFile`.')
_PLACEHOLDER_ID = Exempt(
    'Not a test case: the id pytest gives an empty parametrization (`NOTSET`), or a case id that names a stand-in for '
    'an optional dependency that is missing. Where the dependency is installed, the cases run under their real ids.'
)
_PS_FALLBACK = Exempt(
    'Covers the `ps` fallback, which runs only on a POSIX host without `/proc`, such as macOS, and no macOS cell runs '
    'the harness tests.'
)
_RENDER_LIVE = Exempt("Render's live tier needs `RENDER_API_KEY` and `RENDER_WORKSPACE_ID`, which no workflow has.")
_S3_LIVE = Exempt('Needs a live S3-compatible endpoint and its `S3_*` credentials, which no workflow has.')
_SANDBOX_LIVE = Exempt(
    "The provider's live tier, against the real service: `sandbox-live.yml` runs it, with the service's credentials or "
    'a server it starts, on its daily schedule or a dispatch.'
)
_SIGNATURE_BLOCK = Exempt('An illustrative signature block in the docs, not runnable Python.')
_SKILL_EXAMPLE = Exempt('The skill marks the example `test="skip"`, so its code is at most linted, never run.')
_TEMPORAL_CERTIFI = Exempt(
    'Skipped: replaying this MCP toolset workflow trips the Temporal sandbox with '
    '`Module certifi was imported after initial workflow load`.'
)
_TOGETHER_CONTINUATION = Exempt("`SKIP_REASONS` skips it: Together's backend returns 500 on the tool continuation.")
_UI_ROUNDTRIP = Exempt(
    'Needs Node and the JavaScript UI clients, which no `ci.yml` cell installs; `ui-roundtrip.yml` runs it on every '
    'pull request and push to `main`.'
)
_VERTEX_PROJECT = Exempt('The Vertex AI cases skip without `GOOGLE_CLOUD_PROJECT`, which no cell sets.')
_VM_LIVE = Exempt('The live tier boots VMs, so it needs a host that can run them, and no workflow runs it.')
_NO_ATTACH_BACKEND = _skipped_rule('provide the `attach_backend` fixture to enable this rule')
_NO_COMMANDS = _skipped_rule('backend does not implement SupportsCommands')
_NO_COMMAND_BYTE_STREAM = _skipped_rule('fake has no command byte stream')
_NO_COMMAND_TO_INTERRUPT = _skipped_rule('backend has no running POSIX command to interrupt')
_NO_DESTRUCTIVE_ATTACH = _skipped_rule(
    'provide `destructive_backend`, `attach_backend` and `destroy_environment` to enable this rule'
)
_NO_DESTRUCTIVE_BACKEND = _skipped_rule('provide `destructive_backend` and `destroy_environment` to enable this rule')
_NO_DIRECT_EXIT = _skipped_rule(
    'backend cannot observe the direct command exit independently of inherited output pipes'
)
_NO_FIFOS = _skipped_rule('in-memory fakes and Windows have no FIFOs')
_NO_FOREGROUND_PROCESS = _skipped_rule('fake has no foreground process to cancel')
_NO_FRESH_BACKEND = _skipped_rule('backend does not provide a fresh_backend factory')
_NO_PATH_TRAVERSAL = _skipped_rule('in-memory fake has no real path traversal')
_NO_PERMISSIONS = _skipped_rule('in-memory fake has no permissions')
_NO_POSIX_PROCESSES = _skipped_rule('backend does not run real POSIX processes')
_NO_REALPATH = _skipped_rule('backend does not implement SupportsRealpath')
_NO_SYMLINKS = _skipped_rule('in-memory fake cannot create symlinks')
_PAGING_IN_LIVE_VM = _skipped_rule('shell paging at a host process per 64 KiB; the live tier runs it in a VM')
_PAGING_IN_SMOLVM = _skipped_rule('shell paging at a host process per 64 KiB; smolvm itself runs it in a VM')
_PAGING_PER_COMMAND = _skipped_rule(
    'the shell filesystem pages a large file at a command per 64 KiB; the live tier runs it'
)
_PAGING_PER_CONNECTION = _skipped_rule(
    'the same shell paging as `TestRunOnlyWorkspaceBackend`, at a connection per 64 KiB'
)
_PAGING_PER_EXEC = _skipped_rule('the same shell paging as `TestRunOnlyWorkspaceBackend`, at an exec per 64 KiB')
_WINDOWS_ACLS = _skipped_rule('Windows permissions are access control lists, which `chmod` does not set')

NEVER_RAN_EXEMPT: dict[str, Exempt] = {
    'tests/durable_exec/temporal/test_durability.py::test_durability_mcp_dynamic_toolset_in_workflow@temporal-durability': _TEMPORAL_CERTIFI,
    'tests/durable_exec/temporal/test_durability.py::test_durability_mcptoolset_in_workflow@temporal-durability': _TEMPORAL_CERTIFI,
    'tests/evals/test_dataset.py::test_generate_dataset_bare_none_type_argument_completes[NOTSET]': _PLACEHOLDER_ID,
    'tests/evals/test_dataset.py::test_to_file_roundtrip_bare_none_type_argument[NOTSET]': _PLACEHOLDER_ID,
    'tests/harness/boxd_sandbox/test_boxd_live.py': _SANDBOX_LIVE,
    'tests/harness/boxd_sandbox/test_conformance.py::TestLiveBoxdSandboxBackend': _SANDBOX_LIVE,
    'tests/harness/boxlite_sandbox/test_boxlite_live.py': _VM_LIVE,
    'tests/harness/boxlite_sandbox/test_conformance.py::TestFakeBoxLiteSandboxBackend::test_large_file_round_trip': _PAGING_IN_LIVE_VM,
    'tests/harness/boxlite_sandbox/test_conformance.py::TestFakeBoxLiteSandboxBackend::test_native_realpath_keeps_the_working_dir_and_missing_names': _NO_REALPATH,
    'tests/harness/boxlite_sandbox/test_conformance.py::TestLiveBoxLiteSandboxBackend': _VM_LIVE,
    'tests/harness/bubblewrap_sandbox/test_conformance.py::TestBubblewrapAroundSSH::test_background_child_does_not_hold_up_completed_command': _NO_DIRECT_EXIT,
    'tests/harness/bubblewrap_sandbox/test_conformance.py::TestBubblewrapAroundSSH::test_concurrent_first_use_shares_one_environment': _NO_FRESH_BACKEND,
    'tests/harness/bubblewrap_sandbox/test_conformance.py::TestBubblewrapAroundSSH::test_large_file_round_trip': _PAGING_PER_CONNECTION,
    'tests/harness/bubblewrap_sandbox/test_conformance.py::TestRealBubblewrap::test_concurrent_first_use_shares_one_environment': _NO_FRESH_BACKEND,
    'tests/harness/code_mode/test_remote.py::TestCodeModeErrorRouting::test_native_worker_out_of_memory_resets[remote-bridge]': _ISSUE_8824,
    'tests/harness/code_mode/test_remote.py::test_code_mode_runs_remotely': _ISSUE_8824,
    'tests/harness/code_mode/test_remote.py::test_disconnect_mid_snippet_reports_started_calls': _ISSUE_8824,
    'tests/harness/code_mode/test_temporal.py::test_code_mode_runs_remotely_in_temporal_workflow@harness-temporal-code-mode': _ISSUE_8824,
    'tests/harness/daytona_sandbox/test_conformance.py::TestLiveDaytonaSandboxBackend': _SANDBOX_LIVE,
    'tests/harness/daytona_sandbox/test_daytona_live.py': _SANDBOX_LIVE,
    'tests/harness/docker_container/test_conformance.py::TestFakeDocker::test_background_child_does_not_hold_up_completed_command': _NO_DIRECT_EXIT,
    'tests/harness/docker_container/test_conformance.py::TestFakeDocker::test_large_file_round_trip': _PAGING_PER_EXEC,
    'tests/harness/docker_container/test_conformance.py::TestFakeDocker::test_native_realpath_keeps_the_working_dir_and_missing_names': _NO_REALPATH,
    'tests/harness/docker_container/test_docker_live.py': _SANDBOX_LIVE,
    'tests/harness/e2b_sandbox/test_backend.py::test_timeout_stops_a_command_run_with_a_different_locale': _PS_FALLBACK,
    'tests/harness/e2b_sandbox/test_conformance.py::TestFakeE2BSandboxBackend::test_native_realpath_keeps_the_working_dir_and_missing_names': _NO_REALPATH,
    'tests/harness/e2b_sandbox/test_conformance.py::TestLiveE2BSandboxBackend': _SANDBOX_LIVE,
    'tests/harness/e2b_sandbox/test_e2b_live.py': _SANDBOX_LIVE,
    'tests/harness/kubernetes_sandbox/test_conformance.py::TestFakeKubernetesSandboxBackend::test_native_realpath_keeps_the_working_dir_and_missing_names': _NO_REALPATH,
    'tests/harness/kubernetes_sandbox/test_kubernetes_live.py': _SANDBOX_LIVE,
    'tests/harness/media/test_media.py::TestS3MediaStoreLive': _S3_LIVE,
    'tests/harness/modal_sandbox/test_conformance.py::TestFakeModalSandboxBackend::test_native_realpath_keeps_the_working_dir_and_missing_names': _NO_REALPATH,
    'tests/harness/modal_sandbox/test_conformance.py::TestLiveModalSandboxBackend': _SANDBOX_LIVE,
    'tests/harness/modal_sandbox/test_modal_live.py': _SANDBOX_LIVE,
    'tests/harness/opensandbox_sandbox/test_conformance.py::TestFakeOpenSandboxBackend::test_large_file_round_trip': _PAGING_PER_COMMAND,
    'tests/harness/opensandbox_sandbox/test_conformance.py::TestFakeOpenSandboxBackend::test_native_realpath_keeps_the_working_dir_and_missing_names': _NO_REALPATH,
    'tests/harness/opensandbox_sandbox/test_conformance.py::TestLiveOpenSandboxBackend': _SANDBOX_LIVE,
    'tests/harness/opensandbox_sandbox/test_opensandbox_live.py': _SANDBOX_LIVE,
    'tests/harness/pixeltable/test_store.py::TestPixeltableMemoryStoreSchema::test_incompatible_existing_table_rejected[NOTSET]': _PLACEHOLDER_ID,
    'tests/harness/pixeltable/test_toolset.py::TestPixeltableToolsetQuery::test_query_errors_retry_or_propagate[NOTSET-collect]': _PLACEHOLDER_ID,
    'tests/harness/pixeltable/test_toolset.py::TestPixeltableToolsetQuery::test_query_errors_retry_or_propagate[NOTSET-get_metadata]': _PLACEHOLDER_ID,
    'tests/harness/pixeltable/test_toolset.py::TestPixeltableToolsetQuery::test_query_errors_retry_or_propagate[NOTSET-select]': _PLACEHOLDER_ID,
    'tests/harness/pixeltable/test_toolset.py::TestPixeltableToolsetQuery::test_query_errors_retry_or_propagate[NOTSET-where]': _PLACEHOLDER_ID,
    'tests/harness/render_sandbox/test_conformance.py::TestFakeRenderSandboxBackend::test_native_realpath_keeps_the_working_dir_and_missing_names': _NO_REALPATH,
    'tests/harness/render_sandbox/test_conformance.py::TestLiveRenderSandboxBackend': _RENDER_LIVE,
    'tests/harness/smol_sandbox/test_conformance.py::TestFakeSmolSandboxBackend::test_background_child_does_not_hold_up_completed_command': _NO_DIRECT_EXIT,
    'tests/harness/smol_sandbox/test_conformance.py::TestFakeSmolSandboxBackend::test_large_file_round_trip': _PAGING_IN_LIVE_VM,
    'tests/harness/smol_sandbox/test_conformance.py::TestFakeSmolSandboxBackend::test_native_realpath_keeps_the_working_dir_and_missing_names': _NO_REALPATH,
    'tests/harness/smol_sandbox/test_conformance.py::TestLiveSmolSandboxBackend': _VM_LIVE,
    'tests/harness/smol_sandbox/test_smol_sandbox.py::test_live_program_named_like_an_option_runs': _VM_LIVE,
    'tests/harness/smol_sandbox/test_smol_sandbox.py::test_live_vm_passed_in_is_used': _VM_LIVE,
    'tests/harness/smol_sandbox/test_smolvm_cli.py::TestFakeSmolvmCliBackend::test_background_child_does_not_hold_up_completed_command': _NO_DIRECT_EXIT,
    'tests/harness/smol_sandbox/test_smolvm_cli.py::TestFakeSmolvmCliBackend::test_large_file_round_trip': _PAGING_IN_SMOLVM,
    'tests/harness/smol_sandbox/test_smolvm_cli.py::TestFakeSmolvmCliBackend::test_native_realpath_keeps_the_working_dir_and_missing_names': _NO_REALPATH,
    'tests/harness/smol_sandbox/test_smolvm_cli.py::TestLiveSmolvmCliBackend': _VM_LIVE,
    'tests/harness/sprites_sandbox/test_conformance.py::TestFakeSpritesSandboxBackend::test_attaching_to_a_destroyed_environment_raises_unavailable': _NO_DESTRUCTIVE_ATTACH,
    'tests/harness/sprites_sandbox/test_conformance.py::TestFakeSpritesSandboxBackend::test_background_child_does_not_hold_up_completed_command': _NO_DIRECT_EXIT,
    'tests/harness/sprites_sandbox/test_conformance.py::TestFakeSpritesSandboxBackend::test_concurrent_first_use_shares_one_environment': _NO_FRESH_BACKEND,
    'tests/harness/sprites_sandbox/test_conformance.py::TestFakeSpritesSandboxBackend::test_destroying_environment_during_command_raises_unavailable': _NO_DESTRUCTIVE_BACKEND,
    'tests/harness/sprites_sandbox/test_conformance.py::TestFakeSpritesSandboxBackend::test_native_realpath_keeps_the_working_dir_and_missing_names': _NO_REALPATH,
    'tests/harness/sprites_sandbox/test_conformance.py::TestLiveSpritesSandboxBackend': _SANDBOX_LIVE,
    'tests/harness/sprites_sandbox/test_sprites_live.py': _SANDBOX_LIVE,
    'tests/harness/ssh_workspace/test_conformance.py::TestSSHWorkspaceBackend::test_background_child_does_not_hold_up_completed_command': _NO_DIRECT_EXIT,
    'tests/harness/ssh_workspace/test_conformance.py::TestSSHWorkspaceBackend::test_concurrent_first_use_shares_one_environment': _NO_FRESH_BACKEND,
    'tests/harness/ssh_workspace/test_conformance.py::TestSSHWorkspaceBackend::test_large_file_round_trip': _PAGING_PER_CONNECTION,
    'tests/harness/ssh_workspace/test_conformance.py::TestSSHWorkspaceBackend::test_native_realpath_keeps_the_working_dir_and_missing_names': _NO_REALPATH,
    'tests/harness/test_capability_combine.py::test_capability_combine_policy_holds[BrowserUse]': _BROWSER_USE,
    'tests/harness/test_doc_snippets.py': _SIGNATURE_BLOCK,
    'tests/harness/test_examples.py::test_example_builds_agent[optional_dependency_error1-coding_agent]': _INJECTED_EXTRA_ERROR,
    'tests/harness/test_examples.py::test_example_builds_agent[optional_dependency_error1-pixeltable_video]': _INJECTED_EXTRA_ERROR,
    'tests/harness/test_examples.py::test_example_builds_agent[optional_dependency_error1-research_agent]': _INJECTED_EXTRA_ERROR,
    'tests/harness/test_examples.py::test_example_builds_agent[optional_dependency_error2-coding_agent]': _INJECTED_EXTRA_ERROR,
    'tests/harness/test_examples.py::test_example_builds_agent[optional_dependency_error2-pixeltable_video]': _INJECTED_EXTRA_ERROR,
    'tests/harness/test_examples.py::test_example_builds_agent[optional_dependency_error2-research_agent]': _INJECTED_EXTRA_ERROR,
    'tests/harness/test_history_append_only.py::test_conversation_only_appends[BrowserUse-anthropic]': _BROWSER_USE,
    'tests/harness/test_history_append_only.py::test_conversation_only_appends[BrowserUse-openai-chat]': _BROWSER_USE,
    'tests/harness/test_history_append_only.py::test_conversation_only_appends[BrowserUse-openai-responses]': _BROWSER_USE,
    'tests/harness/test_history_append_only.py::test_only_coder_repairs_tool_arguments_by_default[BrowserUse]': _BROWSER_USE,
    'tests/harness/test_skill_examples.py': _SKILL_EXAMPLE,
    'tests/models/test_google.py::test_google_api_errors_are_handled[Exception-error_response0-503]': _PLACEHOLDER_ID,
    'tests/models/test_google.py::test_google_api_errors_are_handled[Exception-error_response1-400]': _PLACEHOLDER_ID,
    'tests/models/test_google.py::test_google_api_errors_are_handled[Exception-error_response2-429]': _PLACEHOLDER_ID,
    'tests/models/test_google.py::test_google_stream_api_errors_are_wrapped[Exception-error_response0-503]': _PLACEHOLDER_ID,
    'tests/models/test_google.py::test_google_stream_api_errors_are_wrapped[Exception-error_response1-429]': _PLACEHOLDER_ID,
    'tests/models/test_multimodal_tool_returns.py::test_mixed_content_ordering[google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_model_sees_multiple_images[google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-binary-audio-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-binary-document-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-binary-image-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-binary-video-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-audio-anthropic]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-audio-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-audio-groq]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-audio-mistral]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-audio-openai_chat]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-audio-openai_responses]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-audio-xai]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-document-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-document-groq]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-document-mistral]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-image-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-image-groq]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-image-mistral]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-video-anthropic]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-video-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-video-groq]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-video-mistral]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-video-openai_chat]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-video-openai_responses]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-uploaded_file-video-xai]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-url-audio-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-url-document-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-url-image-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-url-video-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-url_force_download-audio-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-url_force_download-document-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-url_force_download-image-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[direct-url_force_download-video-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-binary-audio-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-binary-document-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-binary-image-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-binary-video-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-audio-anthropic]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-audio-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-audio-groq]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-audio-mistral]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-audio-openai_chat]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-audio-openai_responses]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-audio-xai]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-document-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-document-groq]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-document-mistral]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-image-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-image-groq]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-image-mistral]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-video-anthropic]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-video-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-video-groq]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-video-mistral]': _NO_UPLOADED_FILE_SUPPORT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-video-openai_chat]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-video-openai_responses]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-uploaded_file-video-xai]': _NO_UPLOADED_FILE,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-url-audio-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-url-document-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-url-image-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-url-video-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-url_force_download-audio-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-url_force_download-document-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-url_force_download-image-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_multimodal_tool_returns.py::test_multimodal_tool_return_matrix[tool_return_content-url_force_download-video-google_vertex]': _VERTEX_PROJECT,
    'tests/models/test_openai.py::test_metadata_non_string_value[NOTSET]': _PLACEHOLDER_ID,
    'tests/models/test_tool_choice_matrix.py::test_tool_choice_matrix[auto-huggingface]': _TOGETHER_CONTINUATION,
    'tests/models/test_typesafe.py::test_evals_classifier': _MISSING_CLASSIFIER,
    'tests/pax/test_screen.py::test_plugin_takes_the_screen_from_inside_a_tool[True]': _ISSUE_8824,
    'tests/providers/test_bedrock.py::test_bedrock_provider_model_profile_all_geo_prefixes[NOTSET]': _PLACEHOLDER_ID,
    'tests/providers/test_gateway_catalog.py': _GATEWAY_LIVE,
    'tests/providers/test_provider_names.py::test_infer_provider[NOTSET]': _PLACEHOLDER_ID,
    'tests/providers/test_provider_names.py::test_infer_provider_class[NOTSET]': _PLACEHOLDER_ID,
    'tests/providers/test_provider_names.py::test_infer_provider_class[gateway/anthropic-AnthropicProvider-PYDANTIC_AI_GATEWAY_API_KEY]': _GATEWAY_INFERENCE,
    'tests/providers/test_provider_names.py::test_infer_provider_class[gateway/chat-OpenAIProvider-PYDANTIC_AI_GATEWAY_API_KEY]': _GATEWAY_INFERENCE,
    'tests/providers/test_provider_names.py::test_infer_provider_class[gateway/converse-BedrockProvider-PYDANTIC_AI_GATEWAY_API_KEY]': _GATEWAY_INFERENCE,
    'tests/providers/test_provider_names.py::test_infer_provider_class[gateway/google-GoogleCloudProvider-PYDANTIC_AI_GATEWAY_API_KEY]': _GATEWAY_INFERENCE,
    'tests/providers/test_provider_names.py::test_infer_provider_class[gateway/groq-GroqProvider-PYDANTIC_AI_GATEWAY_API_KEY]': _GATEWAY_INFERENCE,
    'tests/providers/test_provider_names.py::test_provider_base_url_replaces_the_default[NOTSET]': _PLACEHOLDER_ID,
    'tests/realtime/simulation/test_conformance.py::test_cassette_obeys_the_codec_lifecycle[NOTSET]': _PLACEHOLDER_ID,
    'tests/realtime/simulation/test_simulation.py::test_exploration[NOTSET]': _PLACEHOLDER_ID,
    'tests/realtime/test_parity_ws.py::test_audio_tool_round_parity[azure-current]': _AZURE_REALTIME,
    'tests/realtime/test_parity_ws.py::test_barge_in_conversation_parity[azure-current]': _AZURE_REALTIME,
    'tests/realtime/test_parity_ws.py::test_continuous_microphone_conversation_parity[azure-current]': _AZURE_REALTIME,
    'tests/realtime/test_parity_ws.py::test_instruction_change_parity[azure-current]': _AZURE_REALTIME,
    'tests/realtime/test_parity_ws.py::test_push_to_talk_conversation_parity[azure-current]': _AZURE_REALTIME,
    'tests/realtime/test_parity_ws.py::test_raising_tool_parity[azure-current]': _AZURE_REALTIME,
    'tests/realtime/test_parity_ws.py::test_untranscribed_continuous_microphone_conversation_parity[azure-current]': _AZURE_REALTIME,
    'tests/test_ag_ui.py::test_load_multimodal_url_sources[NOTSET]': _PLACEHOLDER_ID,
    'tests/test_embeddings.py::test_google_task_prefix[NOTSET]': _PLACEHOLDER_ID,
    'tests/test_examples.py': _DOCS_EXAMPLE,
    'tests/test_history_append_only.py::test_conversation_only_appends[ImageGeneration-anthropic]': _NO_NATIVE_TOOL,
    'tests/test_history_append_only.py::test_conversation_only_appends[ImageGeneration-openai-chat]': _NO_NATIVE_TOOL,
    'tests/test_history_append_only.py::test_conversation_only_appends[NativeTool-openai-chat]': _NO_NATIVE_TOOL,
    'tests/test_history_append_only.py::test_conversation_only_appends[WebFetch-openai-chat]': _NO_NATIVE_TOOL,
    'tests/test_history_append_only.py::test_conversation_only_appends[WebFetch-openai-responses]': _NO_NATIVE_TOOL,
    'tests/test_history_append_only.py::test_conversation_only_appends[WebSearch-openai-chat]': _NO_NATIVE_TOOL,
    'tests/test_windows_workspace.py::TestWindowsWorkspaceBackend::test_attaching_to_a_destroyed_environment_raises_unavailable': _NO_DESTRUCTIVE_ATTACH,
    'tests/test_windows_workspace.py::TestWindowsWorkspaceBackend::test_concurrent_first_use_shares_one_environment': _NO_FRESH_BACKEND,
    'tests/test_windows_workspace.py::TestWindowsWorkspaceBackend::test_destroying_environment_during_command_raises_unavailable': _NO_DESTRUCTIVE_BACKEND,
    'tests/test_windows_workspace.py::TestWindowsWorkspaceBackend::test_fifo_read_does_not_wait_for_writer': _NO_FIFOS,
    'tests/test_windows_workspace.py::TestWindowsWorkspaceBackend::test_permission_denied_uses_builtin_error': _WINDOWS_ACLS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_a_backend_attached_by_ref_reaches_a_detached_process': _NO_ATTACH_BACKEND,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_a_backend_attached_by_ref_reaches_the_same_environment': _NO_ATTACH_BACKEND,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_a_missing_program_exits_127': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_argv_items_are_literal': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_attaching_to_a_destroyed_environment_raises_unavailable': _NO_DESTRUCTIVE_ATTACH,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_background_child_does_not_hold_up_completed_command': _NO_DIRECT_EXIT,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_cancellation_stops_foreground_work': _NO_FOREGROUND_PROCESS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_command_form_must_match_shell': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_command_output_is_complete': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_commands_and_files_share_one_environment': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_concurrent_first_use_shares_one_environment': _NO_FRESH_BACKEND,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_destroying_environment_during_command_raises_unavailable': _NO_DESTRUCTIVE_BACKEND,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_env_is_added': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_fifo_read_does_not_wait_for_writer': _NO_FIFOS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_file_as_parent_raises_not_a_directory': _NO_PATH_TRAVERSAL,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_native_realpath_keeps_the_working_dir_and_missing_names': _NO_REALPATH,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_permission_denied_uses_builtin_error': _NO_PERMISSIONS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_realpath_and_entries_follow_symlinks': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_result_reports_exit_code_stdout_and_stderr': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_run_starts_in_cwd': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_stdin_is_at_eof': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_symlink_loop_does_not_break_listing': _NO_SYMLINKS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_timeout_raises_workspace_timeout_error': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_undecodable_command_bytes_are_replaced': _NO_COMMAND_BYTE_STREAM,
    'tests/test_workspace_conformance.py::TestFilesystemOnlyWorkspaceBackend::test_writes_go_through_a_symlink': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestLocalWorkspaceBackend::test_concurrent_first_use_shares_one_environment': _NO_FRESH_BACKEND,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_a_backend_attached_by_ref_reaches_a_detached_process': _NO_POSIX_PROCESSES,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_a_missing_program_exits_127': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_argv_items_are_literal': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_background_child_does_not_hold_up_completed_command': _NO_DIRECT_EXIT,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_cancellation_stops_foreground_work': _NO_FOREGROUND_PROCESS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_command_form_must_match_shell': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_command_output_is_complete': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_commands_and_files_share_one_environment': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_destroying_environment_during_command_raises_unavailable': _NO_COMMAND_TO_INTERRUPT,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_env_is_added': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_fifo_read_does_not_wait_for_writer': _NO_FIFOS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_file_as_parent_raises_not_a_directory': _NO_PATH_TRAVERSAL,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_native_realpath_keeps_the_working_dir_and_missing_names': _NO_REALPATH,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_permission_denied_uses_builtin_error': _NO_PERMISSIONS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_realpath_and_entries_follow_symlinks': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_result_reports_exit_code_stdout_and_stderr': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_run_starts_in_cwd': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_stdin_is_at_eof': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_symlink_loop_does_not_break_listing': _NO_SYMLINKS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_timeout_raises_workspace_timeout_error': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_undecodable_command_bytes_are_replaced': _NO_COMMAND_BYTE_STREAM,
    'tests/test_workspace_conformance.py::TestProviderBackend::test_writes_go_through_a_symlink': _NO_COMMANDS,
    'tests/test_workspace_conformance.py::TestRunOnlyWorkspaceBackend::test_concurrent_first_use_shares_one_environment': _NO_FRESH_BACKEND,
    'tests/test_workspace_conformance.py::TestRunOnlyWorkspaceBackend::test_native_realpath_keeps_the_working_dir_and_missing_names': _NO_REALPATH,
    'tests/ui_roundtrip/': _UI_ROUNDTRIP,
}
"""Each test, test file, or directory (ending in `/`) whose tests every CI cell skips, with the reason none runs them."""

PATH_GATED_JOBS: dict[str, tuple[str, ...]] = {
    'harness-bubblewrap-integration': ('tests/harness/bubblewrap_sandbox/',),
    'test-clai2-clipboard': ('tests/pax/test_image_input.py', 'tests/pax/test_image_clipboard_native.py'),
}
"""The tests that only a job `ci.yml` runs for some changes runs, by job: exempt on a run that skipped that job."""
