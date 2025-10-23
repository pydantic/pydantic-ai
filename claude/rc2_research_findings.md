# ThreadProtocol RC2 Research Findings

## 1. Multi-Modal Content Types in Pydantic AI

### Type Definitions (messages.py)

```python
# Core type aliases
UserContent: TypeAlias = str | MultiModalContent
MultiModalContent = ImageUrl | AudioUrl | DocumentUrl | VideoUrl | BinaryContent
```

### URL-Based Content Classes

All URL classes inherit from `FileUrl` base class with these common features:
- `url: str` - The URL of the file
- `identifier: str` - Unique ID for referencing in tool arguments
- `force_download: bool` - Whether to download vs pass URL directly
- `vendor_metadata: dict[str, Any] | None` - Provider-specific metadata
- `media_type: str` - Inferred or explicit MIME type

**Specific URL Classes:**
```python
ImageUrl(url, kind='image-url')       # JPEG, PNG, GIF, WebP
AudioUrl(url, kind='audio-url')       # MP3, WAV, FLAC, OGG, AIFF, AAC
VideoUrl(url, kind='video-url')       # MP4, WebM, MOV, etc. + YouTube
DocumentUrl(url, kind='document-url') # PDF, TXT, CSV, DOCX, XLSX, HTML, MD
```

### BinaryContent Class

For inline binary data:
```python
BinaryContent(
    data: bytes,                    # The binary data
    media_type: str,                # MIME type
    identifier: str | None = None,  # Unique ID
    vendor_metadata: dict | None = None
)
```

Special features:
- `data_uri` property for conversion to/from data URIs
- `BinaryImage` subclass for image-specific content
- Automatic format detection based on media_type

### Implications for ThreadProtocol

**Option A: Reference PAI Types Directly**
```typescript
// In ThreadProtocol
type UserContent = string | ImageUrl | AudioUrl | DocumentUrl | VideoUrl | BinaryContent;
// Where these types are defined exactly as PAI defines them
```

**Option B: Simplify for Storage**
```typescript
interface MediaContent {
  type: "image" | "audio" | "video" | "document";
  url?: string;         // For URL-based
  data?: string;        // Base64 for binary
  media_type: string;
  identifier: string;   // For tool references
  metadata?: any;       // Vendor-specific
}

type UserContent = string | MediaContent;
```

## 2. Deferred Tool Handling in Pydantic AI

### How Deferred Tools Work

1. **Tool Execution Modes:**
   - **Immediate**: Normal execution within agent.run()
   - **Requires Approval**: Needs user consent before execution
   - **External/Deferred**: Executed outside the current run

2. **When Tools Are Deferred:**
   ```python
   @agent.tool(requires_approval=True)  # Always needs approval
   def dangerous_tool(): ...

   @agent.tool
   def conditional_tool(ctx: RunContext):
       if needs_approval:
           raise ApprovalRequired  # Conditionally needs approval
       if external_execution:
           raise CallDeferred      # Execute externally
   ```

3. **Agent Run Output with Deferred Tools:**
   ```python
   result = agent.run(...)
   if isinstance(result.output, DeferredToolRequests):
       # Contains:
       # - approvals: List[ToolCallPart] for tools needing approval
       # - calls: List[ToolCallPart] for external execution
   ```

### What Gets Stored in Message History

When tools are deferred, the message history contains:

1. **Initial ModelResponse with ToolCallParts**:
   ```python
   ModelResponse(parts=[
       ToolCallPart(tool_name='delete_file', args={...}, tool_call_id='xyz'),
       ToolCallPart(tool_name='update_file', args={...}, tool_call_id='abc')
   ])
   ```

2. **After Resolution, ModelRequest with ToolReturnParts**:
   ```python
   ModelRequest(parts=[
       ToolReturnPart(
           tool_name='delete_file',
           content='Deleting files is not allowed',  # Denied
           tool_call_id='xyz'
       ),
       ToolReturnPart(
           tool_name='update_file',
           content="File updated",  # Approved & executed
           tool_call_id='abc'
       )
   ])
   ```

### Key Insight: Deferred State is Temporary

- **During deferral**: Message history ends with unresolved ToolCallParts
- **After resolution**: New ModelRequest added with ToolReturnParts
- **No special "deferred" marker** in the final message history
- The "deferred" state only exists between agent runs

## 3. Pending User Actions State Machine

### The Flow

```
1. Model calls tool(s)
   ↓
2. Tool execution determines: immediate/deferred/approval needed
   ↓
3a. Immediate: Execute and continue         → ToolReturnPart added immediately
3b. Deferred: End run with DeferredToolRequests → Thread in "waiting" state
   ↓
4. External resolution (user/system)
   ↓
5. New run with DeferredToolResults         → ToolReturnPart added, continues
```

### What ThreadProtocol Needs to Track

**Option 1: Implicit State (Recommended)**
- If last message has ToolCallParts without corresponding ToolReturnParts → pending
- No explicit "pending" field needed
- State derived from message structure

**Option 2: Explicit State Field**
```typescript
interface AgentTurn {
  // ... existing fields ...
  pending_tool_calls?: {
    tool_call_id: string;
    tool_name: string;
    status: "awaiting_approval" | "external_execution";
    requested_at: string;
  }[];
}
```

### Resolution Tracking

When tools are resolved:
```typescript
interface ToolReturnPart {
  // ... existing fields ...
  resolution_metadata?: {
    deferred_type: "approval" | "external";
    resolved_at: string;
    resolved_by?: string;  // user_id or system
    approval_status?: "approved" | "denied";
  };
}
```

## Recommendations for ThreadProtocol RC2

### 1. Multi-Modal Content

**Recommendation**: Use simplified storage format but maintain PAI compatibility

```typescript
interface MediaContent {
  content_type: "url" | "binary";
  media_type: string;      // MIME type
  media_category: "image" | "audio" | "video" | "document";

  // For URLs
  url?: string;

  // For binary
  data?: string;           // Base64 encoded

  // Common
  identifier: string;      // For tool references
  metadata?: any;          // Provider-specific
}
```

This allows:
- Easy serialization to JSON
- Conversion to/from PAI types
- Clear distinction between URL and binary content

### 2. Deferred Tools

**Recommendation**: Don't add special handling - PAI already handles this well

The message history already captures the full state:
- ToolCallParts without returns = pending
- ToolReturnParts = resolved
- Content in ToolReturnPart tells you if approved/denied

ThreadProtocol just needs to faithfully record what's in the messages.

### 3. Pending State Tracking

**Recommendation**: Use implicit state detection

```typescript
// Helper function to detect pending state
function hasPendingToolCalls(turn: AgentTurn): boolean {
  const toolCalls = new Set<string>();
  const toolReturns = new Set<string>();

  for (const msg of turn.messages) {
    for (const part of msg.parts) {
      if (part.part_kind === 'tool-call') {
        toolCalls.add(part.tool_call_id);
      } else if (part.part_kind === 'tool-return') {
        toolReturns.add(part.tool_call_id);
      }
    }
  }

  // If there are calls without returns, they're pending
  return [...toolCalls].some(id => !toolReturns.has(id));
}
```

### 4. Optional Enhancement: Resolution Metadata

If you want to track HOW things were resolved:

```typescript
interface SystemMessage {
  message_type: "system";
  event_type: "tool.resolved";
  event_data: {
    tool_call_id: string;
    resolution: "approved" | "denied" | "executed_externally";
    resolved_by: string;  // user_id or "system"
    resolved_at: string;
    denial_reason?: string;
  };
}
```

This keeps the ToolReturnPart clean (matching PAI) while adding tracking via SystemMessage.

## Summary

1. **Multi-modal content**: PAI has comprehensive support. ThreadProtocol should use a simplified but compatible format.

2. **Deferred tools**: PAI handles this elegantly through the message structure itself. No special fields needed in ThreadProtocol - just record messages faithfully.

3. **Pending states**: Can be derived from the message structure (ToolCallParts without ToolReturnParts = pending). Optional SystemMessages can track resolution details.

The key insight: **PAI already solved the hard parts**. ThreadProtocol just needs to:
- Store messages faithfully
- Add multi-agent attribution
- Track resolution metadata via SystemMessages if desired
- Let the message structure itself encode the state