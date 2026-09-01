# Handoffs index

Append-only pointer log. Newest at bottom. Full bodies live under `handoffs/` — never overwrite a handoff file.

Several agents/managers share this file. Each entry belongs to the **lane** that wrote it. Read your own lane's latest entry via `latest-handoff.sh`; never pick one by eye.

## Entry format

```
## YYYY-MM-DDTHHMMZ · handoffs/<filename>.md · [<writer> · lane-id:<id> · lane:<label>] <one-line summary>
```

---

<!-- entries below, newest at bottom -->
