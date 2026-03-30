---
name: Kill stale servers directly
description: When ports are blocked by a previous start.sh run, kill the processes directly instead of asking the user
type: feedback
---

When start.sh or dev servers are still running and blocking ports, kill them directly rather than asking the user to do it manually.

**Why:** User stopped the conversation to tell me to handle it — they don't want to be asked about this.
**How to apply:** Before starting servers or when ports are in use, run `lsof -i :<port> -t | xargs kill` to clear them.
