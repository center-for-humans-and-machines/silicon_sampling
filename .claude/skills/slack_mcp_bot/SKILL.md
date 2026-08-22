---
name: slack_mcp_bot
description: Use the Slack MCP server to post a question to the coordination channel, then BLOCK on `wait_for_slack_reply.sh` until the human replies in-thread. Use whenever you need a human-in-the-loop answer over Slack instead of polling `slack_get_thread_replies` from the agent loop (which burns tokens).
---

# Slack ask-and-wait

The repo wires Slack two ways:
- **MCP server** (`slack`, from `@modelcontextprotocol/server-slack`) — for posting, reactions, and ad-hoc reads.
- **`wait_for_slack_reply.sh`** — a single blocking shell command for the wait. On `PATH` in the project container, so just invoke it as `wait_for_slack_reply.sh`.

`SLACK_BOT_TOKEN` and (typically) `SLACK_CHANNEL_IDS` are already in the container env, so you can read `$SLACK_CHANNEL_IDS` instead of asking the user for the channel id.

## Why a separate wait script

Polling `slack_get_thread_replies` in a loop costs **one full LLM turn per poll** (current context resent each time). Over a multi-hour wait that is large input-token spend. `wait_for_slack_reply.sh` does the polling internally in shell, so the model is not re-invoked between polls — only once at call time and once on return.

Use this instead of looping `slack_get_thread_replies` whenever you don't expect an instant answer.

## Ask-and-wait workflow

1. **`slack_post_message`** with the full question to `$SLACK_CHANNEL_IDS` (use the first id if the env var is comma-separated). Save the returned `ts`; that's your `thread_ts`.
2. Optional: tell the user once which thread to reply in.
3. **`wait_for_slack_reply.sh "$CHANNEL_ID" "$THREAD_TS" [DEADLINE_SECONDS]`** via Bash. Blocks until the first non-bot reply lands or the deadline elapses.
4. On exit 0, parse the printed `text:` block as the human's answer; optionally **`slack_add_reaction`** on the printed `ts` to acknowledge.
5. Resume the task with the answer.

If the script exits 1 (deadline), decide between: (a) escalating with a new `slack_post_message` and a longer `wait_for_slack_reply.sh`, or (b) ending the turn cleanly so the human can resume the session in Cursor when convenient.

## Usage

```bash
wait_for_slack_reply.sh <CHANNEL_ID> <THREAD_TS> [DEADLINE_SECONDS]
```

- `CHANNEL_ID` — `C…` (public), `G…` (private), `D…` (DM). Get from `$SLACK_CHANNEL_IDS` or `slack_list_channels`.
- `THREAD_TS` — Slack ts of the thread root, returned by `slack_post_message`. Format `<seconds>.<microseconds>`.
- `DEADLINE_SECONDS` — optional positive integer. Default `3600` (1h). Hard upper bound is the bash tool's `BASH_MAX_TIMEOUT_MS` (6h in the project compose); pick something less. Pick longer (e.g. `21600`) only when you genuinely expect to wait that long, and prefer the "end the turn cleanly" pattern for overnight cases.

The script prints timestamped progress to stdout while it sleeps, then exactly one of:

- **Exit 0** — human replied. Prints:
  ```
  wait_for_slack_reply: human replied
  user: U…
  ts: <slack ts>
  text:
  <message body>
  ```
- **Exit 1** — deadline elapsed without a non-bot reply.
- **Exit 2** — bad arguments / missing `SLACK_BOT_TOKEN`.
- **Exit 3** — `curl` or `jq` not on PATH.
- **Exit 4** — Slack API returned a non-recoverable error (e.g. `invalid_auth`, `channel_not_found`, `missing_scope`). Treat as a configuration bug, not a wait failure.

`ratelimited` from Slack is recoverable and triggers a backoff; it does not return.

## Tunables

Mostly relevant for tests; don't override in normal use.

- `WAIT_FOR_SLACK_REPLY_SLEEP` — seconds between Slack polls (default `30`).

## MCP tools (reference)

`slack_list_channels`, `slack_post_message`, `slack_reply_to_thread`, `slack_add_reaction`, `slack_get_channel_history`, `slack_get_thread_replies`, `slack_get_users`, `slack_get_user_profile`. Prefer `slack_post_message` + `wait_for_slack_reply.sh` for the ask-and-wait flow above; reach for the others only when you need richer interaction (multi-turn back-and-forth, reading older context, profile lookups).
