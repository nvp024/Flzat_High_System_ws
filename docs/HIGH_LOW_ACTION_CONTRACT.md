# FLZAT High–Low Action contract status

Updated: **4 November 2026**.

This document separates rules already implemented in High, behavior verified
with the Low mock and details that still require agreement with the production
Low team.

## Agreed High behavior

- High produces compact `flzat.high_handoff.v2` and commits it atomically before Action
  A is sent.
- Action A carries an absolute shared-filesystem path, not the JSON body.
- The JSON preserves the original request and clarification answers, goal
  description, constraints, missing information, selected entities, relationships
  and trusted position/frame when available. Shared context identifies the map,
  environment and memory source; stored positions are not marked current.
- High never accepts coordinates authored by the VLM; positions are resolved
  from the request snapshot by `context_ref`.
- `STARTED_FAST` and `STARTED_PLAN` mean execution started, not completed.
- Action B `ASK` and `CONFIRM` speak and listen. `INFORM` and `REFUSE` speak and
  complete without listening.
- One B interaction may exist at a time. A rejected B is not queued.
- A natural B answer is returned to that B request and never becomes a new A.
- Stop/cancel is admitted even while ordinary input is rejected as BUSY.

## Compact handoff v2 (4 November 2026)

Top-level fields: `schema_version`, `handoff_id`, `user_request`, `goal`,
`context`, `entities`. `user_request` contains `original` and `clarifications`
(question/answer pairs). `goal` keeps `description`, `constraints` and
`missing_information`; no change to the Qwen `CognitiveDecision.v1` schema.

`context` contains `environment_id`, `map_id`, `source: environment_memory`,
and `current_positions_confirmed: false`. Entities contain `id`, `label`, optional
nonempty `description`, `attributes`, `relationships`, `scene`, and optional
`position` with `frame_id/x/y/z`. Relation strings are preserved as stored;
missing positions are omitted, not filled with zeros or another entity's pose.

No top-level `context_refs`, `status`, `created_at`, `resolved` request, per-entity
scores/quality/timestamps, or repeated per-entity source metadata are sent.
Full snapshot provenance is kept in High's per-request `handoff_audit.json`.
The compact JSONL journal remains identical to the payload sent by path.

Goal A keeps run/task/handoff IDs. The file contains only the handoff ID and must
use `<shared_root>/<run_id>/<task_id>/<handoff_id>.json`; Low mock checks the
resolved path suffix and payload identity. Moving the file to another layout
requires a contract change, not just changing the reader silently. Existing v1
files are not rewritten, and the production Low reader must adopt v2 before
integration. These are High/mock rules pending agreement with production Low.

## Mock-tested Action A mapping

| Business result | ROS Action terminal status | High state |
|---|---|---|
| `STARTED_FAST` | SUCCEEDED | `LOW_RUNNING` |
| `STARTED_PLAN` | SUCCEEDED | `LOW_RUNNING` |
| `REFUSED` | SUCCEEDED | `REFUSED` |
| `FAILED` | ABORTED | `FAILED` |
| `CANCELED` | CANCELED | `CANCELLED` |

Goal rejection and transport failure remain communication failures; High does
not invent a Low business outcome.

## Mock-tested Action B mapping

| Situation | ROS Action terminal status | Result outcome |
|---|---|---|
| Understood answer, including a negative CONFIRM | SUCCEEDED | `ANSWERED` |
| User explicitly declines to answer | SUCCEEDED | `DECLINED` |
| Accepted goal reaches its deadline | SUCCEEDED | `TIMEOUT` |
| INFORM/REFUSE speech completes | SUCCEEDED | `ANSWERED`, empty answer |
| Low cancels accepted B | CANCELED | empty |
| Technical TTS/interpretation failure | ABORTED | empty + error |
| Retry limit reached before deadline | ABORTED | empty + `UNRESOLVED_ANSWER` |

High publishes observable BUSY events on `/flzat/high_system/status`. The mock
retries a B only after observing a matching `action_b_rejected` BUSY event and
uses a finite retry count. Production Low must not assume every B is accepted.

## Items requiring production-Low agreement

Review note (4 November 2026): the current sequential test flow assumes High
finishes its own clarification before sending A, and Low sends B only after
receiving that A. No earlier Low task is expected to send a late B while High
clarifies a new request. High does not yet explicitly reject B based on a
pending HIGH-owned clarification. At the user's direction, this cross-task
overlap risk is recorded for later correction if encountered, not fixed in
the current iteration. `STARTED_*` still does not mean execution completed;
the assumption above is operational, not an enforced lifecycle guarantee.

1. Confirm the exact `.action` field names and ROS lifecycle/business-outcome
   mapping above.
2. Decide whether `UNRESOLVED_ANSWER` should become an explicit B outcome.
3. Define monitoring, completion, failure and cancel/change control after A has
   returned `STARTED_*`.
4. Define artifact retention acknowledgement so High knows when a JSON path may
   be removed.
5. Confirm shared-path mounting rules when High and Low run in different
   containers or machines.
6. Define a separate perception request if Low needs fresh scene analysis;
   Action B remains a user-interaction contract and must not be overloaded.
7. Decide whether `SUPERSEDED` remains in B or is reserved for a later task
   replacement phase.

Until these points are agreed, the workspace demonstrates High and the Low
mock only; it does not claim production Low interoperability.
