# Workspace mutation security

The mutation layer is designed to prevent model mistakes and ordinary concurrent edits from
escaping or corrupting the selected workspace. It is not a sandbox against a malicious local
administrator or a hostile process that can continuously rewrite the directory tree.

## Invariants

- Every target is workspace-relative. Absolute, drive-qualified, parent-traversal, device,
  alternate-data-stream, control-character, ignored, internal, and sensitive paths fail closed.
- Parent directories must already exist. `create_directory` creates exactly one explicitly named
  child at a time; it never performs recursive parent creation. File tools retain their original
  behavior and cannot create directory trees or request a policy bypass.
- `.gitignore` files are readable policy inputs but are not mutation targets, so a change cannot
  rewrite the policy that its own undo must traverse.
- Symlinks, junctions/reparse points, multiply linked files, and Windows files carrying named
  data streams are not mutation targets.
- Directory creation uses no-clobber `mkdir`; an existing ordinary directory is an idempotent
  success, while an existing file or link fails closed.
- Existing files require the raw-byte SHA-256 returned by `read_file`; `null` means the target
  must not exist. Digest, mode, and file identity are checked again immediately before commit.
- Writes use an exclusive same-directory temporary file, flush it, then replace the directory
  entry atomically. New-file creation is no-clobber, so a concurrent creator wins safely.
- Only UTF-8/UTF-8-BOM text is editable. Exact bytes determine hashes; BOM, uniform CRLF/LF, and
  final-newline intent are preserved unless a full write explicitly selects a newline style.
- Successful changes enter a bounded in-memory journal. Undo is LIFO and only succeeds while the
  current bytes and file identity still match that change's postimage.
- Diff display is bounded and control characters are escaped. Truncating a preview never truncates
  the bytes written to disk.

## Platform boundary

POSIX file commits and directory creation use root-anchored directory descriptors and no-follow
operations where the Python runtime exposes them. Windows mutation remains path-based: existing
files use `ReplaceFileW`
with a reserved same-directory backup to carry the original DACL and filesystem attributes forward,
while new files use no-clobber `os.rename`. Documented partial `ReplaceFileW` failures either restore
the backup without clobbering a new target or preserve it under the reserved name and return
`write_recovery_required`; cleanup never guesses which ambiguous copy is authoritative. The
implementation revalidates policy, digest, and parent identity immediately before that namespace
operation, but a hostile local process could still swap a parent for a junction in the final syscall
window. Closing that window requires a native handle-relative Windows rename implementation and is
outside the current threat model.

File flush plus atomic rename also does not claim strict survival across every power-loss and
storage-controller scenario. A detected post-commit durability uncertainty is reported as metadata
instead of misreporting an already-applied change as a normal failure.

## Command boundary

- Commands are launched as an argument vector with `shell=False`, closed stdin, a bounded wall-clock
  timeout, combined bounded output, and credential-like environment variables removed. Commands
  that may issue verification evidence receive a smaller host-owned environment allowlist, so
  ambient runner options such as `PYTEST_ADDOPTS` cannot silently change an exact argv. Shells,
  batch files, and a small hard-deny set of destructive programs are rejected.
- `safe` is the default policy: every ordinary external command requires approval before it can
  start. The M4 CLI stops an active Rich display, shows the exact argv and relative cwd as literal
  text, and defaults to denial. The safe-mode prompt is offered only when both stdin and stdout are
  terminals, so non-interactive input cannot silently approve. `auto` is an explicit opt-in that
  skips the prompt. Every ordinary `auto` command invalidates prior verification because its effects
  are unknown.
- Verification is capability-first and also constrained to recognizable test, build, and check entry
  points. Only a host-registered exact executable path, argv, workspace-relative cwd, evidence kind,
  and label can issue a verification signal. Generic interpreter code, scripts, unknown launchers,
  unbound workspace-owned executables, help, version, collection-only, fixture-listing, and no-run
  variants are rejected as evidence even if a host accidentally registers them. A verifier
  executable inside the workspace is accepted only when the host registers its exact SHA-256 and
  the digest still matches immediately before classification. Output text is never parsed to decide
  success.
- Project trust is opt-in through exactly `<root>/.coding-agent/project.toml`; parent directories are
  never searched. The strict schema exposes only pytest and a fixed no-argument Python-module smoke
  form, not shell text or arbitrary argv. Without an explicitly declared interpreter, passing checks
  remain `checks_only` and cannot validate the target runtime.
- At run start the host binds the policy bytes, selected interpreter hash, and a bounded manifest of
  protected paths. Model-facing mutation tools reject protected files while reads remain available;
  commands are still treated as potentially hostile, so every configured verifier rechecks integrity
  both before and after its process. Drift rejects the evidence even if the process exits zero.
- A content-free policy fingerprint is embedded in the host system prompt checkpoint. Configured
  projects reject missing, malformed, duplicate, or changed anchors on resume; old verification
  evidence is never persisted or restored.
- The runner owns the command lifetime and attempts to terminate descendants on every exit path.
  A root process that exits while descendants remain after a short launcher-settle window is
  incomplete, even when its exit code is zero. If membership, containment, or cleanup cannot be
  confirmed, the runner emits a terminal control fact and the whole Agent run stops instead of
  accepting verification.

## Local Web boundary

- The optional Web host binds only to `127.0.0.1`; there is no configurable remote bind. A
  token-protected request may ask the local host to open the native Windows folder chooser, and the
  browser may also propose an absolute directory during explicit project registration. A picker
  result is convenience, not authority: the server still resolves, validates, fingerprints, and then
  owns the active project together with the model, base URL, permission mode, state root, verifier
  registrations, and run limits. A normal run request contains no path, and the captured project
  context cannot change during that run.
- `TrustedHostMiddleware` accepts only `127.0.0.1` and `localhost`. Responses add a self-only
  Content Security Policy plus `no-referrer`, `nosniff`, frame-deny, and `no-store` headers. These
  are browser hardening measures, not user authentication.
- Workbench mutations require a random per-process control token learned through same-origin
  metadata; requests with an `Origin` header must also match the loopback origin. Project selection
  uses an empty JSON body and task/project creation use JSON, preventing simple cross-origin forms.
  Read-only endpoints remain local and are still not a hostile-same-machine authentication boundary.
- The native picker bridge uses the same mutation token and Origin checks, allows only one dialog at
  a time, and returns stable cancelled/busy/unavailable outcomes without exception detail. It never
  runs in the deterministic demo. While the dialog is open, one server-side navigation reservation
  rejects run admission, project registration/selection, and a second picker request; the reservation
  is released on selection, cancellation, and failure.
- Project/run navigation records are bounded, versioned JSON in the private state directory. Stores
  reject link-like paths and non-regular/hard-linked records, use atomic replacement, and cannot sit
  inside a registered workspace or contain it. New-project creation makes only one absent empty leaf,
  never overwrites an existing directory, and rolls back that leaf if registration fails. Each run
  record freezes the selected directory fingerprint; a replaced directory cannot inherit or replay
  runs from the old physical identity.
- Project history is an untrusted navigation aid, not execution authority. The API rebuilds a
  bounded `DashboardProjection` from validated trace events. A single completed-run detail may add
  only the bounded final Assistant reply from a validated, workspace-bound terminal checkpoint; it
  never exposes the rest of the checkpoint conversation. Missing, corrupt, nonterminal, or
  workspace-mismatched checkpoints degrade to trace-only replay. Raw event messages, other canonical
  messages, read/search/command output, and provider state remain omitted. The only tool-output
  exception is the mutation tool's explicit presentation-safe Diff: timeline entries keep six lines,
  while only `latest_change` may expose at most 80 ordered lines plus an honest completeness flag.
  Runs created before M5.5 metadata are not automatically listed.
- API credentials remain in the backend environment. A process entry point may populate that
  environment from the exact launch-directory `.env.local`; it never searches parents, loads only
  the four allowlisted model settings, disables interpolation, and preserves existing environment
  values. The file is ignored by Git but remains plaintext local storage. Credentials are never sent to browser state, static
  assets, or storage. The browser receives only the documented snapshot whitelist and terminal
  result, not raw events, canonical messages, raw provider responses, arbitrary tool output, provider
  state, or runtime config.
- Worker exceptions and failed-result details are mapped to stable public messages before entering
  browser state; raw exception text and `AgentResult.error` are not exposed by the Web API.
- Web `safe` mode is deliberately fail-closed: there is no browser approval broker, so an ordinary
  command that requires approval is denied. Exact host-registered verification commands retain
  their capability-based path. `auto` must be selected explicitly and still is not a sandbox.
- Graceful shutdown closes run admission and gives the worker a five-second drain window. If it is
  still active, the host requests cooperative cancellation and waits for a short bounded cleanup
  window. The Agent observes that token before model requests, after a blocking model response, and
  between tool calls; it saves a ready checkpoint rather than accepting late work. The host never
  force-kills a model or command call already in progress, and a forced kill, crash, or power loss can
  still interrupt persistence. The Web host does not claim crash-safe execution.

Loopback binding and request headers do not protect against an already-hostile process or user on
the same machine. Do not port-forward, reverse-proxy, or otherwise expose this local session view.

## Live evaluation boundary

- Evaluation is network-disabled by default and has no API-key argument. A live run requires both
  `--live` and `--allow-paid-api`, an explicit/environment model, and `OPENAI_API_KEY`. The endpoint
  is validated before any fixture or model client is created; remote plain HTTP, credentials, query,
  and fragment components fail as CLI configuration errors.
- The CLI reports the endpoint host and a worst-case request-attempt ceiling before starting.
  Evaluation fixes the Agent retry budget at one retry per step and stops a multi-case suite after
  the first runner/provider/harness `error`.
- Each case receives a fresh temporary repository with a known-failing public test. Public tests and
  pytest control files are hashed before the Agent runs; changed, deleted, or newly added controls
  fail integrity checks. Required source files must exist and required edits must change bytes. A
  separate whole-repository manifest compares presence, type, and bytes before and after the Agent.
  It has no filename-based cache exceptions: any path not declared in the scenario's change
  allowlist fails the case even when the regression oracle is green. The registered pytest verifier
  disables bytecode writes and pytest's cache provider so trusted verification remains observational.
- Final judgement runs in a sibling directory that is not an Agent workspace. Only scenario-owned
  source paths are copied there; the regression-oracle tests are materialized afterward. The verifier
  uses the credential-stripped verifier environment, imports pytest before adding the candidate
  source root, disables third-party plugin autoload/cache, and requires a per-run randomized marker,
  a non-empty collected suite, and a passing call report for every collected test. An early zero exit
  or an all-skipped suite is not success.
- Success additionally requires the Agent's own state to be `COMPLETED`; an independently green
  oracle cannot upgrade `COMPLETED_UNVERIFIED`. Reports expose only a versioned metric whitelist and
  distinguish `passed`, task/oracle `failed`, and runtime `error`. They omit commands, raw pytest
  output, provider details, base URLs, credentials, and run IDs; report files use exclusive creation.

This separation protects the score from ordinary public-test softening and several process-exit
tricks. It is neither a hostile-code sandbox nor a secret benchmark: built-in scenario definitions,
including oracle assertions, ship in the installed Python package and same-environment candidate code
could inspect them. Candidate source also executes locally. A genuinely private benchmark or an
untrusted model/repository therefore requires an external service, VM, or container boundary.

### Not an operating-system sandbox

A workspace-relative `cwd` only selects the starting directory. It does not restrict filesystem,
network, registry, device, or process access. Test commands execute repository code, and external
programs may invoke their own helpers. Run untrusted repositories inside a container, VM, or
low-privilege account with separate credentials; neither `safe` nor `auto` claims malicious-code
isolation. Do not place secrets in argv: environment filtering cannot remove secrets already present
in command arguments or program output.

Windows commands are attached while suspended to a kill-on-close Job Object. POSIX currently uses a
fresh session/process group, which is a lifecycle boundary rather than a hostile-code sandbox: a
malicious descendant can deliberately create a new session and escape that group. Use an external
cgroup/container boundary for adversarial repositories; M4 does not claim to provide one itself.

## Verification boundary

`ToolControlFacts` are created by project-owned runtime code and excluded from model-facing tool
messages. A successful mutation or an ordinary `auto` command advances the verification epoch, so
older evidence becomes stale. A failed or timed-out registered verifier records failed evidence; it
does not become a transport error and cannot be replaced by a model's textual claim. Final responses
therefore end as either `COMPLETED` with current passed evidence or `COMPLETED_UNVERIFIED` with
missing, failed, or stale evidence.

This gate proves only that the registered command exited successfully after the latest known external
mutation. It does not prove task correctness, test adequacy, absence of malicious test code, or that
the model did not weaken tests before running them.

## Checkpoint boundary

Checkpoints are passive, versioned, bounded JSON snapshots saved only between complete model/tool
turns or at a terminal response. Loading never replays a tool call and never restores verification
evidence; a resumed run must verify again. The store rejects known credential patterns and private
reasoning fields, but the canonical transcript can still contain source and command output. Its state
directory must therefore be private and outside the workspace.

Schema-v3 checkpoints carry an opaque digest of the resolved workspace path and filesystem identity,
plus bounded RunMemory; schema-v2 payloads migrate conservatively on load.
The CLI requires an explicit repository for resume and rejects a mismatch before constructing a
model client. It also refuses a ready checkpoint when the latest trace says the run already
completed. A new run preallocates its ID and holds a non-blocking OS file lease before constructing
the runtime; resume takes that same per-run lease. The original process and a same-ID resume therefore
cannot execute concurrently.

Resume restores the canonical transcript and bounded RunMemory, including the structured plan,
change summaries, and selected failure facts. The mutation undo journal, interactive approval
decisions, provider-private state, and fresh verification evidence are not restored across processes.
Resume is not an exactly-once transaction: if a process dies after a tool changes disk but before the
ready checkpoint is replaced, the workspace may be newer than the transcript. Revision checks and
fresh observation reduce that risk but cannot erase it.

## Trace and presentation boundary

JSONL traces contain only validated `RunEvent` records, not provider response objects, full prompts,
hidden reasoning, raw read/search results, or raw command output. They can contain bounded plan and
mutation-diff previews because those are explicit presentation artifacts. Trace files are size
bounded and reject incomplete records, invalid UTF-8/JSON, symlinks, and multiply linked files. They
live in the private per-user state directory by default and must not be committed.

`inspect` strictly validates a trace and renders whitelisted fields. A trace is an audit aid, not a
tamper-proof ledger: even a stored `verified=true` can never restore `VerificationLedger` evidence or
authorize runtime behavior. The Rich dashboard is wrapped as best-effort presentation after the
durable trace sink; renderer failure disables the view rather than changing Agent control flow.

## Model transport boundary

The OpenAI adapter uses custom function tools only. It does not use an Agents SDK, hosted code/file
tools, automatic tool execution, or provider-managed conversation state. Requests set `store=false`;
each request explicitly sends a bounded context projection derived from the project-owned canonical
transcript. API keys come from the process environment, optionally populated by the untracked
`.env.local` boundary described above, are removed from child-command environments, and have no CLI
argument. Explicit remote base URLs must use HTTPS and cannot contain userinfo, a query,
or a fragment; plain HTTP is limited to loopback.

Tool schemas currently advertise `strict=false` because several local Pydantic schemas intentionally
contain optional fields with defaults; project-owned validation remains authoritative. SDK retries
default to zero so retry behavior is not hidden below runtime events and deadlines. The project-owned
runner handles two recoverable classes within one bounded attempt budget: connection/timeouts plus
HTTP 408/409/429/5xx use exponential backoff; malformed function-call argument JSON is rejected
before canonical history, tool dispatch, checkpoints, or tool-budget accounting and receives an
immediate protocol-correction request. Normal runs allow at most two retries and evaluation one.
Every attempt and retry category is observable. Correction prompts and terminal results are fixed,
sanitized text; provider arguments, headers, URLs, bodies, credentials, and exception chains are
never retained or surfaced.

Stateless reasoning continuation is an explicit limitation. If a response combines a reasoning item
with function calls, safe continuation would require retaining and replaying encrypted provider
state. M4 rejects that turn instead of silently discarding state or persisting hidden reasoning in a
provider-neutral checkpoint.
