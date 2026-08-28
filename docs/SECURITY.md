# Workspace mutation security

The mutation layer is designed to prevent model mistakes and ordinary concurrent edits from
escaping or corrupting the selected workspace. It is not a sandbox against a malicious local
administrator or a hostile process that can continuously rewrite the directory tree.

## Invariants

- Every target is workspace-relative. Absolute, drive-qualified, parent-traversal, device,
  alternate-data-stream, control-character, ignored, internal, and sensitive paths fail closed.
- Parent directories must already exist. File tools cannot create directory trees or request a
  policy bypass.
- `.gitignore` files are readable policy inputs but are not mutation targets, so a change cannot
  rewrite the policy that its own undo must traverse.
- Symlinks, junctions/reparse points, multiply linked files, and Windows files carrying named
  data streams are not mutation targets.
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

POSIX commits use root-anchored directory descriptors and no-follow operations where the Python
runtime exposes them. Windows replacement remains path-based: existing files use `ReplaceFileW`
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
  timeout, combined bounded output, and credential-like environment variables removed. Shells,
  batch files, and a small hard-deny set of destructive programs are rejected.
- `safe` is the default policy: every ordinary external command requires approval before it can
  start. `auto` is an explicit programmatic opt-in; every ordinary command then invalidates prior
  verification because its effects are unknown.
- Verification is capability-first and also constrained to recognizable test, build, and check entry
  points. Only a host-registered exact executable path, argv, workspace-relative cwd, evidence kind,
  and label can issue a verification signal. Generic interpreter code, scripts, unknown launchers,
  workspace-owned executables, help, version, collection-only, fixture-listing, and no-run variants
  are rejected as evidence even if a host accidentally registers them. Output text is never parsed
  to decide success.
- The runner owns the command lifetime and attempts to terminate descendants on every exit path.
  A root process that exits while descendants remain after a short launcher-settle window is
  incomplete, even when its exit code is zero. If membership, containment, or cleanup cannot be
  confirmed, the runner emits a terminal control fact and the whole Agent run stops instead of
  accepting verification.

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
cgroup/container boundary for adversarial repositories; M3 does not claim to provide one itself.

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

M3 restores the canonical transcript only. `PlanState`, the mutation undo journal, CLI resume UX,
and interactive approval state are not yet persisted across processes.
