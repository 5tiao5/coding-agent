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
