"use strict";

export const COLLAPSED_DIFF_LINES = 8;

function recordValue(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : null;
}

function scalarText(value) {
  return typeof value === "string" ? value : "";
}

function finiteNumber(value, fallback = -1) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function stringLines(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((line) => typeof line === "string");
}

export function diffPreviewView(rawChange, expanded = false) {
  const change = rawChange && typeof rawChange === "object" ? rawChange : {};
  const compact = stringLines(change.preview);
  const expandedLines = stringLines(change.expanded_preview);
  const available = expandedLines.length ? expandedLines : compact;
  const totalLines = available.length;
  const canToggle = totalLines > COLLAPSED_DIFF_LINES;
  const isExpanded = canToggle && expanded === true;
  const visibleLines = isExpanded
    ? available
    : available.slice(0, COLLAPSED_DIFF_LINES);
  const previewTruncated = change.expanded_preview_complete === false;

  let toggleLabel = "";
  if (canToggle) {
    toggleLabel = isExpanded
      ? "收起 Diff"
      : previewTruncated
        ? `展开可用预览（${totalLines} 行）`
        : `展开 Diff（${totalLines} 行）`;
  }

  let note = `工作区变更已完整执行；当前展示 ${totalLines} 行安全 Diff 预览。`;
  if (previewTruncated) {
    note = "工作区变更已完整执行；此处仅显示受安全上限约束的 Diff 预览。";
  } else if (canToggle && !isExpanded) {
    note = `当前显示 ${visibleLines.length} / ${totalLines} 行安全 Diff 预览。`;
  }

  return {
    canToggle,
    isExpanded,
    note,
    previewTruncated,
    toggleLabel,
    totalLines,
    visibleLines,
  };
}

export function workspaceChangeKey(rawChange, runId) {
  const change = recordValue(rawChange) || {};
  return JSON.stringify([
    scalarText(runId),
    scalarText(change.activity_id),
    finiteNumber(change.step),
    finiteNumber(change.offset_seconds),
    scalarText(change.headline),
    scalarText(change.detail),
  ]);
}

export function workspaceChangeLedgerView(rawSnapshot) {
  const snapshot = recordValue(rawSnapshot) || {};
  const known = Array.isArray(snapshot.workspace_changes);
  const latest = recordValue(snapshot.latest_change);
  const changes = known
    ? snapshot.workspace_changes.filter((change) => recordValue(change) !== null)
    : latest
      ? [latest]
      : [];
  const rawOmitted = finiteNumber(snapshot.omitted_change_count, 0);
  const omittedCount = Math.max(0, Math.floor(rawOmitted));

  let note = "";
  if (known && snapshot.workspace_changes_complete === false) {
    note = omittedCount
      ? `较早的 ${omittedCount} 次工作区变更因展示上限未加载；当前显示最近 ${changes.length} 次。`
      : "较早的工作区变更因展示上限未加载。";
  }

  return { changes, known, note, omittedCount };
}
