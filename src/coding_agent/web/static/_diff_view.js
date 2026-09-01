"use strict";

export const COLLAPSED_DIFF_LINES = 8;

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
