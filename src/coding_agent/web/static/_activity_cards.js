import {
  activityCardView,
  activityEntryKey,
  visibleActivityEntries,
} from "./_activity_view.js";
import {
  translateActivityFactLabel,
  translateActivityFactValue,
  translateTimelineDetail,
  translateTimelineHeadline,
} from "./locale-zh.js";

"use strict";

const CATEGORY_META = {
  CONTEXT: { label: "上下文", symbol: "C" },
  MODEL: { label: "决策", symbol: "M" },
  PLAN: { label: "计划", symbol: "P" },
  RUN: { label: "运行", symbol: "R" },
  SAVE: { label: "存档", symbol: "S" },
  SESSION: { label: "会话", symbol: "S" },
  TOOL: { label: "工具", symbol: "T" },
  VERIFY: { label: "验证", symbol: "V" },
};

const LEVELS = new Set(["info", "success", "warning", "error"]);

function asObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asText(value, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function asNumber(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function createElement(tagName, className, text) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (text !== undefined) {
    element.textContent = text;
  }
  return element;
}

function categoryMeta(entry) {
  const key = asText(entry.category).toUpperCase();
  return CATEGORY_META[key] || { label: key || "活动", symbol: "·" };
}

function normalizedLevel(value) {
  return LEVELS.has(value) ? value : "info";
}

function formatDuration(durationMs) {
  const value = asNumber(durationMs, -1);
  if (value < 0) {
    return "";
  }
  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }
  return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)} s`;
}

function formatOffset(offsetSeconds) {
  const value = asNumber(offsetSeconds, -1);
  if (value < 0) {
    return "";
  }
  return `+${value.toFixed(value < 10 ? 1 : 0)}s`;
}

function renderPreview(preview) {
  const block = createElement("div", "tool-preview");
  for (const line of asArray(preview).filter((item) => typeof item === "string").slice(0, 12)) {
    block.append(createElement("code", "preview-line", line));
  }
  return block;
}

function renderActivityFact(fact) {
  const row = createElement("div", "activity-fact");
  row.append(
    createElement("dt", "activity-fact-label", translateActivityFactLabel(fact.label)),
  );
  const value = createElement("dd", "activity-fact-value");
  const translatedValue = translateActivityFactValue(fact.value, fact.format);
  if (fact.format === "code") {
    value.append(createElement("code", "activity-fact-code", translatedValue));
  } else if (fact.format === "status") {
    value.append(createElement("span", "activity-fact-status", translatedValue));
  } else {
    value.append(createElement("span", "", translatedValue));
  }
  row.append(value);
  return row;
}

export function createActivityCards() {
  let detailSequence = 0;
  const expandedKeys = new Set();

  function rememberExpanded(key) {
    if (expandedKeys.has(key)) {
      return;
    }
    if (expandedKeys.size >= 100) {
      const oldestKey = expandedKeys.values().next().value;
      if (oldestKey !== undefined) {
        expandedKeys.delete(oldestKey);
      }
    }
    expandedKeys.add(key);
  }

  function entries(rawEntries, limit) {
    const visible = visibleActivityEntries(rawEntries);
    return Number.isInteger(limit) && limit > 0 ? visible.slice(-limit) : visible;
  }

  function render(rawEntry, runId, ordinal) {
    const entry = asObject(rawEntry);
    const level = normalizedLevel(asText(entry.level));
    const meta = categoryMeta(entry);
    const activityKey = activityEntryKey(entry, runId, ordinal);
    const initialView = activityCardView(entry, expandedKeys.has(activityKey));
    const card = createElement("article", `activity-card level-${level}`);
    card.dataset.activityKey = activityKey;
    card.classList.toggle("has-details", initialView.canToggle);

    const rail = createElement("div", "activity-rail");
    rail.append(createElement("span", "activity-symbol", meta.symbol));

    const content = createElement("div", "activity-content");
    const heading = createElement(
      initialView.canToggle ? "button" : "div",
      `activity-heading${initialView.canToggle ? " activity-toggle" : ""}`,
    );
    if (initialView.canToggle) {
      heading.type = "button";
      heading.dataset.activityKey = activityKey;
    }
    const titleGroup = createElement("span", "activity-title-group");
    const translatedHeadline = translateTimelineHeadline(
      asText(entry.headline, "智能体活动"),
    );
    titleGroup.append(
      createElement("span", "activity-category", meta.label),
      createElement("strong", "activity-title", translatedHeadline),
    );

    const timing = createElement("span", "activity-timing");
    const duration = formatDuration(entry.duration_ms);
    const offset = formatOffset(entry.offset_seconds);
    if (duration) {
      timing.append(createElement("span", "", duration));
    }
    if (offset) {
      timing.append(createElement("span", "", offset));
    }
    let disclosureLabel = null;
    if (initialView.canToggle) {
      const disclosure = createElement("span", "activity-disclosure");
      disclosureLabel = createElement("span", "activity-disclosure-label");
      const chevron = createElement("span", "activity-chevron", "⌄");
      chevron.setAttribute("aria-hidden", "true");
      disclosure.append(disclosureLabel, chevron);
      timing.append(disclosure);
    }
    heading.append(titleGroup, timing);
    content.append(heading);

    const detail = asText(entry.detail).trim();
    if (detail) {
      content.append(
        createElement("p", "activity-detail", translateTimelineDetail(detail)),
      );
    }
    const preview = asArray(entry.preview);
    if (preview.length && asText(entry.category).toUpperCase() !== "TOOL") {
      content.append(renderPreview(preview));
    }

    if (initialView.canToggle) {
      detailSequence += 1;
      const factsId = `activity-facts-${detailSequence}`;
      const facts = createElement("section", "activity-facts");
      facts.id = factsId;
      facts.dataset.activityKey = activityKey;
      facts.tabIndex = 0;
      facts.setAttribute("role", "region");
      facts.setAttribute("aria-label", `${translatedHeadline}的操作详情`);
      const factList = createElement("dl", "activity-fact-list");
      factList.append(...initialView.facts.map(renderActivityFact));
      facts.append(factList);
      if (!initialView.factsComplete) {
        facts.append(
          createElement(
            "p",
            "activity-facts-note",
            "操作详情经过安全裁剪，部分信息未展示。",
          ),
        );
      }
      heading.setAttribute("aria-controls", factsId);

      function paint(expanded) {
        const view = activityCardView(entry, expanded);
        card.classList.toggle("is-expanded", view.isExpanded);
        facts.hidden = !view.isExpanded;
        heading.setAttribute("aria-expanded", String(view.isExpanded));
        heading.setAttribute("aria-label", `${translatedHeadline}：${view.toggleLabel}`);
        disclosureLabel.textContent = view.toggleLabel;
      }

      heading.addEventListener("click", () => {
        const shouldExpand = !expandedKeys.has(activityKey);
        if (shouldExpand) {
          rememberExpanded(activityKey);
        } else {
          expandedKeys.delete(activityKey);
        }
        paint(shouldExpand);
      });
      paint(initialView.isExpanded);
      content.append(facts);
    }

    card.append(rail, content);
    return card;
  }

  function captureInteraction(messageList, activeElement) {
    const scrollPositions = new Map();
    for (const detail of messageList.querySelectorAll(".activity-facts")) {
      const key = detail.dataset.activityKey;
      if (key) {
        scrollPositions.set(key, { left: detail.scrollLeft, top: detail.scrollTop });
      }
    }
    return {
      focusedKey: activeElement?.dataset?.activityKey || "",
      focusedTarget: activeElement?.classList?.contains("activity-facts")
        ? "facts"
        : "toggle",
      inspectionActive: Boolean(
        messageList.querySelector(".activity-card.is-expanded") ||
          activeElement?.classList?.contains("activity-toggle") ||
          activeElement?.classList?.contains("activity-facts"),
      ),
      scrollPositions,
    };
  }

  function restoreInteraction(messageList, state) {
    for (const detail of messageList.querySelectorAll(".activity-facts")) {
      const position = state.scrollPositions.get(detail.dataset.activityKey);
      if (position) {
        detail.scrollLeft = position.left;
        detail.scrollTop = position.top;
      }
    }
    if (!state.focusedKey) {
      return;
    }
    const selector = state.focusedTarget === "facts" ? ".activity-facts" : ".activity-toggle";
    for (const candidate of messageList.querySelectorAll(selector)) {
      if (candidate.dataset.activityKey === state.focusedKey) {
        candidate.focus({ preventScroll: true });
        break;
      }
    }
  }

  function clear() {
    expandedKeys.clear();
  }

  return { captureInteraction, clear, entries, render, restoreInteraction };
}
