"use strict";

export const MAX_ACTIVITY_FACTS = 8;

const ACTIVITY_STATES = new Set(["started", "finished"]);
const FACT_FORMATS = new Set(["text", "code", "status"]);

function asObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asText(value, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function activityId(rawEntry) {
  return asText(asObject(rawEntry).activity_id).trim();
}

function activityState(rawEntry) {
  const value = asText(asObject(rawEntry).activity_state).trim().toLowerCase();
  return ACTIVITY_STATES.has(value) ? value : null;
}

function normalizedFact(rawFact) {
  const fact = asObject(rawFact);
  const label = asText(fact.label).trim();
  const value = asText(fact.value).trim();
  if (!label || !value) {
    return null;
  }
  const rawFormat = asText(fact.format).trim().toLowerCase();
  return {
    format: FACT_FORMATS.has(rawFormat) ? rawFormat : "text",
    label,
    value,
  };
}

export function visibleActivityEntries(rawEntries) {
  const entries = asArray(rawEntries);
  const finishedIds = new Set();
  for (const rawEntry of entries) {
    const id = activityId(rawEntry);
    if (id && activityState(rawEntry) === "finished") {
      finishedIds.add(id);
    }
  }
  return entries.filter((rawEntry) => {
    const id = activityId(rawEntry);
    return !(id && activityState(rawEntry) === "started" && finishedIds.has(id));
  });
}

export function activityEntryKey(rawEntry, runId = "", ordinal = 0) {
  const entry = asObject(rawEntry);
  const id = activityId(entry);
  if (id) {
    if (activityState(entry) !== null) {
      return JSON.stringify([asText(runId), "activity", id, "paired-tool"]);
    }
    return JSON.stringify([
      asText(runId),
      "activity",
      id,
      asText(entry.category),
      asText(entry.headline),
      Number.isFinite(entry.step) ? entry.step : -1,
      Number.isFinite(entry.offset_seconds) ? entry.offset_seconds : -1,
    ]);
  }
  return JSON.stringify([
    asText(runId),
    "timeline",
    Number.isFinite(entry.step) ? entry.step : -1,
    asText(entry.category),
    asText(entry.headline),
    Number.isFinite(entry.offset_seconds) ? entry.offset_seconds : -1,
    ordinal,
  ]);
}

export function activityCardView(rawEntry, expanded = false) {
  const entry = asObject(rawEntry);
  const rawFacts = asArray(entry.facts);
  const facts = rawFacts
    .slice(0, MAX_ACTIVITY_FACTS)
    .map(normalizedFact)
    .filter((fact) => fact !== null);
  const canToggle = facts.length > 0;
  const factsComplete =
    entry.facts_complete === true &&
    rawFacts.length <= MAX_ACTIVITY_FACTS &&
    facts.length === rawFacts.length;
  return {
    canToggle,
    facts,
    factsComplete,
    isExpanded: canToggle && expanded === true,
    toggleLabel: expanded ? "收起操作详情" : "查看操作详情",
  };
}
