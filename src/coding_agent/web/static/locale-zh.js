"use strict";

const PHASE_LABELS = {
  ACTING: "正在执行",
  COMPLETED: "已完成",
  "COMPLETED UNVERIFIED": "已完成，尚未验证",
  CREATED: "准备中",
  DECIDING: "正在决策",
  FAILED: "已失败",
  OBSERVING: "正在观察结果",
  PLANNING: "正在规划",
  REPLANNING: "正在拆分工具批次",
  RESUMED: "已恢复会话",
  RETRYING: "正在重试模型请求",
  VERIFYING: "正在验证",
  WAITING: "等待任务",
};

const TIMELINE_HEADLINES = {
  "Action selected": "下一步已确定",
  "Checkpoint not saved": "检查点未保存",
  "Checkpoint saved": "检查点已保存",
  "Context budget compacted": "上下文已压缩",
  "Failing evidence recorded": "已记录失败证据",
  "Final response deferred; verification scheduled": "暂缓结束，正在安排验证",
  "Passing evidence recorded": "已记录通过证据",
  "Previous evidence invalidated": "旧验证证据已失效",
  "Invalid model response; protocol correction scheduled": "模型工具参数格式异常，正在自动纠正",
  "Selecting the next action": "正在选择下一步",
  "Session resumed": "会话已恢复",
  "Task accepted": "任务已接收",
  "Tool batch too large; split retry requested": "工具批次过大，正在拆分重试",
  "Transient model failure; retry scheduled": "模型暂时不可用，已安排重试",
  "Verification gate evaluated": "验证门已评估",
};

const TIMELINE_DETAILS = {
  "A workspace change requires a fresh check": "工作区已变更，需要重新检查",
  Completed: "已完成",
  "Fresh verification is required": "需要重新验证",
  "Prepared a final response": "已准备最终回复",
  "READY FOR MODEL": "已准备请求模型",
  "Response received": "已收到模型回复",
  TERMINAL: "终态",
  "Workspace run started": "工作区任务已启动",
  FAILED: "失败",
  MISSING: "缺少证据",
  PASSED: "已通过",
  PENDING: "等待验证",
  STALE: "已过期",
  UNVERIFIED: "未验证",
  VERIFIED: "已验证",
};

const METADATA_LABELS = {
  "Ephemeral demo fixture": "临时演示工作区",
  "Local agent": "本地智能体",
  "Local repository": "本地仓库",
  "Offline deterministic demo": "离线确定性演示",
};

const RETRY_REASON_LABELS = {
  "MODEL REQUEST TRANSIENT": "瞬态请求错误",
  "MODEL REQUEST FAILURE": "模型请求异常",
  "MODEL RESPONSE INVALID": "模型工具参数格式无效",
};

function asText(value, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function mappedText(mapping, value) {
  return Object.hasOwn(mapping, value) ? mapping[value] : value;
}

export function translateMetadataLabel(value, fallback) {
  const text = asText(value, fallback);
  const translated = mappedText(METADATA_LABELS, text);
  if (translated !== text) {
    return translated;
  }
  if (text.endsWith(" · SAFE")) {
    return `${text.slice(0, -7)} · 安全模式`;
  }
  if (text.endsWith(" · AUTO")) {
    return `${text.slice(0, -7)} · 自动模式`;
  }
  return text;
}

export function translatePhase(value, activeTools) {
  const phase = asText(value).trim();
  const normalized = phase.replaceAll("_", " ").toUpperCase();
  if (normalized.startsWith("RUNNING ")) {
    const activeTool = Array.isArray(activeTools)
      ? activeTools.map((item) => asText(item, String(item))).find(Boolean)
      : "";
    return `正在运行 ${activeTool || phase.slice("RUNNING ".length)}`;
  }
  return Object.hasOwn(PHASE_LABELS, normalized) ? PHASE_LABELS[normalized] : phase;
}

export function historyFinalFallback(status) {
  const normalized = asText(status).trim();
  if (normalized === "interrupted") {
    return "此次运行的轨迹未正常终止；这里只回放中断前经过白名单过滤的事件。";
  }
  if (normalized === "failed") {
    return "此次运行以失败结束；这里只回放经过白名单过滤的事件，错误详情未纳入历史回放。";
  }
  return "未能从终态检查点恢复最终回复；代码修改与验证记录仍已保留。";
}

export function translateTimelineHeadline(value) {
  const text = asText(value).trim();
  const exact = mappedText(TIMELINE_HEADLINES, text);
  if (exact !== text) {
    return exact;
  }

  let match = text.match(/^Running (.+)$/);
  if (match) {
    return `正在运行 ${match[1]}`;
  }
  match = text.match(/^(.+) completed$/);
  if (match) {
    return `${match[1]} 已完成`;
  }
  match = text.match(/^(.+) failed$/);
  if (match) {
    return `${match[1]} 失败`;
  }
  return text;
}

function translateMutationSummary(text) {
  const match = text.match(
    /^(Created|Updated|Undid|Skipped unchanged|Replayed) (.+) \(\+(\d+)\/-([0-9]+), change (.+)\)$/,
  );
  if (!match) {
    return text;
  }
  const actions = {
    Created: "已创建",
    Replayed: "已重放",
    "Skipped unchanged": "未变更，已跳过",
    Undid: "已撤销",
    Updated: "已更新",
  };
  return `${actions[match[1]]} ${match[2]}（+${match[3]}/-${match[4]}，变更 ${match[5]}）`;
}

export function translateTimelineDetail(value) {
  const text = asText(value).trim();
  const exact = mappedText(TIMELINE_DETAILS, text);
  if (exact !== text) {
    return exact;
  }

  let match = text.match(/^Prepared (\d+) tool call\(s\)$/);
  if (match) {
    return `已准备 ${match[1]} 个工具调用`;
  }
  match = text.match(
    /^Requested (\d+) tool calls; per-turn limit (\d+); split retry requested(?:; rejection (\d+) of (\d+))?$/,
  );
  if (match) {
    const retry = match[3] && match[4] ? `（第 ${match[3]} / ${match[4]} 次调整）` : "";
    return `本轮请求 ${match[1]} 个工具调用，上限 ${match[2]}；已要求模型拆分重试${retry}`;
  }
  match = text.match(/^Summarized (\d+) older tool block\(s\)$/);
  if (match) {
    return `已摘要 ${match[1]} 个较早的工具结果块`;
  }
  match = text.match(
    /^Attempt (\d+) of (\d+) · after ([0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?)s · (.+)$/i,
  );
  if (match) {
    const reason = mappedText(RETRY_REASON_LABELS, match[4]);
    const schedule = Number(match[3]) === 0 ? "立即重试" : `${match[3]} 秒后重试`;
    return `第 ${match[1]} / ${match[2]} 次请求 · ${schedule} · ${reason}`;
  }
  match = text.match(/^Resumed after (\d+) completed step\(s\); fresh verification is required$/);
  if (match) {
    return `已从 ${match[1]} 个完成步骤后恢复；需要重新验证`;
  }
  match = text.match(/^Plan revision (\d+): (\d+) pending, (\d+) in progress, (\d+) completed$/);
  if (match) {
    return `计划版本 ${match[1]}：${match[2]} 项待处理，${match[3]} 项进行中，${match[4]} 项已完成`;
  }
  match = text.match(/^Listed (\d+) of (\d+) discovered entries under (.+)$/);
  if (match) {
    return `已列出 ${match[1]} / ${match[2]} 个条目，位置：${match[3]}`;
  }
  match = text.match(/^Read empty file (.+)$/);
  if (match) {
    return `已读取空文件 ${match[1]}`;
  }
  match = text.match(/^Read (.+) lines (\d+)-(\d+) of (\d+)$/);
  if (match) {
    return `已读取 ${match[1]} 第 ${match[2]}-${match[3]} 行（共 ${match[4]} 行）`;
  }
  match = text.match(/^Found (at least )?(\d+) matches and showed (\d+) from (\d+) files under (.+)$/);
  if (match) {
    const count = match[1] ? `至少 ${match[2]}` : match[2];
    return `找到 ${count} 处匹配，展示 ${match[3]} 处；扫描 ${match[4]} 个文件，位置：${match[5]}`;
  }
  match = text.match(/^Command exited (-?\d+) in (.+)$/);
  if (match) {
    return `命令在 ${match[2]} 退出，退出码 ${match[1]}`;
  }
  match = text.match(/^Command timed out after ([0-9.]+)s in (.+)$/);
  if (match) {
    return `命令在 ${match[2]} 超时（${match[1]} 秒）`;
  }
  match = text.match(/^Command process control failed in (.+); run must stop$/);
  if (match) {
    return `命令在 ${match[1]} 的进程控制失败；运行必须停止`;
  }

  return translateMutationSummary(text);
}

export function translateServerDetail(value) {
  const text = asText(value);
  if (
    text === "this local UI is locked to its configured task" ||
    text === "此本地界面已锁定为预设演示任务"
  ) {
    return "此本地界面仅允许运行预设任务。";
  }
  return text;
}
