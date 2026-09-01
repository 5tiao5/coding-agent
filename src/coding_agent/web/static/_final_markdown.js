"use strict";

function asText(value) {
  return typeof value === "string" ? value : "";
}

function appendTextToken(tokens, value) {
  if (!value) {
    return;
  }
  const previous = tokens.at(-1);
  if (previous?.type === "text") {
    previous.value += value;
  } else {
    tokens.push({ type: "text", value });
  }
}

export function parseInlineMarkdown(value, allowStrong = true) {
  const source = asText(value);
  const tokens = [];
  let plainStart = 0;
  let index = 0;

  function flushPlain(end) {
    appendTextToken(tokens, source.slice(plainStart, end));
  }

  while (index < source.length) {
    if (source[index] === "`") {
      const closing = source.indexOf("`", index + 1);
      if (closing > index + 1) {
        flushPlain(index);
        tokens.push({ type: "code", value: source.slice(index + 1, closing) });
        index = closing + 1;
        plainStart = index;
        continue;
      }
    }

    if (allowStrong && source.startsWith("**", index)) {
      const closing = source.indexOf("**", index + 2);
      if (closing > index + 2) {
        flushPlain(index);
        tokens.push({
          children: parseInlineMarkdown(source.slice(index + 2, closing), false),
          type: "strong",
        });
        index = closing + 2;
        plainStart = index;
        continue;
      }
    }

    index += 1;
  }

  flushPlain(source.length);
  return tokens;
}

export function parseFinalMarkdown(value) {
  const lines = asText(value).replaceAll("\r\n", "\n").split("\n");
  const blocks = [];
  let codeLines = null;
  let list = null;

  function closeList() {
    if (list) {
      blocks.push(list);
      list = null;
    }
  }

  function closeCode() {
    if (codeLines) {
      blocks.push({ type: "code", value: codeLines.join("\n") });
      codeLines = null;
    }
  }

  function appendListItem(ordered, start, content) {
    if (!list || list.ordered !== ordered) {
      closeList();
      list = {
        items: [],
        ordered,
        start: ordered ? start : 1,
        type: "list",
      };
    }
    list.items.push(parseInlineMarkdown(content));
  }

  for (const line of lines) {
    if (/^\s{0,3}```/.test(line)) {
      closeList();
      if (codeLines) {
        closeCode();
      } else {
        codeLines = [];
      }
      continue;
    }
    if (codeLines) {
      codeLines.push(line);
      continue;
    }

    const headingMatch = line.match(/^\s{0,3}(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      closeList();
      blocks.push({
        content: parseInlineMarkdown(headingMatch[2]),
        level: headingMatch[1].length,
        type: "heading",
      });
      continue;
    }

    const orderedMatch = line.match(/^\s{0,3}(\d{1,9})\.\s+(.+)$/);
    if (orderedMatch) {
      appendListItem(true, Number.parseInt(orderedMatch[1], 10), orderedMatch[2]);
      continue;
    }

    const unorderedMatch = line.match(/^\s{0,3}[-+*]\s+(.+)$/);
    if (unorderedMatch) {
      appendListItem(false, 1, unorderedMatch[1]);
      continue;
    }

    closeList();
    if (!line.trim()) {
      continue;
    }
    blocks.push({ content: parseInlineMarkdown(line), type: "paragraph" });
  }

  closeList();
  closeCode();
  return blocks;
}

function appendInlineNodes(container, tokens, documentRef) {
  for (const token of tokens) {
    if (token.type === "code") {
      const code = documentRef.createElement("code");
      code.className = "inline-code";
      code.textContent = token.value;
      container.append(code);
    } else if (token.type === "strong") {
      const strong = documentRef.createElement("strong");
      appendInlineNodes(strong, token.children, documentRef);
      container.append(strong);
    } else {
      container.append(documentRef.createTextNode(token.value));
    }
  }
}

export function appendFinalMarkdown(container, value) {
  const documentRef = container.ownerDocument;
  for (const block of parseFinalMarkdown(value)) {
    if (block.type === "code") {
      const pre = documentRef.createElement("pre");
      pre.className = "answer-code";
      const code = documentRef.createElement("code");
      code.textContent = block.value;
      pre.append(code);
      container.append(pre);
      continue;
    }

    if (block.type === "list") {
      const list = documentRef.createElement(block.ordered ? "ol" : "ul");
      list.className = `answer-list answer-list-${block.ordered ? "ordered" : "unordered"}`;
      if (block.ordered && block.start !== 1) {
        list.start = block.start;
      }
      for (const itemTokens of block.items) {
        const item = documentRef.createElement("li");
        appendInlineNodes(item, itemTokens, documentRef);
        list.append(item);
      }
      container.append(list);
      continue;
    }

    const tagName = block.type === "heading" ? "h3" : "p";
    const element = documentRef.createElement(tagName);
    element.className =
      block.type === "heading"
        ? `answer-heading answer-heading-${block.level}`
        : "answer-paragraph";
    appendInlineNodes(element, block.content, documentRef);
    container.append(element);
  }
}
