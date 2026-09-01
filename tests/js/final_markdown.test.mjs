import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const modulePath = "src/coding_agent/web/static/_final_markdown.js";

async function importMarkdownModule() {
  const source = readFileSync(path.join(repositoryRoot, modulePath), "utf8");
  const encoded = Buffer.from(source).toString("base64");
  return import(`data:text/javascript;base64,${encoded}`);
}

class FakeNode {
  constructor(ownerDocument, tagName = "#text", value = "") {
    this.children = [];
    this.className = "";
    this.ownerDocument = ownerDocument;
    this.start = 1;
    this.tagName = tagName;
    this.value = value;
  }

  append(...children) {
    this.children.push(...children);
  }

  set textContent(value) {
    this.children = [new FakeNode(this.ownerDocument, "#text", String(value))];
  }
}

class FakeDocument {
  createElement(tagName) {
    return new FakeNode(this, tagName.toUpperCase());
  }

  createTextNode(value) {
    return new FakeNode(this, "#text", value);
  }
}

function collectTags(node) {
  return [node.tagName, ...node.children.flatMap(collectTags)];
}

function collectText(node) {
  return node.tagName === "#text" ? node.value : node.children.map(collectText).join("");
}

test("final markdown parses unordered and ordered lists with inline formatting", async () => {
  const { parseFinalMarkdown } = await importMarkdownModule();
  const blocks = parseFinalMarkdown(
    [
      "普通说明。",
      "",
      "- **README.md**：项目说明",
      "- 运行 `python -m routeforge`",
      "",
      "2. **pytest** 全部通过",
      "3. 无界面检查通过",
    ].join("\n"),
  );

  assert.deepEqual(blocks.map((block) => block.type), ["paragraph", "list", "list"]);
  assert.equal(blocks[1].ordered, false);
  assert.equal(blocks[1].items[0][0].type, "strong");
  assert.equal(blocks[1].items[0][0].children[0].value, "README.md");
  assert.equal(blocks[1].items[1][1].type, "code");
  assert.equal(blocks[1].items[1][1].value, "python -m routeforge");
  assert.equal(blocks[2].ordered, true);
  assert.equal(blocks[2].start, 2);
  assert.equal(blocks[2].items.length, 2);
});

test("unmatched markdown markers remain ordinary text", async () => {
  const { parseInlineMarkdown } = await importMarkdownModule();

  assert.deepEqual(parseInlineMarkdown("保留 **未闭合 与 `反引号"), [
    { type: "text", value: "保留 **未闭合 与 `反引号" },
  ]);
});

test("final markdown renders malicious HTML as inert text nodes", async () => {
  const { appendFinalMarkdown } = await importMarkdownModule();
  const documentRef = new FakeDocument();
  const container = new FakeNode(documentRef, "DIV");
  const malicious = '<img src=x onerror="alert(1)"> **安全粗体** `\u003cscript>boom\u003c/script>`';

  appendFinalMarkdown(container, malicious);

  assert.deepEqual(collectTags(container), ["DIV", "P", "#text", "STRONG", "#text", "#text", "CODE", "#text"]);
  assert.equal(collectTags(container).includes("IMG"), false);
  assert.equal(collectTags(container).includes("SCRIPT"), false);
  assert.equal(
    collectText(container),
    '<img src=x onerror="alert(1)"> 安全粗体 \u003cscript>boom\u003c/script>',
  );
});

test("final markdown renderer never uses an HTML injection sink", () => {
  const source = readFileSync(path.join(repositoryRoot, modulePath), "utf8");

  assert.doesNotMatch(source, /\b(?:innerHTML|outerHTML|insertAdjacentHTML)\b/);
});
