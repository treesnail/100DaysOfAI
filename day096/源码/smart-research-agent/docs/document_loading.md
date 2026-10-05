# 文档解析与加载手册（day061 / M6-D1）

> 本文与代码**同源**：每一张表都能由 `LoaderRegistry.table()` /
> `encoding_chain()` / `markdown_syntax_table()` / `html_rules()` /
> `pdf_boundaries()` / `docx_scope()` 现场打印出来，每一个数字都由
> `scripts/documents_demo.py` 实测。

## 1. 一条链：从字节到下游能消费的形状

```text
各种格式  →  Document（全文 + 块序列 + 元数据 + 内容指纹）  →  分块 / 向量化 / 检索
```

```text
base.py             什么文件用哪个解析器（注册表 + 内容优先的类型识别 + 大小护栏）
types.py            归一化后的形状（六种块 / 内容指纹 / 唯一标准形）
text_loader.py      编码怎么定（五步固定顺序 + 「定不下来」要显式标记）
markdown_loader.py  ATX 标题 / 围栏代码 / GFM 表格 / 列表 / 引用 / front matter
html_loader.py      脚本与样式整体跳过、实体解码、表格保留行列
docx_loader.py      只读 word/document.xml，边界写进元数据
pdf_loader.py       内容流 + zlib + 文本操作符 + 行距分段，边界写满
pipeline.py         内容指纹去重 + 三类状态 + 确定性遍历
```

## 2. 五类媒体类型与五个加载器（一一对应）

| 媒体类型 | 加载器 | 后缀 |
|---------|--------|------|
| `text/plain` | `TextLoader` | `.txt` `.text` `.log` |
| `text/markdown` | `MarkdownLoader` | `.md` `.markdown` |
| `text/html` | `HtmlLoader` | `.html` `.htm` |
| `application/pdf` | `PdfLoader` | `.pdf` |
| `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | `DocxLoader` | `.docx` |

**一一对应是刻意的**：一个加载器声称支持两种媒体类型时，它内部一定有一个
`if`，而那个 `if` 应该变成两个加载器。

## 3. 类型识别：内容优先，冲突必报

| 后缀 | 内容 | 结果 |
|------|------|------|
| 有 | 有且一致 | 按它 |
| 有 | 有但不同 | **按内容**，`conflict=True` |
| 有 | 无 | 按后缀 |
| 无 | 有 | 按内容 |
| 无 | 无 | `text/plain`（`decided_by="fallback"`） |

实测：

```text
guide.md       → text/markdown（按 suffix 判断）
fake.txt       → application/pdf（**冲突**：后缀说 text/plain、内容说 application/pdf；按内容处理）
report.docx    → ...wordprocessingml.document（按 suffix 判断）
noextension    → text/plain（按 fallback 判断）
```

`docx` 是唯一一处**特例**：它的魔数 `PK\x03\x04` 只能证明"这是一个 ZIP"，
因此"后缀 docx + 内容 ZIP"不算冲突——那是同一件事的两种描述。

## 4. 编码探测：五步固定顺序

| 步骤 | 规则 | 为什么排在这里 |
|------|------|---------------|
| 1 | BOM（utf-32-be / utf-32-le / utf-8-sig / utf-16-be / utf-16-le） | 有 BOM 就是明示；**UTF-32 必须排在 UTF-16 之前** |
| 2 | 二进制探测（头 8 KiB 出现 NUL 即拒收） | 判据只有 NUL 一个（误判的代价不对称） |
| 3 | utf-8 **严格**解码 | 严格让"成功"本身成为可靠证据 |
| 4 | `gb18030` | GB18030 是 GBK 与 GB2312 的超集，一个编码覆盖三代 |
| 5 | `latin-1` | 永不失败，因此只能排最后；结果标记 `encoding_confident=false` |

实测：

```text
utf-8 中文          → utf-8（按 strict 判断，可信，2 字符）
GB18030 中文        → gb18030（按 fallback 判断，可信，2 字符）
带 BOM 的 utf-16-le → utf-16-le（按 bom 判断，可信，2 字符）
两者都不成立        → latin-1（按 latin-1 判断，**不可信**，2 字符）
```

第 1 步的顺序敏感：`\xff\xfe` 是 UTF-16-LE 的 BOM，而 UTF-32-LE 的 BOM 是
`\xff\xfe\x00\x00`。先判 UTF-16-LE 会把 UTF-32 的文件解成"每个字符后面跟一个 NUL"，
而那种乱码看起来很像文件损坏。

## 5. 归一化：`doc_id` 建立在它上面

```text
doc_id = sha256(规范化全文)[:16]
```

| 规则 | 消除的差异 |
|------|-----------|
| `\r\n` / `\r` → `\n` | Windows 与 Unix 换行 |
| 去掉每行行尾空白 | 编辑器留下的尾随空格 |
| 连续空行折叠到 1 个 | 导出的多余空行 |
| 去掉首尾空白并在结尾补一个 `\n` | 有无结尾换行 |

**不做 Unicode 规范化**（不把全角标点换成半角）：中英文混排的技术文档里，
全角逗号常是内容差异而不是格式差异，合并它会让检索结果悄悄改变。

`doc_id` 的三条性质：同一份内容从两个路径读进来得到同一个 id（去重）；
改一个字节 id 就变（内容寻址）；它不包含文件名、路径与时间。

## 6. 六种块

| 块 | 来源 | 谁需要它 |
|----|------|---------|
| `heading` | `# 标题` / `<h1>` / `w:pStyle=Heading1` | day062 的结构分块、day069 的 RAG 提示 |
| `paragraph` | 空行分段 / `<p>` / `w:p` / PDF 行距分段 | 分块的基本单位 |
| `code` | 围栏代码块 / `<pre>` | 不该被从中间切开 |
| `list_item` | `- ` / `<li>` / `w:numPr` | 保留层级信息 |
| `table` | GFM 表格 / `<table>` / `w:tbl` | 表格问答（渲染成 Markdown 喂模型） |
| `quote` | `> ` / `<blockquote>` | 引用溯源 |

三种格式的表格**共用一个渲染器**（`types.render_table`）：同一份事实
只有一份渲染实现，否则三种格式的表格会以三种样子进提示词。

## 7. 各格式的边界（本日最重要的一节）

| 格式 | 做不到 |
|------|--------|
| PDF | CID 字体（Identity-H）的字形映射（规范 §9.10.2 判定为不可确定）；`/ToUnicode` CMap；扫描件；双栏阅读顺序；除 FlateDecode 外的过滤器；加密 PDF（拒收） |
| DOCX | 旧版 `.doc`；页眉页脚、脚注、批注、文本框（不在 `document.xml` 里）；图片与嵌入对象 |
| HTML | 未闭合标签的容错修复（标准库 `html.parser` 不做隐式修复）；CSS 选择器语义；动态渲染的内容（JS 执行后的 DOM） |
| Markdown | 行内标记的渲染（`**粗体**` / `[链接](url)` 原样保留）；嵌套 YAML 的解析（只识别 `key: value` 与 `key: [a, b]`） |
| 全部 | 编码探测失败时（latin-1）的乱码识别；事实正确性（解析对了不代表内容对） |

## 8. 批量入库

```text
文档 → sha256(规范化文本)[:16] → 先看到的留下，后来者标记 duplicate_of
```

三类状态，**三个键恒存在**（报告的表头不该随内容变化）：

```text
ok          解析成功且是首次出现
duplicate   与前一份文档内容相同（指向重复对象）
error       不在支持范围（或解析失败）
```

实测（9 个文件）：

```text
9 个文件 → 6 份文档 | ok 6 / 重复 1 / 失败 2 | 478 字符
   [ok] docs/guide.md | text/markdown | 264 字符 / 9 块 | 2de1fccfba62a249
   [重复] docs/guide-copy.md（与 docs/guide.md 内容相同）
   [error] docs/broken.txt —— 看起来是二进制文件（头 8192 字节里出现 NUL）
   [error] docs/fake.pdf —— 不是 PDF（头四个字节不是 %PDF-）
```

遍历是**排序后**处理的，因此同一批文件在任何机器上得到同一份结果——
"这次入库比上次少了三个文档"必须能被回答。

## 9. 与知识库的接缝

```python
{
  "doc_id": "2de1fccfba62a249",     # 内容指纹
  "source": "docs/design.md",       # 来源标签（只作为标签，不回头去读它）
  "text": "…",                      # 规范化后的全文
  "metadata": {...},                # 媒体类型 + 加载器写的诊断字段
}
```

这个形状就是 day009 的 `VectorStore.add` 需要的。**今天不做分块**——
day062 会在这里切出片段，而**接口形状不变**。

## 10. 明确不适用

- 直接按本课的抽取结果做法律或财务文档处理：PDF 的阅读顺序与表格还原都不可靠；
- 把入库当作"文件已备份"：本模块只读不写，不做任何完整性校验；
- 用本模块替代 pypdf（6.18.0）/ pdfplumber（0.11.10）/ python-docx（1.2.0）：
  不复用它们只为了保证离线零依赖，**不是因为在质量上等价**。
