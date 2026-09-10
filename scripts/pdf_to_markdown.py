# -*- coding: utf-8 -*-
"""
把技术手册类 PDF 转成适合入知识库的 Markdown。

本项目只吃 Markdown，而实际语料往往是 PDF 手册，所以需要这一步。
转换质量直接决定检索上限，重点处理四件在 PDF 里普遍存在、
又会实实在在污染检索结果的事：

**页眉页脚**
  每页重复的版权声明、文档编号、页码。逐页出现意味着它们在语料里的
  出现频次远高于任何真实内容，BM25 会把它们当成高频噪声，
  向量化时也会稀释正文语义。这里不写死规则，而是统计每行在多少页出现过，
  超过阈值即判定为版面元素删掉 —— 换一份 PDF 也能用。
  书眉（「文档名 + 当前章节名」）每页都变，频次法抓不住，
  改用「多数页首行共有的前缀」来识别。

**标题层级**
  先从目录页解析出权威的「编号 + 标题」清单，再回到正文里匹配。
  不直接用「行首是数字」这类正则：正文里的步骤编号、版本号、命令输出
  都长得像标题，误判会把一节切碎成十几个假章节。

**段落重建**
  PDF 抽出来的文本每行一断，没有空行。而本项目的切分器靠空行分段，
  不重建段落的话整节会被当成一个巨大的单元，只能按字符硬切。

**代码块**
  命令与终端输出必须整体保留：被拆进不同文本块的命令检索出来没有意义，
  而且它们不该被当作句子去和查询算语义相似度。

用法:
    python scripts/pdf_to_markdown.py input.pdf -o output.md
    python scripts/pdf_to_markdown.py input.pdf -o out.md --title "运维手册"
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: 目录项：编号 + 标题 + 点线 + 页码
_TOC_ENTRY_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+?)\s*\.{3,}\s*(\d+)\s*$")

#: 正文中的编号标题。编号与标题之间的空格可有可无——排版偶尔会挤掉它
#: （"7.1Debug logs..."）。放宽到零空格不会带来误判，因为标题文字还要
#: 和目录里的条目对得上才算数。
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)*)\s*(\S.*)$")

#: 有序步骤，如 "1. Log in to ..."
_STEP_RE = re.compile(r"^(\d+)\.\s+(\S.*)$")

#: 项目符号
_BULLET_RE = re.compile(r"^\s*[•▪◦·]\s*(.*)$")

#: 图表标题
_CAPTION_RE = re.compile(r"^(Figure|Table)\s+\d+:\s*(.*)$", re.IGNORECASE)

#: 行首即可判定为命令的关键字
_COMMAND_WORDS = (
    "kubectl", "helm", "oc ", "docker", "podman", "systemctl", "journalctl",
    "netstat", "pstack", "openssl", "curl", "wget", "ssh", "scp", "tar",
    "cat ", "ls ", "cd ", "ps ", "grep ", "tail ", "head ", "awk ", "sed ",
    "jq ", "chmod", "chown", "mkdir", "rm ", "cp ", "mv ", "export ",
    "source ", "echo ", "kill ", "su ", "sudo ", "python", "sh ", "bash ",
    "./", "cnAdminTool", "cmm ", "vi ",
)

#: 终端提示符，如 "[root@host ~]#"
_PROMPT_RE = re.compile(r"^\[[\w.\-]+@[\w.\-]+[^\]]*\]\s*[#$]")

#: 折行后只剩提示符尾巴的命令行，如 "# kubectl get pod ..."。
#: 原始 PDF 里不存在 Markdown 语法，因此行首的 # 一定是 shell 提示符。
#: 不识别的话它会原样进入 Markdown，被渲染成一级标题，
#: 同时也躲开了代码围栏，与后面的命令输出粘成一个段落。
_BARE_PROMPT_RE = re.compile(r"^[#$]\s+\S")

#: 看起来是命令输出的行：全大写表头、键值对、纯路径等
_OUTPUT_LIKE_RE = re.compile(
    r"^(NAME\s|READY\s|STATUS\s|/[\w./*\-]+\s*$|\{.*\}\s*$|\S+=\S+)"
)

#: 已知的字距断裂。PDF 里字母对被拆开，抽取时多出一个空格。
#: 只修确定的几个，不用通用正则 —— "A primary" 这类合法搭配会被误伤。
_KERNING_FIXES = {
    "T elesys": "Telesys",
    "T eam": "Team",
    "T erminate": "Terminate",
    "T able": "Table",
    "T ool": "Tool",
    "T o ": "To ",
    "T he ": "The ",
    "T his ": "This ",
}


def extract_pages(pdf_path: Path) -> list[str]:
    """逐页抽取文本。"""
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise SystemExit("未安装 pypdf，请先运行：pip install pypdf") from exc

    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or "") for page in reader.pages]


def detect_furniture(pages: list[str], ratio: float = 0.5) -> set[str]:
    """
    找出重复出现在多数页面上的版面元素。

    用频次而不是写死的规则来识别，是为了让脚本能换一份 PDF 继续用：
    每份手册的页眉页脚内容都不同，但「每页都出现」这个特征是共通的。
    正文行几乎不可能在半数以上的页面里逐字重复。
    """
    seen_in_pages: Counter[str] = Counter()
    for text in pages:
        unique = {line.strip() for line in text.split("\n") if line.strip()}
        seen_in_pages.update(unique)

    threshold = max(2, int(len(pages) * ratio))
    return {line for line, count in seen_in_pages.items() if count >= threshold}


def detect_running_header(pages: list[str], ratio: float = 0.5) -> str:
    """
    找出书眉的固定前缀。

    书眉形如「文档名 + 当前章节名」，后半段每页都变，逐行比对抓不到，
    但前半段是全书共有的。做法是从各页首行出发逐个单词地延长前缀，
    直到某个单词不再被多数页面共有为止。

    不用相邻两页求公共前缀：同一章节内连续几页的首行完全相同，
    那样求出来的「公共前缀」会含上章节名，各章节各得一个，反而散掉。
    """
    firsts = []
    for text in pages:
        for line in text.split("\n"):
            if line.strip():
                firsts.append(line.strip().split())
                break

    threshold = len(pages) * ratio
    prefix: list[str] = []
    while True:
        following: Counter[str] = Counter()
        for words in firsts:
            if len(words) > len(prefix) and words[: len(prefix)] == prefix:
                following[words[len(prefix)]] += 1
        if not following:
            break
        word, count = following.most_common(1)[0]
        if count < threshold:
            break
        prefix.append(word)

    candidate = " ".join(prefix)
    return candidate if len(candidate) >= 12 else ""


def parse_toc(pages: list[str]) -> tuple[dict[str, str], set[int]]:
    """
    从目录页解析出权威的章节清单。

    返回 ({编号: 标题}, 目录页页码集合)。

    标题在目录里可能折行，因此先把带点线的行与其前一行拼回去再解析。
    拿到这份清单后，正文里的标题识别就变成「精确匹配」而不是「猜」，
    误判率从「每页几个」降到 0。
    """
    toc: dict[str, str] = {}
    toc_pages: set[int] = set()

    for index, text in enumerate(pages):
        lines = [line.strip() for line in text.split("\n")]
        # 点线密集才认为是目录页，正文里偶尔出现的省略号不会触发
        if sum(1 for line in lines if re.search(r"\.{4,}", line)) < 5:
            continue
        toc_pages.add(index)

        buffer = ""
        for line in lines:
            if not line:
                continue
            candidate = f"{buffer} {line}".strip() if buffer else line
            match = _TOC_ENTRY_RE.match(candidate)
            if match:
                number, title, _page = match.groups()
                toc[number] = _fix_text(title.strip())
                buffer = ""
            elif _NUMBERED_RE.match(line) and not re.search(r"\.{4,}", line):
                # 编号开头但还没出现点线，说明标题折到下一行了
                buffer = line
            else:
                buffer = ""

    return toc, toc_pages


def _fix_text(text: str) -> str:
    """修掉 PDF 抽取带来的字距断裂与多余空白。"""
    for wrong, right in _KERNING_FIXES.items():
        text = text.replace(wrong, right)
    return re.sub(r"[ \t]+", " ", text).strip()


def is_code_line(line: str) -> bool:
    """判断一行是否属于命令或终端输出。"""
    stripped = line.strip()
    if not stripped:
        return False
    if _PROMPT_RE.match(stripped) or _BARE_PROMPT_RE.match(stripped):
        return True
    if any(stripped.startswith(word) for word in _COMMAND_WORDS):
        return True
    if _OUTPUT_LIKE_RE.match(stripped):
        return True
    return False


def looks_like_prose(line: str) -> bool:
    """
    判断一行是否是自然语言。

    用于决定代码块在哪里结束：命令输出后面跟着一句正常的说明文字时，
    围栏必须在此收口，否则会把整节正文都吞进代码块。
    """
    stripped = line.strip()
    if not stripped or len(stripped) < 12:
        return False
    if _PROMPT_RE.match(stripped) or _OUTPUT_LIKE_RE.match(stripped):
        return False
    words = stripped.split()
    if len(words) < 3:
        return False
    # 自然语言里字母占绝大多数，命令输出里符号和数字多
    letters = sum(1 for ch in stripped if ch.isalpha() or ch == " ")
    return letters / len(stripped) > 0.75 and stripped[0].isupper()


def is_label_line(line: str) -> bool:
    """
    识别「Expected outcome」「For Example,」这类夹在命令之间的短标签。

    它们太短，looks_like_prose 判不出来，于是被卷进代码围栏里。
    单独认出来才能让围栏在正确的位置收口。
    排除全大写，否则 "NAME READY STATUS RESTARTS" 这种输出表头会被误伤。
    """
    stripped = line.strip()
    if not stripped or stripped.isupper():
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z ,:'()\-]*", stripped):
        return False
    return len(stripped.split()) <= 5


def clean_page(text: str, furniture: set[str], header_prefix: str) -> list[str]:
    """去掉版面元素，返回该页的正文行。"""
    out: list[str] = []
    for raw in text.split("\n"):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in furniture:
            continue
        if header_prefix and stripped.startswith(header_prefix):
            continue
        # 孤立的页码行
        if re.fullmatch(r"\d{1,3}", stripped):
            continue
        if is_code_line(line):
            out.append(line.rstrip())
        else:
            # 抽取时小节编号偶尔被拆成 "2. 1"，粘回去才认得出是标题
            out.append(re.sub(r"^(\d+)\.\s+(?=\d)", r"\1.", _fix_text(line)))
    return out


class MarkdownBuilder:
    """按块累积输出，负责段落合并与代码围栏的开合。"""

    def __init__(self) -> None:
        self.blocks: list[str] = []
        self._para: list[str] = []
        self._code: list[str] = []

    @property
    def in_code_block(self) -> bool:
        """当前是否正处在一段尚未收口的代码围栏中。"""
        return bool(self._code)

    def _flush_para(self) -> None:
        if self._para:
            self.blocks.append(" ".join(self._para))
            self._para = []

    def _flush_code(self) -> None:
        if self._code:
            # 去掉尾部空行再收口，避免围栏里挂一串空白
            while self._code and not self._code[-1].strip():
                self._code.pop()
            self.blocks.append("```\n" + "\n".join(self._code) + "\n```")
            self._code = []

    def flush(self) -> None:
        self._flush_para()
        self._flush_code()

    def add_block(self, text: str) -> None:
        """加入一个独立成段的块（标题、图表标题等）。"""
        self.flush()
        self.blocks.append(text)

    def add_code(self, line: str) -> None:
        self._flush_para()
        # PDF 会把过长的命令折行，末尾留一个连字符。不接回去的话
        # 命名空间之类的标识符就断了，检索出来的命令是错的。
        if self._code and self._code[-1].endswith("-") and re.fullmatch(r"\S+", line.strip()):
            self._code[-1] = self._code[-1] + line.strip()
            return
        self._code.append(line)

    def add_prose(self, line: str) -> None:
        """
        累积正文行，把 PDF 的硬折行还原成段落。

        判定续行的主依据是「上一行没有句末标点，且长度接近满行」——
        两端对齐的正文被折行时每行都接近版心宽度，而
        「Expected outcome」这类独立标签则明显偏短。
        只看首字母大小写不够：英文句子折行后下一行常以专有名词开头。
        """
        self._flush_code()
        if self._para:
            previous = self._para[-1]
            wrapped = not previous.endswith((".", ":", ";", "!", "?")) and len(previous) >= 60
            if not (wrapped or line[:1].islower() or previous.endswith((",", "-", "、"))):
                self._flush_para()
        self._para.append(line)

    def render(self) -> str:
        self.flush()
        return "\n\n".join(self.blocks) + "\n"


def build_markdown(pages: list[str], title: str, toc: dict[str, str],
                   toc_pages: set[int], furniture: set[str],
                   header_prefix: str) -> str:
    """
    把清洗后的页面组装成 Markdown。

    目录之前的封面与法律声明整段丢弃：这些文字与文档主题无关，
    但篇幅不小、措辞正式，留在语料里只会成为检索噪声。
    """
    builder = MarkdownBuilder()
    builder.add_block(f"# {title}")
    body_starts = min(toc_pages) if toc_pages else 0

    for index, text in enumerate(pages):
        if index in toc_pages or index < body_starts:
            continue

        for line in clean_page(text, furniture, header_prefix):
            stripped = line.strip()

            # 标题：编号必须在目录里出现过，且标题文字对得上
            match = _NUMBERED_RE.match(stripped)
            if match:
                number, rest = match.groups()
                expected = toc.get(number)
                if expected and rest.strip().lower().startswith(expected[:24].lower()):
                    level = min(6, number.count(".") + 2)
                    builder.add_block(f"{'#' * level} {number} {expected}")
                    continue

            caption = _CAPTION_RE.match(stripped)
            if caption:
                builder.add_block(f"*{stripped}*")
                continue

            bullet = _BULLET_RE.match(stripped)
            if bullet:
                builder.add_block(f"- {_fix_text(bullet.group(1))}")
                continue

            step = _STEP_RE.match(stripped)
            if step and looks_like_prose(step.group(2)):
                builder.add_block(f"{step.group(1)}. {step.group(2)}")
                continue

            if is_code_line(line):
                builder.add_code(line)
                continue

            # 已在代码块中：非散文、非标签的行继续留在围栏内
            if builder.in_code_block and not looks_like_prose(line) and not is_label_line(line):
                builder.add_code(line)
                continue

            builder.add_prose(stripped)

    return builder.render()


def main() -> int:
    parser = argparse.ArgumentParser(description="PDF 转 Markdown（面向知识库入库）")
    parser.add_argument("pdf", help="输入 PDF")
    parser.add_argument("-o", "--output", required=True, help="输出 Markdown")
    parser.add_argument("--title", default="", help="文档标题，默认取文件名")
    parser.add_argument("--furniture-ratio", type=float, default=0.5,
                        help="某行出现在超过该比例的页面上即判为页眉页脚")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"找不到文件: {pdf_path}")
        return 1

    pages = extract_pages(pdf_path)
    print(f"共 {len(pages)} 页")

    furniture = detect_furniture(pages, args.furniture_ratio)
    print(f"识别出 {len(furniture)} 行版面元素（页眉/页脚/版权声明）")
    for line in sorted(furniture)[:12]:
        print(f"    {line[:78]}")

    header_prefix = detect_running_header(pages, args.furniture_ratio)
    print(f"书眉前缀: {header_prefix!r}")

    toc, toc_pages = parse_toc(pages)
    print(f"目录页: {sorted(p + 1 for p in toc_pages)}，解析出 {len(toc)} 个章节")

    title = args.title or pdf_path.stem.replace("_", " ")
    markdown = build_markdown(pages, title, toc, toc_pages, furniture, header_prefix)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    heading_count = sum(1 for line in markdown.split("\n") if line.startswith("#"))
    print(f"\n已写入 {out_path}")
    print(f"  {len(markdown)} 字符，{heading_count} 个标题，"
          f"{markdown.count('```') // 2} 个代码块")
    return 0


if __name__ == "__main__":
    sys.exit(main())
