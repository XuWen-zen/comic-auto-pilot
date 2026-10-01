# -*- coding: utf-8 -*-
"""
================================================================================
 Comic-Auto-Pilot  ——  四格条漫全自动本地极速制片厂  v2.0 (完全体)
================================================================================

 100% 本地物理运行 / 创意资产零外泄 / 不依赖任何公网 API 路由。

 ┌──────────────────────────── 全局业务流程通路 ────────────────────────────┐
 │  [Gradio 前端]  一句话纯中文盲打剧情                                      │
 │        ↓                                                                 │
 │  [Ollama qwen2.5:14b]  system / prompt 物理分离 + format:json 强约束      │
 │        ↓                                                                 │
 │  [净化层]  <think> 推理链剥离 → 括号配平扫描 → 结构/字段完备性验证         │
 │        ↓                                                                 │
 │  [钢印层]  4 格英文 Prompt 头部强制刻入官方召唤词与负向整容密码            │
 │        ↓                                                                 │
 │  [ComfyUI 对接层]  读取本地 workflow_api.json → 动态扫描节点 → 注入文本    │
 │        ↓                                                                 │
 │  [ComfyUI 后端 :8188]  KSampler 抽卡 → 4x-UltraSharp 重构 → Grid 焊接    │
 │        ↓                           → Text Overlay 打气泡                 │
 │  [Gradio 前端]  吐出 1x4 成品长图 + 诚实日志                              │
 └──────────────────────────────────────────────────────────────────────────┘

 架构硬性原则（反"假成功"熔断）：
   1. 任何一步失败 → 立刻抛错 + 打印 debug 日志，gr.Image 一律返回 None。
   2. 绝不虚构图片路径、绝不使用占位图假装渲染完成。
   3. 图片只有在 ComfyUI /history 真正回传产物、且字节流成功解码后才会出现。

 运行前置条件：
   * Ollama 已启动并已 pull 模型：  ollama pull qwen2.5:14b
   * ComfyUI 已以 API 模式静默启动：python main.py --port 8188
   * 本目录下存在 workflow_api.json（ComfyUI 菜单 → Save (API Format) 导出）
     若该文件不存在，本程序依然可正常跑通剧本生成，并诚实提示"模块待对接"。

 启动：  python main.py     →  浏览器打开 http://127.0.0.1:7860
================================================================================
"""

import gradio as gr
import requests
import json
import re
import os
import io
import copy
import time
import random
import uuid
import traceback
from datetime import datetime
from pathlib import Path

# Pillow 负责两件事：① 解码 ComfyUI 回传的成品字节流 ② 在本地绘制中文对话气泡
try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    _PIL_OK = True
except Exception:                                    # pragma: no cover
    _PIL_OK = False


# ==============================================================================
#  SECTION 0 ── 全局物理配置（唯一需要用户按本机情况修改的区域）
# ==============================================================================
#  【环境架构约定 —— 开源协作者必读】
#  本项目与 ComfyUI 是两个完全独立的 Python 环境，彼此只通过 HTTP 通信：
#
#      comic-auto-pilot/venv          C:\ai\ComfyUI\venv
#      ├─ gradio                      ├─ torch + cu12x / xformers
#      ├─ requests          ──HTTP──▶ ├─ 各类 custom_nodes 依赖
#      └─ pillow            :8188     └─ ComfyUI 本体
#
#  ⚠️ 切勿把两者合并装入 global / pyenv 全局环境：torch 的版本锁会与 gradio
#     依赖树互相撕咬，且会让 clone 本仓库的人被迫先装 8GB 推理依赖。
#
#  下列所有配置均支持环境变量覆盖（无需改动源码即可适配别人的机器）：
#      CAP_OLLAMA_HOST / CAP_OLLAMA_MODEL / CAP_COMFY_HOST / CAP_WORKFLOW / CAP_OUTPUT_DIR
# ==============================================================================

def _env(key: str, default: str) -> str:
    """读取环境变量覆盖项；为空则回落到默认值。"""
    val = os.environ.get(key, "").strip()
    return val or default


# ---- 文本推理大脑：本地 Ollama ------------------------------------------------
OLLAMA_HOST       = _env("CAP_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_GENERATE   = f"{OLLAMA_HOST}/api/generate"    # 原代码此处被截断为 http://127.0.0 —— 已修复
OLLAMA_TAGS       = f"{OLLAMA_HOST}/api/tags"
OLLAMA_MODEL      = _env("CAP_OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_TIMEOUT    = 300                              # 秒，14B 甜点尺寸首次加载进显存较慢
OLLAMA_OPTIONS    = {
    "temperature": 0.75,                             # 略带创意但不发癫
    "top_p": 0.9,
    "num_predict": 1536,                             # 四格 JSON 足够；过小会导致截断
    "repeat_penalty": 1.05,
}
OLLAMA_KEEP_ALIVE = "10m"                            # 模型常驻显存，避免每次重新加载

# ---- 视觉渲染总监：后台静默运行的 ComfyUI ------------------------------------
_HERE             = Path(__file__).resolve().parent
COMFY_HOST        = _env("CAP_COMFY_HOST", "http://127.0.0.1:8188").rstrip("/")
COMFY_WORKFLOW    = Path(_env("CAP_WORKFLOW", str(_HERE / "workflow_api.json")))
COMFY_POLL_SEC    = 1.5                              # /history 轮询间隔
COMFY_RENDER_TMO  = 900                              # 单次 4 格 + 4K 重构总超时（秒）
COMFY_HTTP_TMO    = 30                               # 单次 HTTP 请求超时（秒）

# ---- 成品落盘目录 -------------------------------------------------------------
OUTPUT_DIR        = Path(_env("CAP_OUTPUT_DIR", str(_HERE / "outputs")))

# ---- 黄金触发词钢印（雷打不动刻在每格 Prompt 最开头） -----------------------
#  采用「蓝色大肥鱼女仆」LoRA 的官方四段式正向语法，分段含义：
#    ① 画质密码      ② LoRA 召唤词      ③ 人物外观      ④ 女仆服装      ⑤ 围裙鲸鱼徽记
#  ⚠️ 此处【不要】再加 (derp eyes, blank stare:1.4) 之类的"呆滞眼神"权重词 ——
#     实测会把角色眼睛压成死鱼眼，官方语法用 blue eyes 才是正确写法。
QUALITY_PREFIX = "masterpiece, best quality, very aesthetic, newest,"
CHARACTER_BLOCK = (
    "deepseek_whale_girl, deepseek_maid_outfit, "
    "blue hair, very long hair, wavy hair, gradient blue hair, bangs, ahoge, blue eyes, "
    "whale fin ears, blue hair bows, "
    "maid, maid headdress, navy dress, white frilled apron, gold trim, "
    "dark blue neck bow, blue gemstone brooch, puff sleeves, long sleeves, short dress, "
    "white thighhighs, tail, starry tail pattern, "
    "blue whale emblem on lower apron,"
)
GOLDEN_TRIGGER = f"{QUALITY_PREFIX} {CHARACTER_BLOCK}"

# ---- 空镜 / 切镜格的豁免机制 -------------------------------------------------
#  四格漫画里常有「镜头切走」的分格：电脑屏幕特写、报错弹窗、纯场景空镜……
#  这类格子如果照样刻入角色钢印，蓝发女仆会被硬塞进屏幕里，整个梗就废了。
#  因此支持在该格提示词开头写 [NOCHAR]（或「空镜」），程序会：
#    · 跳过角色钢印，只保留画质词
#    · 追加 no humans 负向约束，防止角色偷偷乱入
NOCHAR_MARKS = ("[nochar]", "[no char]", "[空镜]", "[切镜]", "空镜：", "空镜:")
CUTAWAY_TRIGGER = f"{QUALITY_PREFIX} no humans, no people,"

# 用于判定"钢印是否已存在"的核心指纹，避免模型自己写了一半导致重复拼接。
TRIGGER_FINGERPRINT = "deepseek_whale_girl"

# 钢印自身的全部词条（由上面两种钢印自动推导，改钢印无需同步改别处）。
# 用于把模型自写的半截钢印物理剔除，避免同一词条重复堆叠稀释权重。
_TRIGGER_TOKENS = {t.strip().lower()
                   for t in (GOLDEN_TRIGGER + " " + CUTAWAY_TRIGGER).split(",")
                   if t.strip()}

# 画面描述里一律禁止出现的"呆滞眼神"类词条 —— 这类词会直接毁掉角色眼神。
_BANNED_TOKENS = (
    "derp eyes", "blank stare", "dead fish eyes", "dull eyes",
    "empty eyes", "vacant stare", "expressionless",
)

# ---- 前端底模下拉框 → ComfyUI CheckpointLoader 实际 ckpt 文件名映射 ----------
# 值为 None 表示"不改动工作流里用户手搓的底模"。请按本机 models/checkpoints 下真实文件名填写。
CHECKPOINT_MAP = {
    "Illustrious-XL 动漫底模 (推荐)": None,          # 默认沿用 workflow_api.json 里的原始底模
    "Flux.1 Dev 写实底模": None,                     # 填成 "flux1-dev.safetensors" 即可真正生效
}

# ---- 采样器行为开关 -----------------------------------------------------------
RANDOMIZE_SEED        = True                         # 每次出图重新抽卡（否则永远同一张）
ENFORCE_SAMPLER_CFG   = False                        # True 则强制覆盖工作流的 CFG / scheduler
SAMPLER_CFG_RANGE     = (7.0, 8.0)                   # PRD 规定区间
SAMPLER_SCHEDULER     = "karras"

# ---- 对话气泡渲染策略 ---------------------------------------------------------
#  "local" = 由本进程用 Pillow 在成品长图上绘制（推荐，零自定义节点依赖，中文不出豆腐块）
#  "comfy" = 由 ComfyUI 工作流里的 Text Overlay 节点绘制（需自行安装相应自定义节点包）
BUBBLE_MODE           = _env("CAP_BUBBLE_MODE", "local").lower()

# 气泡固定坐标（PRD 规定 X:100 / Y:150，为每一格的格内局部坐标）
BUBBLE_X, BUBBLE_Y    = 100, 150
OVERRIDE_BUBBLE_XY    = False                        # 仅 comfy 模式生效：强行覆盖节点坐标

# ---- 本地气泡（BUBBLE_MODE="local"）的排版与画风参数 -------------------------
GUTTER_PX             = 20        # 格间纯白边宽度，必须与 build_workflow.py 的 GUTTER_PX 一致
OUTER_BORDER_PX       = 20        # 成品四周再补一圈纯白外边（ImageStitch 只管格间，不管外框）
BUBBLE_MAX_W_RATIO    = 0.52      # 气泡最大宽度占格宽比例
BUBBLE_FONT_DIVISOR   = 20        # 字号 = 格宽 / 本值（1024 宽 → 约 51px）
BUBBLE_FILL           = (255, 255, 255)
BUBBLE_OUTLINE        = (18, 18, 18)
BUBBLE_TEXT_COLOR     = (18, 18, 18)

# 中文字体候选链（按优先级尝试，全部失败则回落 PIL 默认位图字体并告警）
BUBBLE_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyhbd.ttc",      # 微软雅黑 Bold —— 漫画气泡首选
    r"C:\Windows\Fonts\msyh.ttc",        # 微软雅黑 Regular
    r"C:\Windows\Fonts\simhei.ttf",      # 黑体
    r"C:\Windows\Fonts\Deng.ttf",        # 等线
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",       # Linux
    "/System/Library/Fonts/Hiragino Sans GB.ttc",                # macOS
]

# ---- 单格精修用的中文预设词条（下拉框选中后追加到该格提示词尾部）-------------
#  目的：让不会写英文 Prompt 的用户也能精确调整表情/景别/氛围。
#  追加是纯文本操作，零显卡开销；想生效再点「重出这一格」。
NONE_OPTION = "（不追加）"

EXPRESSION_PRESETS = {
    NONE_OPTION: "",
    "惊讶": "surprised, wide eyes, open mouth",
    "脸红害羞": "blushing, embarrassed, shy, looking away",
    "开心大笑": "happy, laughing, smiling, closed eyes",
    "生气": "angry, frowning, puffed cheeks",
    "哭": "crying, tears, teary eyes",
    "慌张": "panicking, flustered, sweat drop",
    "得意": "smug, smirk, confident",
    "困惑": "confused, head tilt, raised eyebrow",
    "困倦": "sleepy, half-closed eyes, yawning",
    "委屈": "pouting, upset, watery eyes",
    "面无表情": "calm, neutral face, serious",
}

SHOT_PRESETS = {
    NONE_OPTION: "",
    "大特写（只有脸）": "extreme close-up, face focus",
    "特写": "close-up, upper body",
    "中景": "medium shot",
    "全身": "full body, wide shot",
    "背后视角": "from behind, back view",
    "俯视": "from above",
    "仰视": "from below",
    "侧脸": "profile view, from side",
    "过肩": "over the shoulder shot",
}

MOOD_PRESETS = {
    NONE_OPTION: "",
    "深夜暖灯": "night, dim room, warm lamp light",
    "清晨阳光": "morning, soft sunlight, window light",
    "戏剧强光": "dramatic lighting, strong shadows, backlighting",
    "漫画速度线": "emphasis lines, speed lines, comic effect",
    "慌张汗滴": "sweat drop, anime comedy effect",
    "黑白惊吓": "monochrome, shocked, horror comedy effect",
    "暖色回忆": "warm sepia tone, nostalgic",
    "冷色压抑": "cold blue tone, gloomy",
}

# ---- 字段容错策略 -------------------------------------------------------------
# PRD 4.2 要求每格"必须同时包含且仅包含 prompt 和 bubble"。
# False = 工业级容错：多余字段丢弃并告警（推荐，避免 14B 偶发多吐一个键就全盘熔断）。
# True  = 字面严格：出现多余字段直接熔断。
STRICT_PANEL_FIELDS   = False

PANEL_KEYS = ["panel_1", "panel_2", "panel_3", "panel_4"]


# ==============================================================================
#  SECTION 1 ── 日志收集器（控制台 + 前端双通道，日志即事实，不美化不虚构）
# ==============================================================================

class RunLogger:
    """把一次运行的全部日志同时打到控制台与 Gradio 前端文本框。"""

    def __init__(self):
        self.lines = []

    def log(self, msg: str = "") -> str:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}" if msg else ""
        print(line, flush=True)
        self.lines.append(line)
        return self.text()

    def raw(self, msg: str) -> str:
        """不带时间戳的原样输出（用于分隔线与剧本正文排版）。"""
        print(msg, flush=True)
        self.lines.append(msg)
        return self.text()

    def text(self) -> str:
        return "\n".join(self.lines)


# ==============================================================================
#  SECTION 2 ── JSON 工业级净化层（PRD 4.1：100% 防 json.loads 崩溃）
# ==============================================================================

# 覆盖 qwen / deepseek / 各类蒸馏模型常见的推理链标签
_REASON_TAG_RE = re.compile(
    r"<\s*(think|thinking|reasoning|thought|analysis|scratchpad)\s*>.*?"
    r"<\s*/\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_ORPHAN_OPEN_RE  = re.compile(r"<\s*(?:think|thinking|reasoning|thought|analysis|scratchpad)\s*>",
                              re.IGNORECASE)
_ORPHAN_CLOSE_RE = re.compile(r"<\s*/\s*(?:think|thinking|reasoning|thought|analysis|scratchpad)\s*>",
                              re.IGNORECASE)
_FENCE_RE        = re.compile(r"```[a-zA-Z0-9_+-]*", re.IGNORECASE)
_TRAILING_COMMA  = re.compile(r",\s*([}\]])")


def strip_reasoning_chain(raw: str) -> str:
    """
    物理剥离推理链碎碎念。三层防护：
      1. 剥离成对的 <think>...</think>（含其他同类标签，大小写与空格均容错）。
      2. 存在孤立 </think> 收尾标签 → 只保留最后一个收尾标签之后的内容。
      3. 残留的孤立标签本体直接抹掉（绝不让 '<' 污染后续括号扫描）。
    """
    text = raw or ""
    text = _REASON_TAG_RE.sub(" ", text)

    # 模型只吐了收尾标签（开头标签被 Ollama 模板吃掉）的经典场景
    tail = list(_ORPHAN_CLOSE_RE.finditer(text))
    if tail:
        text = text[tail[-1].end():]

    text = _ORPHAN_OPEN_RE.sub(" ", text)
    text = _ORPHAN_CLOSE_RE.sub(" ", text)
    text = _FENCE_RE.sub(" ", text)                  # Markdown 代码围栏
    return text.strip()


def _scan_balanced_objects(text: str):
    """
    字符串/转义安全的括号配平扫描器（生成器，按出现顺序产出每个 {...} 子串）。

    为什么不能只用 find("{") + rfind("}")：
      模型可能先吐一段说明再吐 JSON，或在 JSON 之后追加一句"希望你喜欢！"，
      甚至吐出两个 JSON。裸切最外层大括号会把中间的垃圾一起吞进来。
      本扫描器逐字符跟踪 in_string / escape / depth，只产出真正配平的块。

    额外的截断自愈：若 num_predict 把输出砍断（depth>0 收尾），
      自动补齐未闭合的引号与右大括号后再产出，抢救最后一格剧本。
    """
    depth, start = 0, -1
    in_str, esc = False, False

    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    yield text[start:i + 1]
                    start = -1

    if depth > 0 and start != -1:
        patched = text[start:]
        if in_str:
            patched += '"'
        yield patched + ("}" * depth)


def _try_loads(candidate: str):
    """单个候选串的多轮修复式解析。strict=False 容忍字符串内的裸控制字符。"""
    for attempt in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
        try:
            obj = json.loads(attempt, strict=False)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _looks_like_script(obj: dict) -> bool:
    """
    候选字典里是否出现了我们要的特征键（用于在多个 JSON 块中挑出真正有用的那个）。
    同时覆盖三类载荷：整份四格剧本、单格改写 {prompt,bubble}、词条补充 {tags}。
    """
    if not isinstance(obj, dict):
        return False
    if any(_panel_index(k) for k in obj.keys()):
        return True
    keys = {str(k).lower() for k in obj.keys()}
    if keys & {"prompt", "bubble", "tags"}:
        return True
    return bool(keys & {"panels", "comic", "script", "data", "result", "panel"})


def extract_json_object(raw: str, logger: RunLogger) -> dict:
    """
    从 Ollama 原始返回中榨出唯一合法的剧本字典。失败即抛错，绝不返回半成品。
    """
    cleaned = strip_reasoning_chain(raw)
    if not cleaned:
        raise ValueError("Ollama 返回内容在剥离推理链后为空。")

    candidates = list(_scan_balanced_objects(cleaned))
    if not candidates:
        raise ValueError("Ollama 返回内容中找不到任何配平的 JSON 大括号块。")

    fallback = None
    for cand in candidates:
        obj = _try_loads(cand)
        if obj is None:
            continue
        if _looks_like_script(obj):
            return obj
        fallback = fallback or obj

    if fallback is not None:
        logger.log("⚠️  未命中分镜特征键，退化采用首个可解析 JSON 块，交由结构验证层裁决。")
        return fallback

    raise ValueError(
        f"扫描到 {len(candidates)} 个大括号块，但无一能被解析为合法 JSON 字典。"
    )


# ==============================================================================
#  SECTION 3 ── 结构验证 + 字段归一 + 黄金触发词钢印（PRD 4.1 / 4.2）
# ==============================================================================

_PANEL_IDX_RE = re.compile(
    r"^\s*(?:panel|frame|格|第|分镜|p|#)?\s*[_\-\s#:]?\s*([1-4])\s*(?:格|st|nd|rd|th)?\s*$",
    re.IGNORECASE,
)
_PROMPT_ALIASES = ("prompt", "positive", "en", "english", "image_prompt",
                   "english_prompt", "sd_prompt", "描述", "提示词")
_BUBBLE_ALIASES = ("bubble", "speech_bubble", "dialogue", "dialog", "line",
                   "text", "speech", "台词", "对话", "气泡")
_UNWRAP_KEYS    = ("panels", "panel", "comic", "script", "data", "result",
                   "output", "storyboard", "分镜", "剧本")


def _panel_index(key) -> int:
    """把五花八门的键名（panel_1 / Panel 1 / 第1格 / "1"）归一成 1~4，不匹配返回 0。"""
    if isinstance(key, int):
        return key if 1 <= key <= 4 else 0
    m = _PANEL_IDX_RE.match(str(key))
    return int(m.group(1)) if m else 0


def _pick_alias(panel: dict, aliases) -> str:
    """按别名优先级取值；兼容模型把台词吐成 list 的情况。"""
    lowered = {str(k).strip().lower(): v for k, v in panel.items()}
    for alias in aliases:
        if alias in lowered:
            val = lowered[alias]
            if isinstance(val, (list, tuple)):
                val = " ".join(str(x) for x in val)
            if isinstance(val, (int, float)):
                val = str(val)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def normalize_and_validate(obj: dict, logger: RunLogger) -> dict:
    """
    PRD 4.2 熔断式验证：
      · 容器层自动解包（{"panels": {...}} / {"panels": [...]} 都能吃）
      · panel_1 ~ panel_4 四个主键必须全部存在
      · 每格必须同时含有非空 prompt 与 bubble
      · 多余字段按 STRICT_PANEL_FIELDS 决定告警丢弃或直接熔断
    返回严格规范化后的 {"panel_n": {"prompt":..., "bubble":...}} 字典。
    """
    node = obj
    # --- 容器解包：最多三层，防止模型套娃 ---------------------------------
    for _ in range(3):
        if isinstance(node, dict) and len(node) == 1:
            only_key = next(iter(node))
            if str(only_key).strip().lower() in _UNWRAP_KEYS:
                logger.log(f"⚠️  检测到外层容器键 '{only_key}'，自动解包。")
                node = node[only_key]
                continue
        break

    # --- 列表形态（[{...},{...},{...},{...}]）转标准字典 -------------------
    if isinstance(node, list):
        if len(node) < 4:
            raise ValueError(f"结构验证失败：剧本为列表形态但只有 {len(node)} 格，不足 4 格。")
        node = {f"panel_{i + 1}": node[i] for i in range(4)}
        logger.log("⚠️  检测到列表形态剧本，已按顺序映射为 panel_1~panel_4。")

    if not isinstance(node, dict):
        raise ValueError(f"结构验证失败：剧本根节点类型为 {type(node).__name__}，不是字典。")

    # --- 键名归一 ----------------------------------------------------------
    indexed = {}
    for key, val in node.items():
        idx = _panel_index(key)
        if idx and isinstance(val, dict):
            if idx in indexed:
                raise ValueError(f"结构验证失败：分镜序号 {idx} 重复出现（键名 '{key}'）。")
            indexed[idx] = val
            if str(key) != f"panel_{idx}":
                logger.log(f"⚠️  键名 '{key}' 已归一为 'panel_{idx}'。")

    # --- 四格存在性熔断 ----------------------------------------------------
    missing = [f"panel_{i}" for i in (1, 2, 3, 4) if i not in indexed]
    if missing:
        raise KeyError(
            f"结构验证失败：缺少必须的分镜主键 {missing}（实际拿到的键：{list(node.keys())}）"
        )

    # --- 字段完备性熔断 ----------------------------------------------------
    script = {}
    for i in (1, 2, 3, 4):
        panel = indexed[i]
        name = f"panel_{i}"

        extras = [k for k in panel.keys()
                  if str(k).strip().lower() not in _PROMPT_ALIASES + _BUBBLE_ALIASES]
        if extras:
            if STRICT_PANEL_FIELDS:
                raise KeyError(f"结构验证失败：'{name}' 含有非法多余字段 {extras}（严格模式）。")
            logger.log(f"⚠️  '{name}' 含多余字段 {extras}，已按容错策略丢弃。")

        prompt = _pick_alias(panel, _PROMPT_ALIASES)
        bubble = _pick_alias(panel, _BUBBLE_ALIASES)

        if not prompt:
            raise KeyError(f"结构验证失败：'{name}' 缺少非空的 'prompt' 字段。")

        # bubble 必须存在，但【允许为空字符串】：
        # 空镜头、切镜、纯场景特写等分格本来就没有台词，
        # 把"空台词"当成致命错误会把一整份好剧本白白熔断掉。
        has_bubble_key = any(str(k).strip().lower() in _BUBBLE_ALIASES for k in panel.keys())
        if not has_bubble_key:
            raise KeyError(f"结构验证失败：'{name}' 缺少 'bubble' 字段。")
        if not bubble:
            logger.log(f"🔇 {name}: 台词为空，按【无台词静默格】处理（不画气泡）。")

        script[name] = {"prompt": prompt, "bubble": bubble}

    return script


def _token_core(token: str) -> str:
    """
    取出一个 Prompt 词条的"裸词"用于比对：
    剥掉 (xxx:1.4) 的权重语法、[]{} 等强调符号与首尾空白。
    """
    t = token.strip()
    t = re.sub(r":\s*[\d.]+\s*$", "", t)          # 尾部权重数值
    t = t.strip("()[]{}<> \t")
    return t.strip().lower()


def stamp_golden_trigger(script: dict, logger: RunLogger) -> dict:
    """
    黄金触发词钢印。不信任大模型的自觉性 —— 由 Python 在出图前做最后一道物理强制。

    逐词条过筛（而不是正则硬切字符串），好处是改 GOLDEN_TRIGGER 时这里零改动：
      ① 模型自写的钢印词条（命中 _TRIGGER_TOKENS）→ 剔除，由标准钢印统一前置
      ② 死鱼眼类违禁词（命中 _BANNED_TOKENS）     → 剔除，并记入日志
      ③ 剩下的才是真正的画面描述（构图/景别/动作/表情/场景/光照）
    最终 prompt = 标准钢印 + 纯净画面描述，既 100% 命中官方召唤词，又零重复堆叠。
    """
    for name in PANEL_KEYS:
        raw_body = script[name]["prompt"].strip()

        # ---- 空镜豁免：该格是否声明了不要角色 ----
        cutaway = False
        low = raw_body.lower()
        for mark in NOCHAR_MARKS:
            if low.startswith(mark):
                cutaway = True
                raw_body = raw_body[len(mark):].strip().lstrip(",").strip()
                break
        if not cutaway and raw_body.startswith(CUTAWAY_TRIGGER):
            cutaway = True                       # 单格重出时再次流经本函数

        trigger = CUTAWAY_TRIGGER if cutaway else GOLDEN_TRIGGER

        # 已经盖过章的 prompt（单格重出时会再次流经本函数）不必刷"模型自写"告警
        restamp = raw_body.startswith(trigger)

        kept, dropped_trigger, dropped_banned = [], 0, []
        for token in raw_body.split(","):
            if not token.strip():
                continue
            core = _token_core(token)
            if not core:
                continue
            if core in _TRIGGER_TOKENS:
                dropped_trigger += 1
                continue
            if any(bad in core for bad in _BANNED_TOKENS):
                dropped_banned.append(core)
                continue
            kept.append(token.strip())

        if dropped_trigger and not restamp:
            logger.log(f"🔧 {name}: 剔除模型自写的 {dropped_trigger} 个钢印词条，换装标准官方钢印。")
        if dropped_banned:
            logger.log(f"🚫 {name}: 拦截死鱼眼类违禁词 {dropped_banned}（会毁掉角色眼神）。")

        if cutaway:
            logger.log(
                f"🎬 {name}: 空镜格 —— 跳过角色钢印，只保画质词并追加 no humans 约束。")

        body = ", ".join(kept)
        script[name]["prompt"] = f"{trigger} {body}" if body else trigger

    return script


# ==============================================================================
#  SECTION 4 ── 文本推理大脑：Ollama 客户端（system / prompt 物理分离）
# ==============================================================================

SYSTEM_PROMPT = (
    "你是一个专为二次元四格漫画设计的黄金编剧。你的任务是：接收用户输入的一句简短中文剧情，"
    "将其精确拆解为 1x4 的传统四格漫画分镜（起-承-转-合），"
    "并自动为每格画面生成最地道的英文提示词（Prompt）与中文对话气泡（Speech Bubble）。\n"
    "\n"
    "【英文提示词硬性要求】\n"
    "角色的外观与服装词条（蓝发、ahoge、鲸鱼鳍耳、女仆装等）由程序自动前置，"
    "你【不要】重复书写这些人物设定词条。\n"
    "你只需要写该格画面本身的：构图、景别（close-up / medium shot / wide shot）、"
    "动作、表情、场景、道具、光照、氛围。\n"
    "描述词之间用英文逗号分隔，禁止写成完整句子，每格 6~12 个词条为宜。\n"
    "⚠️ 严禁出现 derp eyes / blank stare / dead fish eyes / expressionless 等呆滞眼神词条，"
    "角色眼神必须是生动的，表情请用 surprised / panicking / smiling / embarrassed 这类情绪词。\n"
    "\n"
    "【镜头切走的空镜格】\n"
    "如果某一格是「镜头切走」的画面 —— 例如电脑屏幕特写、报错弹窗、纯场景空镜、"
    "特写道具 —— 即画面里【不应该出现角色本人】，\n"
    "那么该格的 prompt 必须以 [NOCHAR] 开头，然后才写画面描述。\n"
    "例：\"[NOCHAR] close-up of computer monitor, green code scrolling, black background\"\n"
    "\n"
    "【中文气泡硬性要求】\n"
    "bubble 必须是简短口语化中文台词，每格不超过 20 个汉字，不要加引号或说话人名字。\n"
    "如果某一格本来就没有人说话（空镜、切镜、纯场景），bubble 请留空字符串 \"\"，"
    "不要硬编台词。\n"
    "\n"
    "【输出格式硬性要求】\n"
    "只输出纯净 JSON 字典，不要任何解释、前言、总结、Markdown 代码围栏或推理过程。\n"
    "必须严格包含 panel_1 到 panel_4 四个主键，每个主键内部必须且仅能包含 prompt 与 bubble 两个字段：\n"
    "{\n"
    "  \"panel_1\": {\"prompt\": \"...\", \"bubble\": \"...\"},\n"
    "  \"panel_2\": {\"prompt\": \"...\", \"bubble\": \"...\"},\n"
    "  \"panel_3\": {\"prompt\": \"...\", \"bubble\": \"...\"},\n"
    "  \"panel_4\": {\"prompt\": \"...\", \"bubble\": \"...\"}\n"
    "}"
)


def list_ollama_models() -> list:
    """
    拉取本地 Ollama 已安装的模型清单，供前端下拉框选择。
    失败时返回空列表（由调用方决定回落策略），绝不抛错打断界面加载。
    """
    try:
        resp = requests.get(OLLAMA_TAGS, timeout=5)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
    except Exception:
        return []
    # 把默认模型顶到第一位，其余按字母序，方便用户一眼看到推荐项
    names.sort()
    if OLLAMA_MODEL in names:
        names.remove(OLLAMA_MODEL)
        names.insert(0, OLLAMA_MODEL)
    return names


def ollama_preflight(logger: RunLogger, model: str) -> None:
    """开火前的轻量体检：Ollama 活着吗？所选模型 pull 过吗？——提前给出人话级报错。"""
    try:
        resp = requests.get(OLLAMA_TAGS, timeout=5)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
    except requests.exceptions.RequestException as exc:
        raise ConnectionError(
            f"无法连接本地 Ollama（{OLLAMA_HOST}）。请先启动 Ollama 服务。底层报错：{exc}"
        ) from exc

    base = str(model).split(":")[0]
    if not any(n == model or n.startswith(base) for n in names):
        raise ValueError(
            f"本地 Ollama 未安装模型 '{model}'。请先执行：ollama pull {model}\n"
            f"当前已安装：{names or '（空）'}"
        )
    logger.log(f"✅ Ollama 体检通过，模型 '{model}' 已就位。")


def call_ollama(user_input: str, logger: RunLogger, model: str,
                system: str | None = None) -> str:
    """
    呼叫本地 Ollama。严格把 system_prompt 与 user_input 塞进两个独立字段，
    绝不拼成一坨字符串 —— 拼接会直接污染 chat template 导致推理错乱。
    system 缺省为四格剧本提示词；单格改写 / 词条补充等小任务可传入自己的 system。
    """
    payload = {
        "model": model,
        "system": system or SYSTEM_PROMPT,   # ← 独立字段
        "prompt": user_input,                # ← 独立字段
        "stream": False,
        "format": "json",                 # 原生 JSON 强约束
        "options": OLLAMA_OPTIONS,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }

    t0 = time.time()
    try:
        resp = requests.post(OLLAMA_GENERATE, json=payload, timeout=OLLAMA_TIMEOUT)
    except requests.exceptions.Timeout as exc:
        raise TimeoutError(
            f"Ollama 推理超时（>{OLLAMA_TIMEOUT}s）。14B 模型首次加载进显存较慢，可重试或调大 OLLAMA_TIMEOUT。"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ConnectionError(f"Ollama 请求失败：{exc}") from exc

    if resp.status_code != 200:
        raise ConnectionError(
            f"Ollama 返回异常 HTTP 状态码 {resp.status_code}。响应体：{resp.text[:500]}"
        )

    try:
        envelope = resp.json()
    except ValueError as exc:
        raise ValueError(f"Ollama 响应外层不是合法 JSON：{resp.text[:500]}") from exc

    raw = envelope.get("response")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Ollama 响应中 'response' 字段为空。完整外层：{envelope}")

    logger.log(f"✅ Ollama 推理完成，耗时 {time.time() - t0:.1f}s，原始长度 {len(raw)} 字符。")
    return raw


# ------------------------------------------------------------------------------
#  单格级的 Ollama 小任务：改写某一格 / 给某一格补充画面词条
# ------------------------------------------------------------------------------
#  设计动机：用户不一定会写英文 Prompt。与其让他对着一长串英文发愁，
#  不如把"重写这一格"和"再补点细节"也交给本地 14B —— 仍然 100% 本地，无需联网。
# ------------------------------------------------------------------------------

_REWRITE_SYSTEM = (
    "你是二次元四格漫画的分镜编剧。用户会给你整个四格故事的上下文，"
    "以及需要你【重新设计】的那一格。\n"
    "\n"
    "请只为那一格重新设计画面与台词，输出纯净 JSON：\n"
    "{\"prompt\": \"...\", \"bubble\": \"...\"}\n"
    "\n"
    "【prompt 要求】\n"
    "· 只写该格画面本身：构图、景别（close-up / medium shot / wide shot）、动作、"
    "表情、场景、道具、光照、氛围。\n"
    "· 角色的外观与服装词条由程序自动前置，你【不要】书写蓝发、女仆装之类的人物设定词。\n"
    "· 用英文逗号分隔的词条，禁止写成句子，6~12 个词条为宜。\n"
    "· 严禁 derp eyes / blank stare / dead fish eyes / expressionless 等呆滞眼神词。\n"
    "· 如果这一格是镜头切走的空镜（屏幕特写、报错弹窗、纯场景），"
    "prompt 必须以 [NOCHAR] 开头。\n"
    "\n"
    "【bubble 要求】\n"
    "· 简短口语化中文台词，不超过 20 个汉字，不要引号和说话人名字。\n"
    "· 该格若本来没人说话，bubble 留空字符串 \"\"。\n"
    "\n"
    "必须与故事上下文衔接，且要和原来那一格【明显不同】，给出新的演绎。"
)

_EXTEND_SYSTEM = (
    "你是 Stable Diffusion 二次元提示词专家。用户会给你一格漫画现有的英文提示词。\n"
    "请补充 4~8 个能让画面更生动具体的英文词条（表情、动作细节、道具、光照、氛围、构图）。\n"
    "\n"
    "硬性要求：\n"
    "· 只输出【新增】词条，不要重复用户已有的任何词条。\n"
    "· 不要写角色外观/服装词条（蓝发、女仆装等由程序管理）。\n"
    "· 严禁 derp eyes / blank stare / dead fish eyes / expressionless 等呆滞眼神词。\n"
    "· 输出纯净 JSON：{\"tags\": \"tag1, tag2, tag3\"}"
)


def _ollama_json(system: str, user: str, model: str, logger: RunLogger) -> dict:
    """跑一个输出 JSON 的小任务，复用主链路那套工业级净化层（<think> 剥离 / 括号配平）。"""
    raw = call_ollama(user, logger, model, system)
    return extract_json_object(raw, logger)


def rewrite_panel_via_ollama(idea: str, script: dict, idx: int,
                             model: str, logger: RunLogger):
    """让 Ollama 重新设计某一格的画面与台词。返回 (prompt, bubble)。"""
    others = "\n".join(
        f"  第{i}格：{script[f'panel_{i}']['bubble'] or '（无台词）'}"
        for i in (1, 2, 3, 4) if i != idx
    )
    old = script[f"panel_{idx}"]
    # 把钢印剥掉再给模型看，否则它会被那一长串人设词带跑
    old_body = old["prompt"]
    for trig in (GOLDEN_TRIGGER, CUTAWAY_TRIGGER):
        if old_body.startswith(trig):
            old_body = old_body[len(trig):].strip()
            break

    user = (
        f"整个四格故事：{idea or '（未提供）'}\n"
        f"其余三格的台词：\n{others}\n\n"
        f"需要你重新设计的是【第 {idx} 格】。\n"
        f"它原来的画面词条是：{old_body or '（空）'}\n"
        f"它原来的台词是：{old['bubble'] or '（无台词）'}\n\n"
        f"请重新设计第 {idx} 格。"
    )

    obj = _ollama_json(_REWRITE_SYSTEM, user, model, logger)

    # 容错：模型可能把结果再套一层 {"panel_3": {...}}
    if "prompt" not in obj:
        for v in obj.values():
            if isinstance(v, dict) and "prompt" in v:
                obj = v
                break

    prompt = str(obj.get("prompt") or "").strip()
    if not prompt:
        raise ValueError(f"Ollama 未返回可用的 prompt 字段。实际返回：{obj}")
    bubble = obj.get("bubble")
    bubble = "" if bubble is None else str(bubble).strip()
    return prompt, bubble


def extend_prompt_via_ollama(current_prompt: str, model: str, logger: RunLogger) -> str:
    """让 Ollama 为现有提示词补充若干画面词条，返回追加用的词条串。"""
    body = current_prompt
    for trig in (GOLDEN_TRIGGER, CUTAWAY_TRIGGER):
        if body.startswith(trig):
            body = body[len(trig):].strip()
            break

    obj = _ollama_json(_EXTEND_SYSTEM, f"现有提示词：{body or '（空）'}", model, logger)
    tags = obj.get("tags")
    if isinstance(tags, (list, tuple)):
        tags = ", ".join(str(t) for t in tags)
    tags = str(tags or "").strip().strip(",").strip()
    if not tags:
        raise ValueError(f"Ollama 未返回可用的 tags 字段。实际返回：{obj}")
    return tags


# ==============================================================================
#  SECTION 5 ── 视觉渲染总监：ComfyUI API 图纸对齐与渲染（PRD 4.3）
# ==============================================================================
#  核心设计哲学：【停止盲猜节点 ID】
#  节点 ID 完全由用户本机手搓的工作流决定，硬编码必然对不上。
#  因此本层通过"拓扑溯源 + 标题锚定 + 类型兜底"三重策略动态扫描图纸：
#     1. 拓扑溯源（最可靠）：找出全部 KSampler，沿其 positive 链路反向走，
#        定位到真正喂正向提示词的 CLIPTextEncode —— 天然排除负面提示词节点。
#     2. 标题锚定：若用户在节点标题里写了 panel_1 / 第1格，直接按标题对号入座。
#     3. 类型兜底：上述都拿不到时，按 class_type 过滤 + 节点 ID 排序取前 4。
# ==============================================================================

_SAMPLER_HINTS   = ("ksampler", "sampler")
_CLIP_HINTS      = ("cliptextencode", "clip_text_encode", "textencode")
_OVERLAY_HINTS   = ("overlay", "drawtext", "text_overlay", "addtext", "texttoimage",
                    "textbox", "caption")
_SAVE_HINTS      = ("saveimage", "previewimage", "image_save", "imagesave")
_TEXT_FIELDS     = ("text", "caption", "bubble", "bubble_text", "text_1", "string",
                    "value", "prompt")
_NEGATIVE_WORDS  = ("worst quality", "low quality", "lowres", "bad anatomy",
                    "bad hands", "nsfw", "watermark", "jpeg artifacts", "embedding:")


class ComfyWorkflowError(RuntimeError):
    """ComfyUI 图纸解析 / 渲染链路上的所有可诊断故障。"""


class ComfyPilot:
    """ComfyUI API 控制端：读图纸 → 扫节点 → 注文本 → 投任务 → 轮询 → 取成品。"""

    def __init__(self, host: str, workflow_path: Path, logger: RunLogger):
        self.host = host.rstrip("/")
        self.workflow_path = Path(workflow_path)
        self.logger = logger
        self.client_id = str(uuid.uuid4())
        self.graph = {}
        self.prompt_nodes = {}      # {panel_idx: node_id}  正向提示词落点
        self.bubble_nodes = {}      # {panel_idx: (node_id, field)}  气泡文本落点
        self.save_nodes = []        # 产物输出节点（优先 1x4 拼接图所在节点）

    # ---------------------------------------------------------------- 基础 HTTP
    def _url(self, path: str) -> str:
        return f"{self.host}/{path.lstrip('/')}"

    def ping(self) -> None:
        """确认后台 ComfyUI 真的活着（而不是等渲染时才发现端口没开）。"""
        try:
            resp = requests.get(self._url("/system_stats"), timeout=5)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise ComfyWorkflowError(
                f"无法连接本地 ComfyUI（{self.host}）。请确认后台已以 API 模式启动"
                f"（python main.py --port 8188）。底层报错：{exc}"
            ) from exc
        self.logger.log(f"✅ ComfyUI 后端在线：{self.host}")

    # ------------------------------------------------------------ 图纸读取解析
    def load_workflow(self) -> dict:
        """读取并校验 workflow_api.json，返回 {node_id: node} 原始图纸字典。"""
        if not self.workflow_path.is_file():
            raise FileNotFoundError(
                f"找不到物理工作流图纸：{self.workflow_path}\n"
                f"请在 ComfyUI 里手动拼好完美工作流后，用菜单 "
                f"[Workflow → Export (API)] / [Save (API Format)] 导出为 workflow_api.json "
                f"并放到本 main.py 同级目录。"
            )
        try:
            with open(self.workflow_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ComfyWorkflowError(f"workflow_api.json 不是合法 JSON：{exc}") from exc

        # 用户误把 UI 格式（含 "nodes" 数组）当 API 格式导出 —— 这是最高频踩坑
        if isinstance(data, dict) and "nodes" in data and isinstance(data.get("nodes"), list):
            raise ComfyWorkflowError(
                "检测到你导出的是【UI 格式】工作流（含 nodes 数组），本程序需要【API 格式】。"
                "请在 ComfyUI 中使用 Export (API) / Save (API Format) 重新导出。"
            )
        if not isinstance(data, dict) or not data:
            raise ComfyWorkflowError("workflow_api.json 顶层结构异常：应为非空的节点字典。")

        valid = {k: v for k, v in data.items()
                 if isinstance(v, dict) and "class_type" in v}
        if not valid:
            raise ComfyWorkflowError("workflow_api.json 中没有任何带 class_type 的合法节点。")

        self.graph = data
        self.logger.log(f"📐 图纸载入成功：{self.workflow_path.name}，共 {len(valid)} 个节点。")
        return data

    # ------------------------------------------------------------- 节点扫描工具
    @staticmethod
    def _norm(s) -> str:
        return re.sub(r"[\s_\-()]", "", str(s or "")).lower()

    def _title(self, node_id: str) -> str:
        meta = self.graph.get(node_id, {}).get("_meta") or {}
        return str(meta.get("title", "") or "")

    def _class_of(self, node_id: str) -> str:
        return str(self.graph.get(node_id, {}).get("class_type", "") or "")

    def _inputs_of(self, node_id: str) -> dict:
        ins = self.graph.get(node_id, {}).get("inputs")
        return ins if isinstance(ins, dict) else {}

    def _panel_of_node(self, node_id: str) -> int:
        """从节点标题里抽取分镜序号（支持 panel_1 / Panel 2 / 第3格 / P4）。"""
        title = self._title(node_id)
        m = re.search(r"(?:panel|frame|p|第|格)\s*[_\-#:]?\s*([1-4])", title, re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"\b([1-4])\b", title)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _id_sort_key(node_id: str):
        """节点 ID 多为数字字符串，按数值排序才符合用户搭图的视觉顺序。"""
        try:
            return (0, int(node_id))
        except (TypeError, ValueError):
            return (1, str(node_id))

    def _is_text_encoder(self, node_id: str) -> bool:
        cls = self._norm(self._class_of(node_id))
        if any(h in cls for h in _CLIP_HINTS):
            return True
        # 兼容第三方文本编码节点：有 CLIP 输入且有字符串 text 输入
        ins = self._inputs_of(node_id)
        return "clip" in ins and isinstance(ins.get("text"), str)

    def _resolve_conditioning_source(self, link, depth: int = 0) -> str:
        """
        沿 conditioning 链路反向溯源，穿透 ControlNetApply / ConditioningCombine /
        ConditioningSetArea 等中间节点，找到最终的文本编码节点 ID。
        """
        if depth > 12 or not isinstance(link, (list, tuple)) or not link:
            return ""
        node_id = str(link[0])
        if node_id not in self.graph:
            return ""
        if self._is_text_encoder(node_id):
            return node_id

        # 继续往上游走：优先 conditioning / positive 命名的入口
        ins = self._inputs_of(node_id)
        ordered = sorted(ins.items(),
                         key=lambda kv: 0 if self._norm(kv[0]) in
                         ("conditioning", "positive", "conditioningto", "conditioning1")
                         else 1)
        for _, val in ordered:
            if isinstance(val, (list, tuple)) and val:
                found = self._resolve_conditioning_source(val, depth + 1)
                if found:
                    return found
        return ""

    # ------------------------------------------- 策略一/二/三：定位正向提示词节点
    def discover_prompt_nodes(self) -> dict:
        samplers = [nid for nid in self.graph
                    if any(h in self._norm(self._class_of(nid)) for h in _SAMPLER_HINTS)
                    or "positive" in self._inputs_of(nid)]
        samplers.sort(key=self._id_sort_key)

        positives, negatives = [], set()
        for sid in samplers:
            ins = self._inputs_of(sid)
            pos = self._resolve_conditioning_source(ins.get("positive"))
            neg = self._resolve_conditioning_source(ins.get("negative"))
            if neg:
                negatives.add(neg)
            if pos and pos not in positives:
                positives.append((sid, pos))

        mapping = {}

        # 策略二优先：标题里明确写了分镜序号，按标题对号入座（最符合用户意图）
        for sid, pos in positives:
            idx = self._panel_of_node(pos) or self._panel_of_node(sid)
            if idx and idx not in mapping:
                mapping[idx] = pos
        if len(mapping) == 4:
            self.logger.log("🔎 提示词节点定位：标题锚定 + 拓扑溯源 命中 4/4。")
            return mapping

        # 策略一：纯拓扑顺序（按 KSampler 的节点 ID 升序 = 用户从左到右搭的 4 条链）
        if len(positives) >= 4:
            mapping = {i + 1: positives[i][1] for i in range(4)}
            self.logger.log("🔎 提示词节点定位：KSampler positive 拓扑溯源命中 4/4。")
            return mapping

        # 策略三兜底：全图扫 CLIPTextEncode，排除负面提示词节点
        self.logger.log("⚠️  拓扑溯源不足 4 条链路，启用 class_type 兜底扫描。")
        encoders = [nid for nid in self.graph if self._is_text_encoder(nid)]
        encoders = [nid for nid in encoders if nid not in negatives]
        encoders = [nid for nid in encoders
                    if not any(w in str(self._inputs_of(nid).get("text", "")).lower()
                               for w in _NEGATIVE_WORDS)]
        encoders.sort(key=self._id_sort_key)

        titled = {}
        for nid in encoders:
            idx = self._panel_of_node(nid)
            if idx and idx not in titled:
                titled[idx] = nid
        if len(titled) == 4:
            self.logger.log("🔎 提示词节点定位：兜底扫描 + 标题锚定命中 4/4。")
            return titled

        if len(encoders) >= 4:
            self.logger.log("🔎 提示词节点定位：兜底扫描按节点 ID 顺序取前 4。")
            return {i + 1: encoders[i] for i in range(4)}

        raise ComfyWorkflowError(
            f"图纸对齐失败：只在 workflow_api.json 中找到 {len(encoders)} 个可用的正向"
            f"文本编码节点，需要 4 个（四格各一条链路）。\n"
            f"建议：在 ComfyUI 中把 4 个正向 CLIPTextEncode 节点标题分别改名为 "
            f"panel_1 / panel_2 / panel_3 / panel_4，本程序即可 100% 精准对齐。"
        )

    # ------------------------------------------------ 定位 Text Overlay 气泡节点
    def discover_bubble_nodes(self) -> dict:
        """
        返回 {panel_idx: (node_id, 文本字段名)}。
        气泡节点的 class_type 在各家自定义节点包里命名极度混乱
        （Text Overlay / ImageTextOverlay / Draw Text / Mikey TextOverlay ...），
        因此按"类名含 overlay/text 关键词 + 具备字符串文本输入端口"双重特征识别。
        """
        if BUBBLE_MODE != "comfy":
            self.logger.log(
                "💬 气泡策略：local —— 由本地 Pillow 在成品长图上绘制，"
                "跳过 ComfyUI 文字节点扫描。"
            )
            return {}

        prompt_ids = set(self.prompt_nodes.values())
        candidates = []

        for nid in self.graph:
            if nid in prompt_ids or self._is_text_encoder(nid):
                continue
            cls = self._norm(self._class_of(nid))
            ins = self._inputs_of(nid)

            field = next((f for f in _TEXT_FIELDS if isinstance(ins.get(f), str)), None)
            if not field:
                continue

            cls_hit = any(h in cls for h in _OVERLAY_HINTS)
            # 节点带图像输入 + 文本输入 ⇒ 极可能就是往图上写字的叠加节点
            img_hit = any(self._norm(k) in ("image", "images", "canvas") for k in ins)
            if cls_hit or img_hit:
                candidates.append((nid, field))

        candidates.sort(key=lambda t: self._id_sort_key(t[0]))

        mapping = {}
        for nid, field in candidates:
            idx = self._panel_of_node(nid)
            if idx and idx not in mapping:
                mapping[idx] = (nid, field)

        if len(mapping) == 4:
            self.logger.log("🔎 气泡节点定位：标题锚定命中 4/4。")
            return mapping

        if len(candidates) >= 4:
            self.logger.log("🔎 气泡节点定位：按节点 ID 顺序取前 4。")
            return {i + 1: candidates[i] for i in range(4)}

        # 气泡不是渲染的必要条件 —— 缺了只告警，不熔断整条出图通路
        self.logger.log(
            f"⚠️  只识别到 {len(candidates)} 个 Text Overlay 气泡节点（需要 4 个），"
            f"本次将跳过台词注入，图照出但没有对话气泡。"
            f"建议把 4 个 Text Overlay 节点标题改名为 panel_1~panel_4。"
        )
        return {i + 1: candidates[i] for i in range(len(candidates))}

    # ----------------------------------------------------------- 定位产物输出节点
    def discover_save_nodes(self) -> dict:
        """
        返回 {panel_idx: save_node_id}。
        四格各自独立落盘是「单格重出」能力的前提 —— 有了逐格输出节点，
        就能靠 ComfyUI 的输出驱动剪枝，只渲染用户点的那一格。
        """
        saves = [nid for nid in self.graph
                 if any(h in self._norm(self._class_of(nid)) for h in _SAVE_HINTS)]
        if not saves:
            raise ComfyWorkflowError(
                "图纸对齐失败：workflow_api.json 中找不到任何 SaveImage / PreviewImage 输出节点，"
                "ComfyUI 不会产出任何可取回的图片。"
            )
        saves.sort(key=self._id_sort_key)

        # 优先按标题里的 panel 序号对号入座
        mapping = {}
        for nid in saves:
            idx = self._panel_of_node(nid)
            if idx and idx not in mapping:
                mapping[idx] = nid

        if len(mapping) != 4:
            # 兜底：沿每个输出节点的 images 链路反查它属于哪一格
            mapping = {}
            for nid in saves:
                idx = self._panel_of_image_chain(self._inputs_of(nid).get("images"))
                if idx and idx not in mapping:
                    mapping[idx] = nid

        if len(mapping) != 4:
            if len(saves) >= 4:
                mapping = {i + 1: saves[i] for i in range(4)}
                self.logger.log("⚠️  输出节点无法按格识别，已按节点 ID 顺序强行对应前 4 个。")
            else:
                raise ComfyWorkflowError(
                    f"图纸对齐失败：只找到 {len(saves)} 个输出节点，而单格重出功能需要四格"
                    f"各自独立的 SaveImage。请重新运行：python build_workflow.py"
                )

        self.logger.log(f"🔎 逐格输出节点：{ {k: mapping[k] for k in sorted(mapping)} }")
        return mapping

    def _panel_of_image_chain(self, link, depth: int = 0) -> int:
        """沿 IMAGE 链路往上游走，用途是判定某个输出节点归属于哪一格。"""
        if depth > 12 or not isinstance(link, (list, tuple)) or not link:
            return 0
        nid = str(link[0])
        if nid not in self.graph:
            return 0
        idx = self._panel_of_node(nid)
        if idx:
            return idx
        for key, val in self._inputs_of(nid).items():
            if isinstance(val, (list, tuple)) and val and self._norm(key) in (
                    "image", "images", "samples", "latent"):
                found = self._panel_of_image_chain(val, depth + 1)
                if found:
                    return found
        return 0

    def align_blueprint(self) -> None:
        """一次性完成全图纸对齐，把 4 个提示词端口与 4 个输出端口全部锁定。"""
        self.load_workflow()
        self.prompt_nodes = self.discover_prompt_nodes()
        self.bubble_nodes = self.discover_bubble_nodes()
        self.save_nodes = self.discover_save_nodes()

        self.logger.raw("  ── 图纸端口对齐结果 ──")
        for i in (1, 2, 3, 4):
            pid = self.prompt_nodes.get(i, "—")
            sid = self.save_nodes.get(i, "—")
            self.logger.raw(
                f"   panel_{i}  →  Prompt 节点 #{pid} [{self._class_of(pid) or '?'}]"
                f"  →  输出节点 #{sid}"
            )

    # ------------------------------------------------------------ 放大链路旁路
    def _bypass_upscalers(self, wf: dict) -> int:
        """
        把所有 ImageUpscaleWithModel 节点从链路里"短路"掉：
        凡是吃它输出的下游节点，改为直接吃它自己的 image 上游。
        用于前端「高清重构」开关关闭时的快速出图（实测每格省 ~3.5s，四格省 ~24s）。
        """
        targets = {}
        for nid, node in wf.items():
            if "imageupscalewithmodel" in self._norm(node.get("class_type", "")):
                src = (node.get("inputs") or {}).get("image")
                if isinstance(src, (list, tuple)) and src:
                    targets[nid] = list(src)

        if not targets:
            return 0

        rewired = 0
        for node in wf.values():
            ins = node.get("inputs")
            if not isinstance(ins, dict):
                continue
            for key, val in list(ins.items()):
                if isinstance(val, (list, tuple)) and val and str(val[0]) in targets:
                    ins[key] = targets[str(val[0])]
                    rewired += 1
        return rewired

    # -------------------------------------------------------------- 文本动态注入
    def build_payload(self, script: dict, checkpoint: str | None,
                      panels=(1, 2, 3, 4), hires: bool = True, seeds: dict | None = None) -> dict:
        """
        深拷贝图纸并把剧本数据推送到已锁定的端口上。
        深拷贝是硬要求：绝不污染内存里的原始图纸，保证每次运行都是干净基线。

        panels：本次要渲染哪几格。只保留这几格的输出节点 ——
                ComfyUI 的执行是输出驱动的，删掉其余输出节点后它会自动剪枝上游，
                于是「只重出第 3 格」真的只跑第 3 格的 KSampler。
        hires ：False 时短路 4x-UltraSharp 放大链路。
        seeds ：{panel_idx: seed}，用于复现或指定某格的抽卡结果；缺省则随机。
        """
        wf = copy.deepcopy(self.graph)
        panels = tuple(sorted(set(int(p) for p in panels)))
        # 注意：必须保持调用方传入 dict 的【同一性】，抽到的 seed 要能回传给前端做复现。
        # 写成 `seeds = seeds or {}` 会在传入空字典时悄悄新建对象，把回传链路切断。
        if seeds is None:
            seeds = {}

        # ---- 注入四格文本（未被渲染的格也注入，保持图纸自描述一致）----------
        for i in (1, 2, 3, 4):
            name = f"panel_{i}"
            nid = self.prompt_nodes.get(i)
            if nid:
                wf[nid].setdefault("inputs", {})["text"] = script[name]["prompt"]

            pair = self.bubble_nodes.get(i)
            if pair:
                bid, bfield = pair
                wf[bid].setdefault("inputs", {})[bfield] = script[name]["bubble"]
                if OVERRIDE_BUBBLE_XY:
                    for axis, val in (("x", BUBBLE_X), ("y", BUBBLE_Y),
                                      ("x_offset", BUBBLE_X), ("y_offset", BUBBLE_Y)):
                        if axis in wf[bid]["inputs"]:
                            wf[bid]["inputs"][axis] = val

        # ---- 按格剪枝：删掉不需要的输出节点 --------------------------------
        if set(panels) != {1, 2, 3, 4}:
            for i, sid in self.save_nodes.items():
                if i not in panels and sid in wf:
                    del wf[sid]
            self.logger.log(
                f"✂️  本次只渲染 panel_{list(panels)}，已剪掉其余输出节点"
                f"（ComfyUI 会自动跳过对应的 KSampler 链路）。"
            )

        # ---- 逐格指定/随机 seed --------------------------------------------
        for i in panels:
            ksa = self._sampler_of_panel(wf, i)
            if not ksa:
                continue
            want = seeds.get(i)
            new_seed = int(want) if want else random.randint(0, 2 ** 53 - 1)
            for key in ("seed", "noise_seed"):
                if key in wf[ksa].get("inputs", {}):
                    wf[ksa]["inputs"][key] = new_seed
            seeds[i] = new_seed

        self.logger.log("🎲 本次抽卡 seed：" +
                        ", ".join(f"panel_{i}={seeds.get(i)}" for i in panels))

        # ---- 采样器参数 / 底模切换 -----------------------------------------
        for node in wf.values():
            ins = node.get("inputs")
            if not isinstance(ins, dict):
                continue
            if ENFORCE_SAMPLER_CFG:
                if isinstance(ins.get("cfg"), (int, float)):
                    ins["cfg"] = round(random.uniform(*SAMPLER_CFG_RANGE), 1)
                if isinstance(ins.get("scheduler"), str):
                    ins["scheduler"] = SAMPLER_SCHEDULER
            if checkpoint and isinstance(ins.get("ckpt_name"), str):
                ins["ckpt_name"] = checkpoint

        if checkpoint:
            self.logger.log(f"🎛️  底模已切换为：{checkpoint}")

        # ---- 高清重构开关 ---------------------------------------------------
        if not hires:
            n = self._bypass_upscalers(wf)
            self.logger.log(
                f"⚡ 高清重构已关闭：短路 4x-UltraSharp（改接 {n} 处连线），"
                f"出图更快但细节较软。"
            )

        return wf

    def _sampler_of_panel(self, wf: dict, idx: int) -> str:
        """
        找到某一格的采样器节点（为其单独写入 seed）。
        做法：从该格的正向提示词节点出发，找出把它当 positive 的那个采样器。
        """
        enc = self.prompt_nodes.get(idx)
        if not enc:
            return ""
        for nid, node in wf.items():
            ins = node.get("inputs")
            if not isinstance(ins, dict):
                continue
            if any(k in ins for k in ("seed", "noise_seed")):
                if self._resolve_conditioning_source(ins.get("positive")) == enc:
                    return nid
        return ""

    # ------------------------------------------------------------------ 任务投递
    def queue_prompt(self, workflow: dict) -> str:
        body = {"prompt": workflow, "client_id": self.client_id}
        try:
            resp = requests.post(self._url("/prompt"), json=body, timeout=COMFY_HTTP_TMO)
        except requests.exceptions.RequestException as exc:
            raise ComfyWorkflowError(f"向 ComfyUI 投递任务失败：{exc}") from exc

        if resp.status_code != 200:
            # ComfyUI 的 400 响应体里带有极其精确的 node_errors，必须原样抛给用户
            detail = resp.text[:1500]
            try:
                err = resp.json()
                detail = json.dumps(err, ensure_ascii=False, indent=2)[:1500]
            except ValueError:
                pass
            raise ComfyWorkflowError(
                f"ComfyUI 拒绝了本次任务（HTTP {resp.status_code}）。后端诊断：\n{detail}"
            )

        pid = resp.json().get("prompt_id")
        if not pid:
            raise ComfyWorkflowError(f"ComfyUI 未返回 prompt_id。响应：{resp.text[:500]}")
        self.logger.log(f"🚀 任务已入队，prompt_id = {pid}")
        return pid

    # ------------------------------------------------------------------ 状态轮询
    def wait_for_result(self, prompt_id: str):
        """
        轮询 /history 直到任务真正完成。生成器形式，边等边把进度吐给前端。
        在拿到真实产物之前，绝不返回任何图片。
        """
        t0 = time.time()
        last_beat = 0.0

        while True:
            elapsed = time.time() - t0
            if elapsed > COMFY_RENDER_TMO:
                raise TimeoutError(
                    f"ComfyUI 渲染超时（>{COMFY_RENDER_TMO}s，prompt_id={prompt_id}）。"
                    f"请检查 ComfyUI 控制台是否卡在 4K 重构或爆显存。"
                )

            try:
                resp = requests.get(self._url(f"/history/{prompt_id}"), timeout=COMFY_HTTP_TMO)
                resp.raise_for_status()
                history = resp.json()
            except requests.exceptions.RequestException as exc:
                raise ComfyWorkflowError(f"轮询 ComfyUI /history 失败：{exc}") from exc
            except ValueError as exc:
                raise ComfyWorkflowError(f"ComfyUI /history 返回非 JSON：{exc}") from exc

            entry = history.get(prompt_id)
            if entry:
                status = (entry.get("status") or {})
                status_str = str(status.get("status_str", "")).lower()
                outputs = entry.get("outputs") or {}

                # ① 后端明确报错 → 立刻熔断并透传诊断消息
                if status_str == "error":
                    msgs = json.dumps(status.get("messages", []), ensure_ascii=False)[:1500]
                    raise ComfyWorkflowError(
                        f"ComfyUI 执行报错（status_str={status_str}）。后端消息：\n{msgs}"
                    )

                # ② 拿到产物 → 正常收工
                if outputs:
                    self.logger.log(f"✅ ComfyUI 渲染完成，总耗时 {elapsed:.1f}s。")
                    yield ("done", entry)
                    return

                # ③ 已标记完成却没有任何产物 → 立刻熔断。
                #    不认这个信号就会空转到 COMFY_RENDER_TMO 才报错，白等十几分钟。
                #    常见原因：工作流里只有 PreviewImage 之类不落盘的输出节点。
                if status.get("completed") is True:
                    raise ComfyWorkflowError(
                        f"ComfyUI 已标记任务完成（status_str={status_str or 'n/a'}）但产物清单为空。"
                        f"请确认工作流里存在 SaveImage 输出节点，且其 images 端口确实连到了"
                        f"最终的 1x4 拼接结果。"
                    )

            if elapsed - last_beat >= 5:
                last_beat = elapsed
                yield ("tick", self.logger.log(
                    f"⏳ ComfyUI 正在抽卡 / 4x-UltraSharp 重构…… 已等待 {int(elapsed)}s"
                ))

            time.sleep(COMFY_POLL_SEC)

    # ------------------------------------------------------------------ 产物取回
    def fetch_panel_images(self, entry: dict, panels) -> dict:
        """
        从 /history 产物清单里按格取回真实字节流，解码为 {panel_idx: PIL.Image}。
        只认本次确实渲染的那几格；任何一格缺失都要明确报出来，绝不静默少一格。
        """
        outputs = entry.get("outputs") or {}
        want = tuple(sorted(set(int(p) for p in panels)))

        # 输出节点 ID → 格号 的反查表
        node_to_panel = {sid: idx for idx, sid in self.save_nodes.items()}

        got, missing = {}, []
        for idx in want:
            sid = self.save_nodes.get(idx)
            items = (outputs.get(sid) or {}).get("images") or [] if sid else []
            if not items:
                # 兜底：有些自定义输出节点会把图挂在别的 node id 下
                items = [it for nid, o in outputs.items()
                         if node_to_panel.get(nid) == idx
                         for it in (o.get("images") or [])]
            if not items:
                missing.append(idx)
                continue

            best = None
            for item in items:
                try:
                    blob = self._download(item)
                    img = Image.open(io.BytesIO(blob))
                    img.load()
                except Exception as exc:
                    self.logger.log(
                        f"⚠️  panel_{idx} 产物 {item.get('filename')} 取回失败：{exc}")
                    continue
                area = img.width * img.height
                if best is None or area > best[0]:
                    best = (area, img, item)

            if best is None:
                missing.append(idx)
                continue

            _, img, item = best
            got[idx] = img.convert("RGB")
            self.logger.log(
                f"🖼️  panel_{idx} 取回：{item.get('filename')}  ({img.width}×{img.height})")

        if missing:
            raise ComfyWorkflowError(
                f"ComfyUI 任务已完成，但以下分镜没有取回任何图片：panel_{missing}。"
                f"原始 outputs 节点：{list(outputs.keys())}；"
                f"已对齐的逐格输出节点：{ {k: self.save_nodes[k] for k in sorted(self.save_nodes)} }"
            )
        return got

    def _download(self, item: dict) -> bytes:
        params = {
            "filename": item.get("filename", ""),
            "subfolder": item.get("subfolder", "") or "",
            "type": item.get("type", "output") or "output",
        }
        resp = requests.get(self._url("/view"), params=params, timeout=COMFY_HTTP_TMO)
        resp.raise_for_status()
        if not resp.content:
            raise ComfyWorkflowError(f"/view 返回空字节流：{params}")
        return resp.content

    # -------------------------------------------------------------- 对外总入口
    def render(self, script: dict, checkpoint: str | None,
               panels=(1, 2, 3, 4), hires: bool = True, seeds: dict | None = None):
        """
        渲染流程的生成器。
        产出 ("tick", log_text) 作为进度心跳；最终产出 ("panels", {idx: PIL}, seeds)。
        任意环节失败 → 直接抛异常，由上层统一熔断，绝不吐假图。

        只渲染 panels 指定的那几格 —— 这是「单格重出」的核心：
        用户只对第 3 格不满意时，只烧第 3 格（~11s），另外三格复用已有图。
        """
        self.ping()
        self.align_blueprint()

        seeds = dict(seeds or {})
        payload = self.build_payload(script, checkpoint, panels, hires, seeds)
        prompt_id = self.queue_prompt(payload)

        entry = None
        for kind, data in self.wait_for_result(prompt_id):
            if kind == "tick":
                yield ("tick", data)
            else:
                entry = data

        images = self.fetch_panel_images(entry, panels)
        yield ("panels", (images, seeds))


# ==============================================================================
#  SECTION 5.5 ── 本地中文对话气泡绘制层（BUBBLE_MODE = "local"）
# ==============================================================================
#  为什么气泡放在本地画而不是交给 ComfyUI：
#    原生 ComfyUI 没有任何「往图上写字」的节点，必须装第三方自定义节点包；
#    而那类节点普遍不自带 CJK 字体，中文极易渲染成豆腐块 □□□。
#    本层直接用系统微软雅黑在本进程绘制 —— 零额外依赖、中文永不乱码、
#    坐标与换行完全可控，且图片字节流从未离开本机。
# ==============================================================================

def compose_geometry(panels: dict):
    """
    计算 1x4 合成图的几何信息。合成与「点击坐标反查格号」共用这一份计算，
    二者绝不会算出不一致的分格边界。
      W/heights : 统一宽度后各格的尺寸
      tops      : 各格在【合成图坐标系】（含外白边）中的顶边 y
    """
    if not panels or any(not panels.get(i) for i in (1, 2, 3, 4)):
        return None
    W = min(panels[i].width for i in (1, 2, 3, 4))
    heights = [panels[i].height if panels[i].width == W
               else round(panels[i].height * W / panels[i].width)
               for i in (1, 2, 3, 4)]
    tops, y = [], OUTER_BORDER_PX
    for h in heights:
        tops.append(y)
        y += h + GUTTER_PX
    return {"W": W, "heights": heights, "tops": tops, "border": OUTER_BORDER_PX}


def locate_click(panels: dict, x: int, y: int):
    """
    把成品图上的一次点击映射为 (格号, 格内局部 x, 格内局部 y)。
    点在格间白边或外白边上时，归属到最近的一格（用户想点的显然是那一格）。
    """
    geom = compose_geometry(panels)
    if not geom:
        return None
    lx = max(0, min(int(x) - geom["border"], geom["W"] - 1))

    best, best_d = 1, None
    for i, (top, h) in enumerate(zip(geom["tops"], geom["heights"]), start=1):
        if top <= y < top + h:
            return i, lx, int(y - top)
        d = top - y if y < top else y - (top + h)
        if best_d is None or d < best_d:
            best, best_d = i, d
    top = geom["tops"][best - 1]
    h = geom["heights"][best - 1]
    return best, lx, max(0, min(int(y - top), h - 1))


class ComicComposer:
    """把四格独立画面焊接成 1x4 条漫，并画上中文对话气泡（全程本地，不碰显卡）。"""

    def __init__(self, logger: RunLogger):
        self.logger = logger

    # ---------------------------------------------------------------- 字体加载
    def _load_font(self, size: int):
        for path in BUBBLE_FONT_CANDIDATES:
            if os.path.isfile(path):
                try:
                    return ImageFont.truetype(path, size), os.path.basename(path)
                except Exception:
                    continue
        self.logger.log(
            "⚠️  未找到任何可用的中文字体，气泡将回落到 PIL 默认位图字体（中文会显示为方块）。"
        )
        return ImageFont.load_default(), "default"

    # ------------------------------------------------------------ 中文友好折行
    @staticmethod
    def _wrap(text: str, font, max_w: int) -> list:
        """
        逐字符折行：中文没有空格，按词折行会撑爆气泡。
        英文/数字串视作一个不可拆单元，避免把 "ComfyUI" 劈成两半。
        """
        tokens, buf = [], ""
        for ch in text:
            if ch.isascii() and (ch.isalnum() or ch in "'-"):
                buf += ch
            else:
                if buf:
                    tokens.append(buf); buf = ""
                tokens.append(ch)
        if buf:
            tokens.append(buf)

        lines, cur = [], ""
        for tok in tokens:
            if tok == "\n":
                lines.append(cur); cur = ""
                continue
            trial = cur + tok
            if font.getlength(trial) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur); cur = tok
        if cur:
            lines.append(cur)
        return [ln for ln in lines if ln.strip()] or [text]

    # ---------------------------------------------------------------- 单格绘制
    def _draw_one(self, draw, band_box, text: str, font, line_h: int, layout=None):
        """
        在给定的一格区域（band_box）内绘制一只对话气泡。

        layout（全部为该格的【格内局部坐标】，像素）：
            bx, by : 气泡左上角锚点，缺省 = PRD 的 X:100 / Y:150
            tx, ty : 尖角顶点位置；<=0 表示"自动"（气泡左下方的经典指向）
        尖角可以指向任意方向 —— 会自动选择离尖角最近的那条气泡边作为根部，
        所以把尖角拖到气泡上方/左侧/右侧都能画出正确形状。
        """
        bx0, by0, bx1, by1 = band_box
        panel_w, panel_h = bx1 - bx0, by1 - by0
        lay = layout or {}

        max_text_w = int(panel_w * BUBBLE_MAX_W_RATIO)
        lines = self._wrap(text, font, max_text_w)
        text_w = max(int(font.getlength(ln)) for ln in lines)
        text_h = line_h * len(lines)

        pad = max(10, int(font.size * 0.55))
        bub_w, bub_h = text_w + pad * 2, text_h + pad * 2

        # ---- 气泡锚点（夹紧在格内，防止溢出画面）----
        margin = max(12, int(panel_w * 0.02))
        want_x = int(lay.get("bx") or BUBBLE_X)
        want_y = int(lay.get("by") or BUBBLE_Y)
        x0 = bx0 + max(margin, min(want_x, max(margin, panel_w - bub_w - margin)))
        y0 = by0 + max(margin, min(want_y, max(margin, panel_h - bub_h - margin)))
        x1, y1 = x0 + bub_w, y0 + bub_h

        radius = min(bub_h // 2, int(font.size * 1.1))
        lw = max(2, int(font.size / 14))

        # ---- 尖角顶点 ----
        tx, ty_ = int(lay.get("tx") or 0), int(lay.get("ty") or 0)
        if tx > 0 or ty_ > 0:
            tip = (bx0 + tx, by0 + ty_)
        else:
            tip = (x0 + int(bub_w * 0.12), y1 + int(bub_h * 0.42))
        tip = (max(bx0 + 2, min(tip[0], bx1 - 2)),
               max(by0 + 2, min(tip[1], by1 - 2)))

        # ① 圆角主体（白底 + 黑描边）
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius,
                               fill=BUBBLE_FILL, outline=BUBBLE_OUTLINE, width=lw)

        # ② 尖角：选离尖角最近的一条边作根部，先用纯白盖掉该段边框，再补两条黑斜边
        over = {"bottom": tip[1] - y1, "top": y0 - tip[1],
                "right": tip[0] - x1, "left": x0 - tip[0]}
        side = max(over, key=lambda k: over[k])
        if over[side] > 0:
            # 尖角根部宽度随尖角长度适度变宽：否则拖远之后会细得像一根针
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            dist = max(abs(tip[0] - cx), abs(tip[1] - cy))
            half = max(6, min(int(font.size * 0.26 + dist * 0.035),
                              int(font.size * 0.9)))
            if side in ("bottom", "top"):
                lo, hi = x0 + radius + half, x1 - radius - half
                c = max(lo, min(tip[0], hi)) if hi > lo else (x0 + x1) // 2
                edge = (y1 - lw) if side == "bottom" else (y0 + lw)
                b1, b2 = (c - half, edge), (c + half, edge)
            else:
                lo, hi = y0 + radius + half, y1 - radius - half
                c = max(lo, min(tip[1], hi)) if hi > lo else (y0 + y1) // 2
                edge = (x1 - lw) if side == "right" else (x0 + lw)
                b1, b2 = (edge, c - half), (edge, c + half)

            draw.polygon([b1, b2, tip], fill=BUBBLE_FILL)
            draw.line([b1, tip], fill=BUBBLE_OUTLINE, width=lw, joint="curve")
            draw.line([tip, b2], fill=BUBBLE_OUTLINE, width=lw, joint="curve")

        # ③ 台词文本（逐行居中）
        cur_y = y0 + pad
        for ln in lines:
            lx = x0 + (bub_w - int(font.getlength(ln))) // 2
            draw.text((lx, cur_y), ln, font=font, fill=BUBBLE_TEXT_COLOR)
            cur_y += line_h

        return len(lines)

    # ---------------------------------------------------------------- 对外入口
    def compose(self, panels: dict, script: dict, layouts: dict | None = None):
        """
        把四格独立图片焊接成 1x4 竖排条漫，逐格绘制中文气泡，再补一圈纯白外边。

        焊接放在本地而不是 ComfyUI，换来的是「单格重出」能力：
        重算一格只需 ~11s，其余三格直接复用内存里的 PIL 对象，
        而改台词/改气泡根本不碰显卡，本地重排是毫秒级的。
        """
        missing = [i for i in (1, 2, 3, 4) if not panels.get(i)]
        if missing:
            raise ValueError(f"焊接失败：缺少 panel_{missing} 的图片，拒绝输出残缺条漫。")

        # 几何一律由 compose_geometry 统一计算：前端点击反查用的是同一份结果，
        # 所以"点哪里"和"画在哪里"不可能对不上。
        geom = compose_geometry(panels)
        W = geom["W"]
        tops = [t - geom["border"] for t in geom["tops"]]      # 转成未加外边的坐标系

        # 以最窄的一格为基准宽度，其余等比缩放 —— 防止不同批次参数不一致导致错位
        bands = []
        for i in (1, 2, 3, 4):
            im = panels[i].convert("RGB")
            if im.width != W:
                im = im.resize((W, geom["heights"][i - 1]), Image.LANCZOS)
            bands.append(im)

        img = Image.new("RGB", (W, sum(geom["heights"]) + GUTTER_PX * 3), BUBBLE_FILL)
        for b, top in zip(bands, tops):
            img.paste(b, (0, top))

        if BUBBLE_MODE == "comfy":
            self.logger.log("💬 气泡策略 comfy：跳过本地绘制（台词应已由工作流节点烧进画面）。")
        else:
            font_size = max(20, int(W / BUBBLE_FONT_DIVISOR))
            font, font_name = self._load_font(font_size)
            ascent, descent = (font.getmetrics() if hasattr(font, "getmetrics")
                               else (font_size, 0))
            line_h = int((ascent + descent) * 1.18) or int(font_size * 1.3)

            draw = ImageDraw.Draw(img)
            drawn = 0
            for i in (1, 2, 3, 4):
                text = (script[f"panel_{i}"].get("bubble") or "").strip()
                if not text:
                    # 无台词静默格（空镜、切镜、纯场景）—— 不画气泡，保持画面干净
                    self.logger.log(f"🔇 panel_{i} 无台词，跳过气泡绘制（静默格）。")
                    continue
                band = (0, tops[i - 1], W, tops[i - 1] + bands[i - 1].height)
                lay = (layouts or {}).get(i) or {}
                n = self._draw_one(draw, band, text, font, line_h, lay)
                drawn += 1
                ax = int(lay.get("bx") or BUBBLE_X)
                ay = int(lay.get("by") or BUBBLE_Y)
                tip = (f"尖角 {int(lay['tx'])},{int(lay['ty'])}"
                       if (lay.get("tx") or lay.get("ty")) else "尖角自动")
                self.logger.log(
                    f"💬 panel_{i} 气泡（{n} 行，锚点 {ax},{ay}；{tip}）：{text}"
                )
            self.logger.log(
                f"🖍️  气泡层完成：{drawn}/4 格有台词，字体 {font_name} @ {font_size}px。")

        if OUTER_BORDER_PX > 0:
            img = ImageOps.expand(img, border=OUTER_BORDER_PX, fill=BUBBLE_FILL)

        self.logger.log(
            f"🧩 1x4 焊接完成：单格 {W}×{bands[0].height}，"
            f"{GUTTER_PX}px 格边 + {OUTER_BORDER_PX}px 外边 → 成品 {img.width}×{img.height}。"
        )
        return img


# ==============================================================================
#  SECTION 6 ── 业务总控制器（Gradio 回调；生成器实现边跑边刷日志）
# ==============================================================================

def format_script_report(script: dict) -> str:
    rows = ["=" * 60, "🎬 四格剧本（英文 Prompt 已完成黄金触发词钢印）", "=" * 60]
    for i in (1, 2, 3, 4):
        panel = script[f"panel_{i}"]
        rows += [
            f"【第 {i} 格】",
            f"  Prompt : {panel['prompt']}",
            f"  气泡台词: {panel['bubble']}",
            "",
        ]
    rows.append("=" * 60)
    return "\n".join(rows)


def new_studio_state() -> dict:
    """
    一次创作会话的全部可复用资产。放在 gr.State 里（每个浏览器标签各自独立）。
      script : 四格剧本（prompt + bubble）
      panels : {idx: PIL.Image}  —— 已渲染好的分格画面，单格重出时其余三格直接复用
      seeds  : {idx: int}        —— 逐格抽卡结果，便于复现或锁定满意的那一格
    """
    return {"script": None, "panels": {}, "seeds": {}, "idea": "", "layouts": {}}


def layout_fields(state: dict, idx: int):
    """把某一格的气泡布局（锚点 + 尖角）回填到前端数值框。"""
    lay = ((state or {}).get("layouts") or {}).get(int(idx)) or {}
    return (gr.update(value=int(lay.get("bx") or BUBBLE_X)),
            gr.update(value=int(lay.get("by") or BUBBLE_Y)),
            gr.update(value=int(lay.get("tx") or 0)),
            gr.update(value=int(lay.get("ty") or 0)))


def _recompose(state: dict, logger: RunLogger):
    """用当前 state 重新本地合成一张成品（不碰显卡）。"""
    return ComicComposer(logger).compose(
        state["panels"], state["script"], state.get("layouts"))


def place_bubble(mode: str, state: dict, evt: gr.SelectData):
    """
    在成品图上点一下 → 把气泡锚点或尖角顶点放到那里，立即本地重排。
    这是"画布"式交互的核心：点击坐标经 locate_click 反查成 (格号, 格内坐标)，
    于是用户点第 3 格的某个位置，改的就是第 3 格的布局。
    """
    logger = RunLogger()
    state = dict(state or new_studio_state())
    script, panels = state.get("script"), state.get("panels") or {}
    noop = (gr.update(),) * 5

    if not script or len(panels) < 4:
        return logger.log("❌ 还没有完整四格成品，先生成一次再来摆气泡。"), None, state, *noop
    if not evt or evt.index is None:
        return logger.log("❌ 没拿到点击坐标。"), None, state, *noop

    try:
        x, y = int(evt.index[0]), int(evt.index[1])
    except (TypeError, ValueError, IndexError):
        return logger.log(f"❌ 点击坐标格式异常：{evt.index!r}"), None, state, *noop

    hit = locate_click(panels, x, y)
    if not hit:
        return logger.log("❌ 无法定位点击位置所属的分格。"), None, state, *noop
    idx, lx, ly = hit

    layouts = dict(state.get("layouts") or {})
    lay = dict(layouts.get(idx) or {})
    if "尖角" in str(mode):
        lay["tx"], lay["ty"] = lx, ly
        what = f"尖角顶点 → 格内 ({lx}, {ly})"
    else:
        lay["bx"], lay["by"] = lx, ly
        what = f"气泡左上角 → 格内 ({lx}, {ly})"
    layouts[idx] = lay
    state["layouts"] = layouts

    logger.log(f"🎯 panel_{idx} {what}（成品坐标 {x},{y}）")
    if not (script[f"panel_{idx}"].get("bubble") or "").strip():
        logger.log(f"⚠️  panel_{idx} 当前没有台词，气泡不会显示 —— 先给这一格填台词。")

    try:
        final = _recompose(state, logger)
    except Exception as exc:
        traceback.print_exc()
        return logger.log(f"❌ 本地重排失败：{exc}"), None, state, *noop

    _archive(final, logger)
    return (logger.log("🏁 已本地重排完成（零显卡开销）。"), final, state,
            gr.update(value=idx), *layout_fields(state, idx))


def apply_layout(idx: int, bx, by, tx, ty, state: dict):
    """用数值框里的精确坐标应用气泡布局（点击之后的微调手段）。"""
    logger = RunLogger()
    state = dict(state or new_studio_state())
    idx = int(idx)
    if not state.get("script") or len(state.get("panels") or {}) < 4:
        return logger.log("❌ 还没有完整四格成品。"), None, state

    layouts = dict(state.get("layouts") or {})
    layouts[idx] = {"bx": int(bx or 0), "by": int(by or 0),
                    "tx": int(tx or 0), "ty": int(ty or 0)}
    state["layouts"] = layouts
    logger.log(f"🎯 panel_{idx} 布局：锚点 {int(bx or 0)},{int(by or 0)}  "
               f"尖角 {int(tx or 0)},{int(ty or 0)}（0 = 自动）")
    try:
        final = _recompose(state, logger)
    except Exception as exc:
        traceback.print_exc()
        return logger.log(f"❌ 本地重排失败：{exc}"), None, state
    _archive(final, logger)
    return logger.log("🏁 已本地重排完成。"), final, state


def spread_layout(idx: int, state: dict):
    """把当前格的气泡布局套用到全部四格。"""
    logger = RunLogger()
    state = dict(state or new_studio_state())
    idx = int(idx)
    if not state.get("script") or len(state.get("panels") or {}) < 4:
        return logger.log("❌ 还没有完整四格成品。"), None, state

    lay = dict((state.get("layouts") or {}).get(idx) or {})
    state["layouts"] = {i: dict(lay) for i in (1, 2, 3, 4)}
    logger.log(f"📐 已把 panel_{idx} 的气泡布局套用到全部四格。")
    try:
        final = _recompose(state, logger)
    except Exception as exc:
        traceback.print_exc()
        return logger.log(f"❌ 本地重排失败：{exc}"), None, state
    _archive(final, logger)
    return logger.log("🏁 已本地重排完成。"), final, state


def reset_layout(state: dict):
    """恢复四格气泡布局为默认（锚点 X:100 Y:150，尖角自动）。"""
    logger = RunLogger()
    state = dict(state or new_studio_state())
    if not state.get("script") or len(state.get("panels") or {}) < 4:
        return (logger.log("❌ 还没有完整四格成品。"), None, state,
                *layout_fields(new_studio_state(), 1))

    state["layouts"] = {}
    logger.log(f"↩️  气泡布局已恢复默认（锚点 {BUBBLE_X},{BUBBLE_Y}，尖角自动）。")
    try:
        final = _recompose(state, logger)
    except Exception as exc:
        traceback.print_exc()
        return (logger.log(f"❌ 本地重排失败：{exc}"), None, state,
                *layout_fields(state, 1))
    _archive(final, logger)
    return (logger.log("🏁 已本地重排完成。"), final, state, *layout_fields(state, 1))


def panel_fields(state: dict, idx: int):
    """把某一格的 prompt / bubble / seed 回填到前端精修区。"""
    script = (state or {}).get("script")
    if not script:
        return gr.update(), gr.update(), gr.update()
    p = script.get(f"panel_{int(idx)}") or {}
    seed = (state.get("seeds") or {}).get(int(idx), 0)
    return (gr.update(value=p.get("prompt", "")),
            gr.update(value=p.get("bubble", "")),
            gr.update(value=seed or 0))


def _archive(image, logger: RunLogger, tag: str = "comic"):
    """成品落盘存档（创意资产永远只存在于本地硬盘）。"""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image.save(path)
        logger.log(f"💾 成品已存档：{path}")
    except Exception as exc:
        logger.log(f"⚠️  本地存档失败（不影响前端展示）：{exc}")


def _comfy_preconditions(logger: RunLogger):
    """渲染前置条件检查。返回 None 表示通过，否则返回该报给用户的日志文本。"""
    if not _PIL_OK:
        return logger.log("❌ 未安装 Pillow，无法解码 ComfyUI 产物。请执行：pip install pillow")
    if not COMFY_WORKFLOW.is_file():
        return logger.log(
            "⚠️  Ollama 剧本生成成功，ComfyUI 模块待对接。\n"
            f"    原因：未找到物理图纸 {COMFY_WORKFLOW.name}\n"
            "    请执行：python build_workflow.py  自动生成四格图纸。\n"
            "    按契约，图像通路未打通 → gr.Image 返回 None（不使用任何占位假图）。"
        )
    return None


def generate_comic(idea: str, ollama_model: str, ckpt_choice: str,
                   enable_render: bool, hires: bool, state: dict):
    """
    【完整四格】主回调（生成器）。
    输出契约恒为 (日志, 图片或None, state, 精修-prompt, 精修-bubble, 精修-seed)：
      · 在 ComfyUI 真正回传产物之前，图片永远是 None。
      · 任何异常都转为可读日志 + None，绝不伪造"生成成功"。
    """
    logger = RunLogger()
    state = dict(state or new_studio_state())
    hold = (gr.update(), gr.update(), gr.update())      # 精修区不动

    if not idea or not idea.strip():
        yield logger.log("❌ 请先输入一句中文剧情再召唤战神。"), None, state, *hold
        return

    idea = idea.strip()
    state["idea"] = idea
    yield logger.log("=" * 60), None, state, *hold
    yield logger.log(f"🎯 剧情输入：{idea}"), None, state, *hold
    yield logger.log(f"🧠 文本大脑：{ollama_model}   🎛️ 底模：{ckpt_choice}"), None, state, *hold
    yield logger.log(
        f"🖌️  渲染：{'开' if enable_render else '关（仅出剧本）'}   "
        f"✨ 高清重构：{'开（4x-UltraSharp，+约 3.5s/格）' if hires else '关（快速模式）'}"
    ), None, state, *hold

    # ---------- 阶段一：文本大脑 --------------------------------------------
    try:
        ollama_preflight(logger, ollama_model)
        yield logger.log(f"🧠 正在呼叫本地 Ollama（{ollama_model}）扩写四格剧本……"), None, state, *hold
        raw = call_ollama(idea, logger, ollama_model)
    except Exception as exc:
        traceback.print_exc()
        yield logger.log(f"❌ 文本大脑阶段失败：{exc}"), None, state, *hold
        return

    # ---------- 阶段二：净化 + 验证 + 钢印 ----------------------------------
    try:
        script = stamp_golden_trigger(
            normalize_and_validate(extract_json_object(raw, logger), logger), logger)
    except Exception as exc:
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("[DEBUG] Ollama 原始返回内容（完整，用于排错）：")
        print(raw)
        print("=" * 60 + "\n")
        yield logger.log(
            f"❌ 剧本解析/验证阶段失败：{exc}\n"
            f"（完整原始返回已打印到控制台，请查看启动 main.py 的终端窗口）"
        ), None, state, *hold
        return

    state["script"] = script
    state["panels"] = {}
    state["seeds"] = {}
    filled = panel_fields(state, 1)

    yield logger.log("🎉 Ollama 四格剧本生成并校验成功。"), None, state, *filled
    yield logger.raw(format_script_report(script)), None, state, *filled

    # ---------- 阶段三：ComfyUI 渲染 ----------------------------------------
    if not enable_render:
        yield logger.log("ℹ️  已按用户选择跳过 ComfyUI 渲染。gr.Image 返回 None。"), None, state, *filled
        return

    blocked = _comfy_preconditions(logger)
    if blocked:
        yield blocked, None, state, *filled
        return

    pilot = ComfyPilot(COMFY_HOST, COMFY_WORKFLOW, logger)
    checkpoint = CHECKPOINT_MAP.get(ckpt_choice)
    panels = seeds = None

    try:
        yield logger.log("🎨 开始对接后台 ComfyUI 渲染总监（四格并联）……"), None, state, *filled
        for kind, data in pilot.render(script, checkpoint, (1, 2, 3, 4), hires):
            if kind == "tick":
                yield data, None, state, *filled          # 进度期间图片恒为 None
            else:
                panels, seeds = data
    except Exception as exc:
        traceback.print_exc()
        yield logger.log(
            f"❌ ComfyUI 渲染阶段失败：{exc}\n"
            f"    （剧本已生成成功，可关掉渲染开关单独复用剧本）\n"
            f"    按契约，渲染未完成 → gr.Image 返回 None。"
        ), None, state, *filled
        return

    if not panels:
        yield logger.log("❌ 渲染结束但未取回任何分格画面，拒绝伪造成功。"), None, state, *filled
        return

    state["panels"], state["seeds"] = panels, seeds
    filled = panel_fields(state, 1)

    try:
        final = _recompose(state, logger)
    except Exception as exc:
        traceback.print_exc()
        yield logger.log(f"❌ 本地焊接/气泡阶段失败：{exc}"), None, state, *filled
        return

    _archive(final, logger)
    yield logger.log(
        "🏁 全链路贯通。不满意某一格？到左下「单格精修」改提示词后只重烧那一格（约 11s）。"
    ), final, state, *filled


def _render_single(idx: int, state: dict, ckpt_choice: str, hires: bool,
                   want_seed: int, logger: RunLogger):
    """
    单格渲染的公共内核（生成器）。
    三种重出方式（手改提示词 / 换抽卡 / Qwen 重写）共用这一段，避免逻辑分叉。
    产出 ("tick", log) 心跳，成功时最后产出 ("done", 成品图)。
    """
    script = state["script"]

    blocked = _comfy_preconditions(logger)
    if blocked:
        yield ("fail", blocked)
        return

    yield ("tick", logger.log(
        f"🎲 seed：{want_seed if want_seed else '随机抽卡'}"
        f"{'（锁定构图）' if want_seed else ''}   ✨ 高清重构：{'开' if hires else '关'}"
    ))

    pilot = ComfyPilot(COMFY_HOST, COMFY_WORKFLOW, logger)
    got = None
    try:
        for kind, data in pilot.render(script, CHECKPOINT_MAP.get(ckpt_choice),
                                       (idx,), hires,
                                       {idx: want_seed} if want_seed else None):
            if kind == "tick":
                yield ("tick", data)
            else:
                got, new_seeds = data
                state["seeds"] = {**state.get("seeds", {}), **new_seeds}
    except Exception as exc:
        traceback.print_exc()
        yield ("fail", logger.log(
            f"❌ panel_{idx} 重出失败：{exc}\n    按契约，gr.Image 返回 None。"))
        return

    if not got or idx not in got:
        yield ("fail", logger.log(f"❌ panel_{idx} 未取回画面，拒绝伪造成功。"))
        return

    state["panels"] = {**state.get("panels", {}), idx: got[idx]}

    have = sorted(state["panels"])
    if len(have) < 4:
        yield ("fail", logger.log(
            f"✅ panel_{idx} 已重出。但目前只持有 panel_{have} 共 {len(have)} 格，"
            f"不足 4 格无法焊接完整条漫 —— 请先跑一次完整四格。"))
        return

    try:
        final = _recompose(state, logger)
    except Exception as exc:
        traceback.print_exc()
        yield ("fail", logger.log(f"❌ 本地焊接/气泡阶段失败：{exc}"))
        return

    _archive(final, logger)
    yield ("done", final)


def reroll_panel(idx: int, ckpt_choice: str, hires: bool, state: dict):
    """
    【换个抽卡】提示词和台词一个字都不改，只换 seed 重新抽一张。
    用于"内容对了但这张画得不好看"的场景。
    """
    logger = RunLogger()
    state = dict(state or new_studio_state())
    idx = int(idx)

    if not state.get("script"):
        yield logger.log("❌ 还没有剧本。请先生成一次完整四格。"), None, state, gr.update()
        return

    yield logger.log("=" * 60), None, state, gr.update()
    yield logger.log(
        f"🎲 换个抽卡：panel_{idx}（提示词与台词保持原样，只换 seed）"), None, state, gr.update()

    for kind, data in _render_single(idx, state, ckpt_choice, hires, 0, logger):
        if kind == "tick":
            yield data, None, state, gr.update()
        elif kind == "fail":
            yield data, None, state, gr.update()
            return
        else:
            yield (logger.log(f"🏁 panel_{idx} 已换新一张。"), data, state,
                   gr.update(value=state["seeds"].get(idx, 0)))


def rewrite_panel(idx: int, ollama_model: str, lock_seed: bool,
                  ckpt_choice: str, hires: bool, state: dict):
    """
    【让 Qwen 重写这一格】不需要用户懂英文提示词：
    本地 14B 读取整个四格上下文，为这一格重新设计画面与台词，然后直接重烧。
    lock_seed=True 时沿用该格原来的 seed —— 构图骨架保持，内容换新。
    """
    logger = RunLogger()
    state = dict(state or new_studio_state())
    idx = int(idx)
    hold = (gr.update(), gr.update(), gr.update())

    script = state.get("script")
    if not script:
        yield logger.log("❌ 还没有剧本。请先生成一次完整四格。"), None, state, *hold
        return

    yield logger.log("=" * 60), None, state, *hold
    yield logger.log(
        f"🧠 让 {ollama_model} 重写 panel_{idx} 的画面与台词……"), None, state, *hold

    try:
        ollama_preflight(logger, ollama_model)
        new_prompt, new_bubble = rewrite_panel_via_ollama(
            state.get("idea", ""), script, idx, ollama_model, logger)
    except Exception as exc:
        traceback.print_exc()
        yield logger.log(f"❌ Qwen 重写失败：{exc}"), None, state, *hold
        return

    script = copy.deepcopy(script)
    script[f"panel_{idx}"]["prompt"] = new_prompt
    script[f"panel_{idx}"]["bubble"] = new_bubble
    script = stamp_golden_trigger(script, logger)
    state["script"] = script

    filled = panel_fields(state, idx)
    yield logger.log(f"✍️  新画面：{new_prompt}"), None, state, *filled
    yield logger.log(
        f"✍️  新台词：{new_bubble or '（无台词静默格）'}"), None, state, *filled

    want_seed = int(state.get("seeds", {}).get(idx, 0)) if lock_seed else 0
    for kind, data in _render_single(idx, state, ckpt_choice, hires, want_seed, logger):
        if kind == "tick":
            yield data, None, state, *filled
        elif kind == "fail":
            yield data, None, state, *filled
            return
        else:
            yield (logger.log(f"🏁 panel_{idx} 已按 Qwen 的新设计重出。"),
                   data, state, *panel_fields(state, idx))


def append_presets(prompt_text: str, expr: str, shot: str, mood: str):
    """
    把下拉框选中的中文预设翻译成英文词条，追加到提示词尾部。
    纯文本操作，不碰显卡也不碰 Ollama —— 想生效再点「重出这一格」。
    """
    adds = []
    for table, pick in ((EXPRESSION_PRESETS, expr), (SHOT_PRESETS, shot),
                        (MOOD_PRESETS, mood)):
        tags = table.get(pick or NONE_OPTION, "")
        if tags:
            adds.append(tags)
    if not adds:
        return gr.update(), "⚠️  三个下拉框都是「（不追加）」，没有东西可以追加。"

    base = (prompt_text or "").strip().rstrip(",")
    # 去重：已经存在的词条不再重复追加（重复会稀释权重）
    have = {t.strip().lower() for t in base.split(",") if t.strip()}
    fresh = [t.strip() for group in adds for t in group.split(",")
             if t.strip() and t.strip().lower() not in have]
    if not fresh:
        return gr.update(), "ℹ️  选中的词条已经都在提示词里了，无需重复追加。"

    merged = f"{base}, {', '.join(fresh)}" if base else ", ".join(fresh)
    return (gr.update(value=merged),
            f"➕ 已追加 {len(fresh)} 个词条：{', '.join(fresh)}\n"
            f"   （还没出图 —— 点「🔁 重出这一格」才会生效）")


def qwen_extend_prompt(prompt_text: str, ollama_model: str):
    """让 Qwen 为当前提示词补充画面细节词条，只填进文本框，不自动出图。"""
    logger = RunLogger()
    if not prompt_text or not prompt_text.strip():
        return gr.update(), logger.log("❌ 提示词是空的，没什么可以补充。")
    try:
        ollama_preflight(logger, ollama_model)
        tags = extend_prompt_via_ollama(prompt_text, ollama_model, logger)
    except Exception as exc:
        traceback.print_exc()
        return gr.update(), logger.log(f"❌ Qwen 补充词条失败：{exc}")

    base = prompt_text.strip().rstrip(",")
    have = {t.strip().lower() for t in base.split(",") if t.strip()}
    fresh = [t.strip() for t in tags.split(",")
             if t.strip() and t.strip().lower() not in have]
    if not fresh:
        return gr.update(), logger.log("ℹ️  Qwen 给的词条已经都在提示词里了。")

    merged = f"{base}, {', '.join(fresh)}"
    logger.log(f"➕ Qwen 补充了 {len(fresh)} 个词条：{', '.join(fresh)}")
    return gr.update(value=merged), logger.log(
        "   （还没出图 —— 点「🔁 重出这一格」才会生效）")


def rerender_panel(idx: int, prompt_text: str, bubble_text: str, seed_val,
                   ckpt_choice: str, hires: bool, state: dict):
    """
    【单格重出】按文本框里的提示词/台词重烧指定的那一格，其余三格复用已有画面。
    这是本项目相对"整张重生成"的核心效率优势：67s → 约 14s。
    """
    logger = RunLogger()
    state = dict(state or new_studio_state())
    idx = int(idx)

    script = state.get("script")
    if not script:
        yield logger.log("❌ 还没有剧本。请先点上方按钮生成一次完整四格。"), None, state
        return
    if not prompt_text or not prompt_text.strip():
        yield logger.log(f"❌ panel_{idx} 的英文提示词不能为空。"), None, state
        return

    # 用户在精修框里的编辑结果写回剧本，并重新过一遍钢印与违禁词拦截
    script = copy.deepcopy(script)
    script[f"panel_{idx}"]["prompt"] = prompt_text.strip()
    # 允许清空台词 —— 用户把台词框清掉就是要这一格变成无台词静默格
    script[f"panel_{idx}"]["bubble"] = (bubble_text or "").strip()
    script = stamp_golden_trigger(script, logger)
    state["script"] = script

    yield logger.log("=" * 60), None, state
    yield logger.log(f"🔁 单格重出：panel_{idx}（其余三格复用，不重算）"), None, state
    yield logger.log(f"   提示词：{script[f'panel_{idx}']['prompt'][:140]}……"), None, state

    want_seed = int(seed_val) if seed_val else 0
    for kind, data in _render_single(idx, state, ckpt_choice, hires, want_seed, logger):
        if kind == "tick":
            yield data, None, state
        elif kind == "fail":
            yield data, None, state
            return
        else:
            yield logger.log(f"🏁 panel_{idx} 已替换，条漫重新焊接完成。"), data, state


def recompose_only(idx: int, bubble_text: str, state: dict):
    """
    【仅改台词】完全不碰显卡：改掉某一格的气泡文字后本地重排，毫秒级返回。
    """
    logger = RunLogger()
    state = dict(state or new_studio_state())
    idx = int(idx)

    script, panels = state.get("script"), state.get("panels") or {}
    if not script:
        return logger.log("❌ 还没有剧本。请先生成一次完整四格。"), None, state
    if len(panels) < 4:
        return logger.log(
            f"❌ 目前只持有 {len(panels)} 格画面，不足 4 格无法焊接。请先生成一次完整四格。"
        ), None, state
    text = (bubble_text or "").strip()
    script = copy.deepcopy(script)
    script[f"panel_{idx}"]["bubble"] = text          # 留空 = 把这一格改成无台词静默格
    state["script"] = script

    logger.log(
        f"💬 仅改台词：panel_{idx} → "
        f"{'「' + text + '」' if text else '（清空，改为无台词静默格）'}（不启动显卡）")
    try:
        final = _recompose(state, logger)
    except Exception as exc:
        traceback.print_exc()
        return logger.log(f"❌ 本地焊接/气泡阶段失败：{exc}"), None, state

    _archive(final, logger)
    return logger.log("🏁 台词已更新，条漫本地重排完成（零显卡开销）。"), final, state


def refresh_ollama_models():
    """刷新 Ollama 模型下拉框。"""
    names = list_ollama_models()
    if not names:
        return gr.update(choices=[OLLAMA_MODEL], value=OLLAMA_MODEL)
    return gr.update(choices=names, value=names[0])


# ==============================================================================
#  SECTION 7 ── Gradio 极简前端大底座（保持原始框架，不引入任何额外 Web 框架）
# ==============================================================================

_OLLAMA_CHOICES = list_ollama_models() or [OLLAMA_MODEL]

with gr.Blocks(title="Comic-Auto-Pilot 本地全自动漫画制片厂") as demo:
    gr.Markdown("# 🚀 Comic-Auto-Pilot (四格条漫全自动本地极速制片厂) v3.0")
    gr.Markdown("### 🔒 100% 本地物理运行 | 创意资产零外泄 | 单格可重烧,不满意只改那一格")

    studio = gr.State(new_studio_state())

    with gr.Row():
        with gr.Column(scale=1):
            user_text = gr.Textbox(
                label="📥 请输入你脑海中的漫画剧情大白话（纯中文）",
                placeholder="例如：大肥鱼深夜在阁楼偷偷摸鱼，被主人推门当场抓包……",
                lines=4,
            )
            with gr.Row():
                ollama_drop = gr.Dropdown(
                    choices=_OLLAMA_CHOICES, value=_OLLAMA_CHOICES[0],
                    label="🧠 文本大脑（本地 Ollama 模型）", scale=4,
                )
                refresh_btn = gr.Button("🔄", scale=1, min_width=50)
            model_drop = gr.Dropdown(
                choices=list(CHECKPOINT_MAP.keys()),
                value=list(CHECKPOINT_MAP.keys())[0],
                label="🎛️ 本地大底模选择（User 自主加载）",
            )
            with gr.Row():
                render_switch = gr.Checkbox(
                    value=True, label="🎨 启用 ComfyUI 渲染",
                )
                hires_switch = gr.Checkbox(
                    value=True, label="✨ 4x-UltraSharp 高清重构（关掉省约 24s）",
                )
            run_btn = gr.Button("🔥 召唤 5080 战神！一键全自动产粮！", variant="primary")

            # ---------------- 单格精修区 ----------------
            with gr.Accordion("🔧 单格精修 —— 只重烧你不满意的那一格（约 11s）", open=True):
                gr.Markdown(
                    "整张重生成要 **56s**,单格重出只要 **11s**,改台词更是 **零显卡开销**。"
                )
                panel_radio = gr.Radio(
                    choices=[("第 1 格", 1), ("第 2 格", 2), ("第 3 格", 3), ("第 4 格", 4)],
                    value=1, label="挑一格",
                )
                panel_prompt = gr.Textbox(
                    label="该格英文提示词（人物钢印自动前置,不必重复写；"
                          "开头写 [NOCHAR] = 镜头切走的空镜格,不要角色）",
                    lines=5, max_lines=10,
                )
                panel_bubble = gr.Textbox(
                    label="该格中文台词（留空 = 无台词静默格,不画气泡）", lines=2,
                )
                panel_seed = gr.Number(
                    label="seed（0 = 随机抽卡；填入上次的 seed 可锁定复现）",
                    value=0, precision=0,
                )
                gr.Markdown("**不会写英文提示词?下面三个按钮都不需要你动提示词:**")
                with gr.Row():
                    reroll_btn = gr.Button("🎲 换个抽卡（原样重抽）")
                    rewrite_btn = gr.Button("🧠 让 Qwen 重写这一格", variant="primary")
                lock_seed_chk = gr.Checkbox(
                    value=False,
                    label="🔒 Qwen 重写时锁定 seed（构图骨架保持不变,只换内容）",
                )

                gr.Markdown("**想微调表情/景别/氛围?选好后追加到提示词,再点重出:**")
                with gr.Row():
                    expr_drop = gr.Dropdown(
                        choices=list(EXPRESSION_PRESETS), value=NONE_OPTION,
                        label="表情",
                    )
                    shot_drop = gr.Dropdown(
                        choices=list(SHOT_PRESETS), value=NONE_OPTION, label="景别",
                    )
                    mood_drop = gr.Dropdown(
                        choices=list(MOOD_PRESETS), value=NONE_OPTION, label="氛围/光照",
                    )
                with gr.Row():
                    append_btn = gr.Button("➕ 追加选中的词条")
                    qwen_ext_btn = gr.Button("🧠 让 Qwen 补充画面细节")

                with gr.Row():
                    rerender_btn = gr.Button("🔁 重出这一格（走显卡）", variant="primary")
                    bubble_btn = gr.Button("💬 仅改台词（本地秒出）")

            # ---------------- 气泡布局画布区 ----------------
            with gr.Accordion("🎯 气泡布局 —— 直接在右边成品图上点,即可摆位", open=True):
                gr.Markdown(
                    "**用法**:先选下面的模式,再到右边成品图上点一下。\n"
                    "点哪一格就改哪一格(会自动帮你切到那一格),每次点击本地重排约 0.5s,**不碰显卡**。\n"
                    "尖角可以拖到气泡的上/下/左/右任意一侧,程序会自动选最近的边当根部。"
                )
                place_mode = gr.Radio(
                    choices=["① 放气泡（左上角）", "② 放尖角（顶点）"],
                    value="① 放气泡（左上角）", label="点击时要摆的是什么",
                )
                with gr.Row():
                    lay_bx = gr.Number(label="气泡 X", value=BUBBLE_X, precision=0)
                    lay_by = gr.Number(label="气泡 Y", value=BUBBLE_Y, precision=0)
                    lay_tx = gr.Number(label="尖角 X（0=自动）", value=0, precision=0)
                    lay_ty = gr.Number(label="尖角 Y（0=自动）", value=0, precision=0)
                with gr.Row():
                    apply_btn = gr.Button("📐 应用这些数值")
                    spread_btn = gr.Button("🪄 套用到四格")
                    reset_btn = gr.Button("↩️ 恢复默认")

            gr.Markdown(
                f"**物理链路**：Ollama `{OLLAMA_HOST}` → ComfyUI `{COMFY_HOST}` → "
                f"图纸 `{COMFY_WORKFLOW.name}`"
            )

        with gr.Column(scale=1):
            log_output = gr.Textbox(
                label="🤖 赛博编剧后台日志监控（诚实日志，不虚构成功）",
                interactive=False,
                lines=20,
                max_lines=30,
                autoscroll=True,
                buttons=["copy"],            # Gradio 6 的复制按钮写法
            )
            image_output = gr.Image(
                label="🖼️ 1x4 最终成片条漫 —— 在图上点一下即可摆气泡位置",
                type="pil",
                interactive=False,
            )

    # ---------------- 事件绑定 ----------------
    refresh_btn.click(fn=refresh_ollama_models, inputs=None, outputs=[ollama_drop])

    LAY_BOXES = [lay_bx, lay_by, lay_tx, lay_ty]

    run_btn.click(
        fn=generate_comic,
        inputs=[user_text, ollama_drop, model_drop, render_switch, hires_switch, studio],
        outputs=[log_output, image_output, studio, panel_prompt, panel_bubble, panel_seed],
    ).then(fn=layout_fields, inputs=[studio, panel_radio], outputs=LAY_BOXES)

    # 切换分格 → 把该格的提示词/台词/seed 与气泡布局一并回填
    panel_radio.change(
        fn=panel_fields,
        inputs=[studio, panel_radio],
        outputs=[panel_prompt, panel_bubble, panel_seed],
    )
    panel_radio.change(
        fn=layout_fields, inputs=[studio, panel_radio], outputs=LAY_BOXES,
    )

    rerender_btn.click(
        fn=rerender_panel,
        inputs=[panel_radio, panel_prompt, panel_bubble, panel_seed,
                model_drop, hires_switch, studio],
        outputs=[log_output, image_output, studio],
    )

    # 换个抽卡：提示词台词都不动，只换 seed
    reroll_btn.click(
        fn=reroll_panel,
        inputs=[panel_radio, model_drop, hires_switch, studio],
        outputs=[log_output, image_output, studio, panel_seed],
    )

    # 让 Qwen 重写这一格的画面与台词，然后直接重烧
    rewrite_btn.click(
        fn=rewrite_panel,
        inputs=[panel_radio, ollama_drop, lock_seed_chk, model_drop,
                hires_switch, studio],
        outputs=[log_output, image_output, studio,
                 panel_prompt, panel_bubble, panel_seed],
    )

    # 预设词条追加（纯文本，不碰显卡）
    append_btn.click(
        fn=append_presets,
        inputs=[panel_prompt, expr_drop, shot_drop, mood_drop],
        outputs=[panel_prompt, log_output],
    )

    # 让 Qwen 补充画面细节词条（只填文本框，不自动出图）
    qwen_ext_btn.click(
        fn=qwen_extend_prompt,
        inputs=[panel_prompt, ollama_drop],
        outputs=[panel_prompt, log_output],
    )

    bubble_btn.click(
        fn=recompose_only,
        inputs=[panel_radio, panel_bubble, studio],
        outputs=[log_output, image_output, studio],
    )

    # ---- 画布式气泡摆位：直接点成品图 ----
    image_output.select(
        fn=place_bubble,
        inputs=[place_mode, studio],
        outputs=[log_output, image_output, studio, panel_radio, *LAY_BOXES],
    )

    apply_btn.click(
        fn=apply_layout,
        inputs=[panel_radio, lay_bx, lay_by, lay_tx, lay_ty, studio],
        outputs=[log_output, image_output, studio],
    )

    spread_btn.click(
        fn=spread_layout,
        inputs=[panel_radio, studio],
        outputs=[log_output, image_output, studio],
    )

    reset_btn.click(
        fn=reset_layout,
        inputs=[studio],
        outputs=[log_output, image_output, studio, *LAY_BOXES],
    )


if __name__ == "__main__":
    print("=" * 70)
    print(" Comic-Auto-Pilot v3.0  启动中……")
    print(f"  · Ollama   : {OLLAMA_GENERATE}")
    print(f"      已安装模型: {', '.join(_OLLAMA_CHOICES) if _OLLAMA_CHOICES else '（拉取失败）'}")
    print(f"  · ComfyUI  : {COMFY_HOST}")
    print(f"  · 工作流图纸: {COMFY_WORKFLOW}  "
          f"[{'已就位' if COMFY_WORKFLOW.is_file() else '缺失 → 请先运行 build_workflow.py'}]")
    print(f"  · 成品目录 : {OUTPUT_DIR}")
    print("=" * 70)

    demo.queue()                     # 生成器式实时日志推送必须开启队列
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
