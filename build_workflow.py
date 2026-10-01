# -*- coding: utf-8 -*-
"""
================================================================================
 build_workflow.py ── 四格条漫 ComfyUI API 图纸生成器
================================================================================

 为什么需要这个脚本（而不是手搓 workflow_api.json）：
   手工在 ComfyUI 里拉 4 条并联链路 = 28 个节点 + 40 多根连线，极易错连；
   而四格漫画的 4 条链路本质上是同一条链的机械复制。
   因此由本脚本按参数程序化生成，保证 4 条链路像素级对称、连线零失误、随时可复现。

 生成的拓扑（全部为 ComfyUI 原生节点，零自定义节点依赖）：

     CheckpointLoaderSimple ─┬─▶ LoraLoader ─┬─▶ CLIPTextEncode × 4 (panel_1~4 正向)
     (illustriousXL_v01)     │  (whale maid)  │        │
                             │                │        ▼
                             │                └─▶ CLIPTextEncode × 1 (共享负向)
                             │                         │
     EmptyLatentImage ───────┼─────────────────────────┼──▶ KSampler × 4
     (832×1216)              │                         │    (25步 CFG8 karras)
                             │                         │         │
                             └─▶ VAEDecode × 4 ◀───────┴─────────┘
                                      │
                                      ▼
                            ImageUpscaleWithModel × 4   ← UpscaleModelLoader
                            (4x-UltraSharp → 3328×4864)   (4x-UltraSharp.pth)
                                      │
                                      ▼
                               ImageScale × 4  (lanczos → 1024×1496)
                                      │
                                      ▼
                   ImageStitch × 3  (direction=down, spacing=20px, white)
                                      │
                                      ▼
                            SaveImage  "Final 1x4 Grid Save"
                                   1024 × 6044

 关于分辨率的工程取舍（重要）：
   直接把 4 张 3328×4864 的 4x 放大图焊接 = 3328×19516 ≈ 2.6 亿像素，
   16G 显存下 PNG 编码会慢到无法接受、成品文件上百 MB。
   因此链路设计为【先用 4x-UltraSharp 真实重构细节，再 lanczos 回落到目标宽度】——
   UltraSharp 的超分重构在高分辨率下真实发生过，细节被保留下来，
   最终落地一个手机/平板可读的 1024×6044 竖排长图。
   要更大只需调下方 PANEL_OUT_W。

 气泡说明：
   本图纸【不含】文字叠加节点 —— 原生 ComfyUI 没有任何可写中文的节点。
   中文对话气泡由 main.py 在本地用 Pillow 绘制（微软雅黑），
   好处是零自定义节点依赖、中文绝不出豆腐块、坐标完全可控。

 用法：
     python build_workflow.py              # 生成 workflow_api.json
     python build_workflow.py --width 1440 # 换成更大的成品宽度
================================================================================
"""

import argparse
import json
from pathlib import Path

# ---- 可调参数（默认值全部抄自用户手搓的「蓝色大肥鱼workflow.json」）-----------
CKPT_NAME     = "illustriousXL_v01.safetensors"
LORA_NAME     = "deepseek_maid_outfit_illustrious_v10.safetensors"
LORA_STRENGTH = 1.0
UPSCALE_MODEL = "4x-UltraSharp.pth"

LATENT_W, LATENT_H = 832, 1216          # SDXL 原生竖构图分辨率
STEPS         = 25
CFG           = 8.0                     # PRD 规定 7.0~8.0
SAMPLER       = "euler_ancestral"
SCHEDULER     = "karras"                # PRD 规定 karras
DENOISE       = 1.0

PANEL_OUT_W   = 1024                    # 每格回落后的宽度（成品长图宽度）
GUTTER_PX     = 20                      # 格间纯白边，必须是偶数（ImageStitch step=2）
STITCH_COLOR  = "white"

NEGATIVE_PROMPT = (
    "worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, error, "
    "missing fingers, extra digits, fewer digits, cropped, jpeg artifacts, signature, "
    "watermark, username, blurry, bad feet."
)
# 占位文本：运行时由 main.py 动态替换为 Ollama 生成的四格 Prompt
PLACEHOLDER = "PLACEHOLDER_WILL_BE_REPLACED_BY_OLLAMA"


def build(panel_out_w: int = PANEL_OUT_W) -> dict:
    panel_out_h = round(panel_out_w * LATENT_H / LATENT_W)   # 保持原始长宽比，防拉伸

    def node(class_type, inputs, title=None):
        n = {"class_type": class_type, "inputs": inputs}
        if title:
            n["_meta"] = {"title": title}
        return n

    g = {}

    # ---------------- 共享的加载器与共享负向提示词 ----------------
    g["1"] = node("CheckpointLoaderSimple", {"ckpt_name": CKPT_NAME}, "底模 Illustrious-XL")
    g["2"] = node("LoraLoader", {
        "model": ["1", 0], "clip": ["1", 1],
        "lora_name": LORA_NAME,
        "strength_model": LORA_STRENGTH, "strength_clip": LORA_STRENGTH,
    }, "LoRA 蓝色大肥鱼女仆")
    g["3"] = node("CLIPTextEncode", {
        "clip": ["2", 1], "text": NEGATIVE_PROMPT,
    }, "Negative 共享负向（勿改名，含 panel 字样会被误认）")
    g["4"] = node("EmptyLatentImage", {
        "width": LATENT_W, "height": LATENT_H, "batch_size": 1,
    }, "空 Latent 832x1216")
    g["5"] = node("UpscaleModelLoader", {"model_name": UPSCALE_MODEL}, "4x-UltraSharp 放大模型")

    # ---------------- 四条并联分镜链路 ----------------
    # 节点 ID 采用 1{i}{step} 规律，使数值排序天然等于 panel 顺序，
    # 即便用户把标题改没了，main.py 的拓扑溯源兜底也能正确对号入座。
    scaled = []
    for i in (1, 2, 3, 4):
        enc, ksa, vae, ups, scl = f"1{i}0", f"1{i}1", f"1{i}2", f"1{i}3", f"1{i}4"

        g[enc] = node("CLIPTextEncode", {
            "clip": ["2", 1], "text": PLACEHOLDER,
        }, f"panel_{i} 正向提示词")

        g[ksa] = node("KSampler", {
            "model": ["2", 0],
            "positive": [enc, 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": 1088840937755405 + i,      # 运行时由 main.py 重新抽卡
            "steps": STEPS, "cfg": CFG,
            "sampler_name": SAMPLER, "scheduler": SCHEDULER,
            "denoise": DENOISE,
        }, f"panel_{i} KSampler")

        g[vae] = node("VAEDecode", {"samples": [ksa, 0], "vae": ["1", 2]},
                      f"panel_{i} VAEDecode")

        g[ups] = node("ImageUpscaleWithModel", {
            "upscale_model": ["5", 0], "image": [vae, 0],
        }, f"panel_{i} 4x-UltraSharp 重构")

        g[scl] = node("ImageScale", {
            "image": [ups, 0], "upscale_method": "lanczos",
            "width": panel_out_w, "height": panel_out_h, "crop": "disabled",
        }, f"panel_{i} 回落 {panel_out_w}x{panel_out_h}")

        scaled.append(scl)

    # ---------------- 四格【各自独立落盘】----------------
    #  为什么不在 ComfyUI 里用 ImageStitch 焊接成一张：
    #    焊接一旦发生在后端，四格就被焊死了 —— 用户只对第 3 格不满意，
    #    也必须把 4 格全部重烧（实测 56s）。
    #  改为四格各自输出、由本地 Pillow 焊接后：
    #    · 单格重出只需 ~11s，其余三格原样复用
    #    · 气泡/台词改动完全不碰显卡，本地秒级重排
    #  代价：1x4 焊接与格边由 main.py 的 ComicComposer 负责（它本来就要画气泡）。
    for i, scl in enumerate(scaled, start=1):
        g[f"3{i:02d}"] = node("SaveImage", {
            "images": [scl, 0], "filename_prefix": f"cap_panel_{i}",
        }, f"panel_{i} Save")

    total_h = panel_out_h * 4 + GUTTER_PX * 3
    print(f"[OK] 图纸构建完成：{len(g)} 个节点，全部为 ComfyUI 原生节点。")
    print(f"     输出方式 : 四格各自独立 SaveImage（焊接交由本地 Pillow，支持单格重出）")
    print(f"     单格渲染 : {LATENT_W}x{LATENT_H}  →  4x-UltraSharp  →  "
          f"{LATENT_W * 4}x{LATENT_H * 4}")
    print(f"     单格回落 : {panel_out_w}x{panel_out_h} (lanczos)")
    print(f"     本地焊接 : {panel_out_w}x{total_h}  （含 3 道 {GUTTER_PX}px 纯白格边）")
    print(f"     采样设定 : {STEPS} 步 / CFG {CFG} / {SAMPLER} / {SCHEDULER}")
    return g


def main():
    ap = argparse.ArgumentParser(description="生成四格条漫 ComfyUI API 图纸")
    ap.add_argument("--width", type=int, default=PANEL_OUT_W,
                    help=f"每格回落后的宽度，默认 {PANEL_OUT_W}")
    ap.add_argument("-o", "--out", default=str(Path(__file__).resolve().parent / "workflow_api.json"),
                    help="输出路径，默认 ./workflow_api.json")
    args = ap.parse_args()

    graph = build(args.width)
    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(graph, fh, ensure_ascii=False, indent=2)
    print(f"[OK] 已写入：{out}")


if __name__ == "__main__":
    main()
