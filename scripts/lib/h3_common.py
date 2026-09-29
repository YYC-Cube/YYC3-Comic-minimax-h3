# -*- coding: utf-8 -*-
"""
@file h3_common.py
@author YanYuCloudCube Team <admin@0379.email>
@version v1.1.0
@created 2026-09-02
@updated 2026-09-03
@status stable
@copyright Copyright (c) 2025-2026 YYC3 Team
@license MIT


h3_common.py — MiniMax-H3 生产线共享库（Phase 1 · T1.1/T1.3）
来源任务：docs/04-演进规划与闭环优化机制.md

提供三大能力：
1. VramConfig / load_pipeline：M4 Max 统一模型加载入口（NF4/Pruned × FL2VA/Ref2VA）
2. Manifest：批次 manifest.json 单一事实源（schema_version=1）
3. PerformanceTimer：耗时 + 内存峰值采集（RSS / MPS 已分配显存）

manifest.json 结构：
{
  "schema_version": 1,
  "batch": "01",
  "model": {"variant": "nf4", "pipeline": "ref2va"},
  "params": {"height": 480, "width": 832, "num_frames": 124, "num_inference_steps": 50, "prompt": "..."},
  "started_at": "ISO8601",
  "ended_at": "ISO8601|null",
  "records": [
    {
      "ref_img": "person_a.jpg", "seed": 42, "status": "SUCCESS",
      "video_path": "output_batch01/person_a/h3_seed_42.mp4", "time": "14:30:21",
      "gen_seconds": 321.5, "peak_rss_gb": 41.2, "mps_alloc_gb": 12.3,
      "lipsync": {"backend": "syncnet", "confidence": 8.42, "av_offset": 3,
                  "score_norm": 0.63, "scored_at": "ISO8601"} | null,
      "human": {"score": null, "tags": ""}
    }
  ]
}
"""
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ============================================================
# 模型加载（T1.3：收敛 19 个脚本的重复 vram_config/from_pretrained）
# ============================================================

MODEL_ID_NF4 = "DiffSynth-Studio/MiniMax-H3-NF4"
MODEL_ID_PRUNED = "DiffSynth-Studio/MiniMax-H3-Pruned"
PROCESSOR_ID = "MiniMaxAI/MiniMax-H3"
# 本地权重根目录：存在 <root>/<model_id短名>/ 时直接用本地文件，避免重复下载
LOCAL_WEIGHTS_ROOT = os.environ.get("H3_WEIGHTS_DIR", "/Users/yanyu/models")


def m4_max_vram_config():
    """M4 Max 128GB 优化配置：CPU offload + MPS 计算（注意：device 必须是 torch.device 对象）"""
    import torch  # type: ignore
    return {
        "offload_dtype": torch.float32,
        "offload_device": torch.device("cpu"),
        "onload_dtype": torch.bfloat16,
        "onload_device": torch.device("mps"),
        "preparing_dtype": torch.bfloat16,
        "preparing_device": torch.device("mps"),
        "computation_dtype": torch.bfloat16,
        "computation_device": torch.device("mps"),
    }


def gpu_vram_config():
    """按平台自适应：CUDA（DGX GB10 统一内存）优先，无 CUDA 回退 M4 Max MPS 配置"""
    import torch  # type: ignore
    if torch.cuda.is_available():
        return {
            "offload_dtype": torch.float32,
            "offload_device": torch.device("cpu"),
            "onload_dtype": torch.bfloat16,
            "onload_device": torch.device("cuda"),
            "preparing_dtype": torch.bfloat16,
            "preparing_device": torch.device("cuda"),
            "computation_dtype": torch.bfloat16,
            "computation_device": torch.device("cuda"),
        }
    return m4_max_vram_config()


def weight_files(variant: str, pipeline: str):
    """按 variant(nf4|pruned) 和 pipeline(fl2va|ref2va) 返回权重清单

    pruned 实况（2026-09-16 磁盘核对）：Pruned 仅 DiT 有剪枝版（AdaLN 分支剪枝 +
    adaln_t_table 查表，DiffSynth 按 model_hash 自动映射 MiniMaxH3DiTComfyPruned）；
    text-encoder / video_vae / audio_vae 无剪枝版，复用 nf4 全量权重。
    """
    if variant == "pruned":
        return [
            f"minimax-h3-{pipeline}-pruned-nf4.safetensors",  # 剪枝 DiT（pruned+nf4 混合量化）
            "minimax-h3-text-encoder-nf4.safetensors",
            "video_vae_nf4.safetensors",
            "audio_vae_nf4.safetensors",
        ]
    return [
        f"minimax-h3-{pipeline}-nf4.safetensors",
        "minimax-h3-text-encoder-nf4.safetensors",
        "video_vae_nf4.safetensors",
        "audio_vae_nf4.safetensors",
    ]


def _model_config(variant: str, files: list):
    """本地权重目录存在 → ModelConfig(path=具体文件)（跳过下载）；否则走 model_id 在线下载"""
    from diffsynth.pipelines.minimax_h3_audio_video import ModelConfig  # type: ignore
    vc = gpu_vram_config()
    model_id = MODEL_ID_NF4 if variant == "nf4" else MODEL_ID_PRUNED
    local_dir = Path(LOCAL_WEIGHTS_ROOT) / "MiniMax-H3-NF4"
    if (local_dir / files[0]).exists():
        return [ModelConfig(path=str(local_dir / f), **vc) for f in files]
    return [ModelConfig(model_id=model_id, origin_file_pattern=f, **vc) for f in files]


def load_pipeline(variant: str = "nf4", pipeline: str = "ref2va", vram_limit: int = 96):
    """统一模型加载入口。variant: nf4|pruned；pipeline: fl2va|ref2va"""
    import torch  # type: ignore
    from diffsynth.pipelines.minimax_h3_audio_video import (  # type: ignore
        MiniMaxH3Pipeline, ModelConfig)

    # processor：本地已下载则直接指向目录（避免在线下载），否则走 model_id
    local_proc = Path(LOCAL_WEIGHTS_ROOT) / "MiniMax-H3" / pipeline.upper() / "processor"
    proc_cfg = (ModelConfig(path=str(local_proc)) if (local_proc / "preprocessor_config.json").exists()
                else ModelConfig(model_id=PROCESSOR_ID, origin_file_pattern=f"{pipeline.upper()}/processor/"))
    return MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "mps",
        model_configs=_model_config(variant, weight_files(variant, pipeline)),
        processor_config=proc_cfg,
        vram_limit=vram_limit,
    )


# ============================================================
# 性能采集（T3.1 基线能力，随 T1.1 一并落地）
# ============================================================

class PerformanceTimer:
    """耗时 + 内存峰值。用法：with PerformanceTimer() as t: ...

    注意：peak_rss_gb 取 resource.getrusage(RUSAGE_SELF).ru_maxrss，
    是「本进程自启动以来的 RSS 单调水印」，非本段代码的独立占用——
    同一进程内串行多 seed 时各 seed 读数只会持平或递增（勿当 per-seed 值解读）。
    """

    def __init__(self):
        self.start: float = 0.0
        self.seconds: float = 0.0
        self.peak_rss_gb: float = 0.0
        self.mps_alloc_gb: float | None = None
        self.cuda_alloc_gb: float | None = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = round(time.perf_counter() - self.start, 3)
        # ru_maxrss 单位双平台异义（getrusage(2)）：macOS=字节，Linux=KB
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        self.peak_rss_gb = round(rss_bytes / (1024 ** 3), 3)
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                self.cuda_alloc_gb = round(float(torch.cuda.memory_allocated()) / (1024 ** 3), 2)
            elif torch.backends.mps.is_available():
                self.mps_alloc_gb = round(float(torch.mps.current_allocated_memory()) / (1024 ** 3), 2)
        except Exception:
            pass
        return False


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ============================================================
# Manifest（T1.1：批次单一事实源）
# ============================================================

class Manifest:
    def __init__(self, path, batch="01", variant="nf4", pipeline="ref2va", params=None):
        self.path = Path(path)
        self.data = {
            "schema_version": 1,
            "batch": str(batch),
            "model": {"variant": variant, "pipeline": pipeline},
            "params": params or {},
            "started_at": now_iso(),
            "ended_at": None,
            "records": [],
        }

    # ---------- 基础IO ----------
    @classmethod
    def load(cls, path):
        m = cls.__new__(cls)
        m.path = Path(path)
        m.data = json.loads(m.path.read_text(encoding="utf-8"))
        return m

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # 原子写，断电不损坏

    @property
    def records(self):
        return self.data["records"]

    # ---------- 记录操作 ----------
    def add_record(self, ref_img, seed, status, video_path, **perf):
        rec = {
            "ref_img": ref_img,
            "seed": seed,
            "status": status,  # SUCCESS | FAILED | SKIPPED | READ_FAILED
            "video_path": video_path,
            "time": now_hms(),
            "gen_seconds": perf.get("gen_seconds"),
            "peak_rss_gb": perf.get("peak_rss_gb"),
            "mps_alloc_gb": perf.get("mps_alloc_gb"),
            "lipsync": None,   # 由 score_lipsync.py 回填
            "human": {"score": None, "tags": ""},  # 人工打分回填
        }
        self.records.append(rec)
        return rec

    def find(self, ref_img, seed):
        for r in self.records:
            if r["ref_img"] == ref_img and str(r["seed"]) == str(seed):
                return r
        return None

    def set_lipsync(self, ref_img, seed, backend, confidence, av_offset, score_norm):
        rec = self.find(ref_img, seed)
        if rec:
            rec["lipsync"] = {
                "backend": backend,
                "confidence": confidence,
                "av_offset": av_offset,
                "score_norm": score_norm,
                "scored_at": now_iso(),
            }

    def set_human(self, ref_img, seed, score=None, tags=None):
        rec = self.find(ref_img, seed)
        if rec:
            if score is not None:
                rec["human"]["score"] = score
            if tags is not None:
                rec["human"]["tags"] = tags

    def finish(self):
        self.data["ended_at"] = now_iso()
        self.save()

    # ---------- 导出（兼容 analyze / 面板） ----------
    def flat_rows(self):
        """展平为行记录，lipsync/human 拆列，供 analyze 直接消费"""
        for r in self.records:
            lip = r.get("lipsync") or {}
            hu = r.get("human") or {}
            yield {
                "ref_img": r["ref_img"],
                "seed": r["seed"],
                "status": r["status"],
                "video_path": r["video_path"],
                "time": r["time"],
                "lipsync_score": lip.get("score_norm"),
                "lipsync_confidence": lip.get("confidence"),
                "lipsync_backend": lip.get("backend"),
                "gen_seconds": r.get("gen_seconds"),
                "score": hu.get("score"),
                "tags": hu.get("tags") or "",
            }


# ============================================================
# 报告表格（report_batchXX.md，新增「口型分」列，与 manifest 同步）
# ============================================================

REPORT_HEADER = "| 参考图 | Seed | 状态 | 视频相对路径 | 时间 | 口型分 | 评分(1~10) | 缺陷标签 |"
REPORT_SEP = "|--------|------|------|--------------|------|--------|------------|----------|"


def init_report(md_path: Path, title: str, ref_images_dir, seed_list, output_root_dir):
    if not md_path.exists():
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"任务启动：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"参考图目录：`{ref_images_dir}`\n")
            f.write(f"视频输出目录：`{output_root_dir}`\n")
            f.write(f"Seed列表：{seed_list}\n\n")
            f.write(REPORT_HEADER + "\n")
            f.write(REPORT_SEP + "\n")


def report_row(ref_img, seed, status, video_path, lipsync="-"):
    """lipsync: score_norm(0~1) 或 '-'"""
    return f"| {ref_img} | {seed} | {status} | `{video_path}` | {now_hms()} | {lipsync} |  |  |\n"


# ============================================================
# 启发式音画同步代理分（score_lipsync 的降级后端）
# ============================================================

def extract_audio_wav(video_path: Path, wav_path: Path, sr: int = 16000) -> bool:
    """用 ffmpeg 提取单声道 wav。失败返回 False（原因打 stderr，不静默）。"""
    import shutil
    import sys as _sys
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        # 09-28 事故③：cron PATH 无 homebrew 时 ffmpeg 不可达，评分链曾全程静默
        print("⚠️ extract_audio_wav：PATH 中找不到 ffmpeg（cron 场景见 nightly_run.sh "
              "顶部 PATH 导出）", file=_sys.stderr)
        return False
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(video_path),
             "-ac", "1", "-ar", str(sr), "-vn", str(wav_path)],
            check=True, capture_output=True,
        )
        return wav_path.exists()
    except subprocess.CalledProcessError as e:
        print(f"⚠️ extract_audio_wav 失败：{video_path.name} → "
              f"{(e.stderr or b'')[-200:]!r}", file=_sys.stderr)
        return False
    except Exception:
        return False


def heuristic_sync_score(video_path: Path, work_dir: Path) -> dict:
    """
    无SyncNet权重时的降级方案：
    音频RMS能量包络 vs 视频口型区运动能量 的分段相关系数 → 归一化到 0~1。
    返回 {"backend": "heuristic", "confidence": corr, "av_offset": 0, "score_norm": x}
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import wave

    video_path = Path(video_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir / f"{video_path.stem}_audio.wav"

    if not extract_audio_wav(video_path, wav_path):
        return {"backend": "heuristic", "confidence": None, "av_offset": 0, "score_norm": None}

    # 1) 音频RMS包络
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        audio = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    dur = n / max(sr, 1)
    if dur < 1.0:
        return {"backend": "heuristic", "confidence": None, "av_offset": 0, "score_norm": None}

    # 2) 口型区运动能量（画面中下部裁剪帧行人代表区域）
    cap = cv2.VideoCapture(str(video_path))
    frame_energies = []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w_ = frame.shape[:2]
        mouth = frame[int(h * 0.55):int(h * 0.90), int(w_ * 0.25):int(w_ * 0.75)]
        gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            frame_energies.append(float(np.mean(np.abs(gray - prev))))
        prev = gray
    cap.release()
    if len(frame_energies) < 10:
        return {"backend": "heuristic", "confidence": None, "av_offset": 0, "score_norm": None}

    # 3) 帧级对齐：两路信号都转成「活跃度」事件轨道，用事件级最近邻匹配评分。
    #    根因笔记：
    #    - Pearson相关在稀疏二值脉冲上即使完全重合也很低（大量双零帧压低协方差）；
    #    - AAC编码会让音轨时长轻微漂移（实测2s内容→2.05s），后段事件累计偏移1帧，
    #      固定滞后的全局Jaccard也无法吸收这种渐进漂移。
    #    解法：事件集合的最近邻匹配（事件|偏差|<=tol即算命中），对渐进漂移鲁棒。
    n_frames = len(frame_energies) + 1
    # 音频插值到帧级，取逐帧能量（差分）→ 活跃度（发声 onset/offset 事件）
    a_frame = np.interp(np.linspace(0, len(audio) - 1, n_frames), np.arange(len(audio)), audio)
    a_act_raw = np.abs(np.diff(a_frame))
    a_thr = max(0.3 * a_act_raw.std(), 1e-4)
    a_active = (a_act_raw > a_thr).astype(np.float32)

    # 视频能量 → 活跃度（口型运动事件）
    v_e = np.array(frame_energies, dtype=np.float32)
    v_thr = max(0.3 * v_e.std(), 1e-4)
    v_active = (v_e > v_thr).astype(np.float32)

    k = min(len(a_active), len(v_active))
    if k < 8 or a_active.std() < 1e-6 or v_active.std() < 1e-6:
        return {"backend": "heuristic", "confidence": None, "av_offset": 0, "score_norm": None}

    tol = 1  # 事件匹配容差（帧）
    a_events = [i for i in range(k) if a_active[i] > 0]
    v_events = [i for i in range(k) if v_active[i] > 0]
    if not a_events or not v_events:
        return {"backend": "heuristic", "confidence": None, "av_offset": 0, "score_norm": None}

    # 事件级最近邻匹配：任一方向被tol内最近邻命中即算匹配
    matched_a = sum(1 for i in a_events if any(abs(i - j) <= tol for j in v_events))
    matched_v = sum(1 for j in v_events if any(abs(j - i) <= tol for i in a_events))
    total_events = len(a_events) + len(v_events)
    corr = (matched_a + matched_v) / total_events if total_events else 0.0
    best_lag = 0

    return {
        "backend": "heuristic",
        "confidence": round(corr, 4),
        "av_offset": best_lag,
        "score_norm": round((corr + 1) / 2, 4),  # [-1,1] → [0,1]
    }


def syncnet_score(video_path: Path, work_dir: Path) -> Optional[dict]:
    """
    SyncNet 后端（优先）：官方 syncnet_python 仓入口直调。
    score_norm = conf / (abs(conf) + 5)，conf≈6-8（Ref2VA 实测口径）→ 0.55-0.62
    不可用时返回 None，由调用方降级 heuristic。

    实现说明（2026-09-29 定版）：此前为 aspirational 的 syncnet_pipeline
    封装（`syncnet_python.syncnet_pipeline` 模块从未落地，一调即 ImportError
    降级）→ 重写为 subprocess 直调 run_pipeline.py → run_syncnet.py，
    与 TC-G4-002 及 DGX 三档评分实证同款链路（诚实留证原则）。
    """
    return syncnet_score_impl(video_path, work_dir)


def syncnet_score_impl(video_path: Path, work_dir: Path) -> Optional[dict]:
    """run_pipeline（检测/跟踪/裁剪）→ run_syncnet（评估置信度）。

    环境变量：
        H3_SYNCNET_DIR  syncnet_python 仓根目录（默认按平台：
                        mac ~/YYC-Cube/tools/syncnet/syncnet_python，
                        Linux ~/tools/syncnet/syncnet_python）
        H3_SYNCNET_PY   评分解释器（默认 sys.executable；需含 scenedetect/
                        insightface/python_speech_features 等依赖）
    短视频口径：--min_track 40 --min_face_size 60——默认 100 帧轨长门槛对
    <100 帧短视频必空轨（TC-G4-002 实测根因）；本项目半身像脸宽 ~55px
    同理需降 min_face_size。
    任一步失败（仓缺失/零轨/解析失败/超时）返回 None，由调用方降级。
    """
    default_dir = Path.home() / (
        "YYC-Cube/tools/syncnet/syncnet_python" if sys.platform == "darwin"
        else "tools/syncnet/syncnet_python")
    root = Path(os.environ.get("H3_SYNCNET_DIR") or default_dir).expanduser()
    if not (root / "run_pipeline.py").exists():
        return None
    # 解释器候选链：环境变量 > ComfyUI venv（G4-004 实证依赖齐备）>
    # mac h3 仓 venv（含 torch 但缺 cv2，备选）> 当前解释器
    candidates = [os.environ.get("H3_SYNCNET_PY")]
    if sys.platform == "darwin":
        candidates.append(str(Path.home() / "YYC-Cube/tools/ComfyUI/.venv/bin/python"))
        candidates.append(str(Path.home() / "YYC-Cube/YYC3-MiniMax-H3/.venv/bin/python"))
    candidates.append(sys.executable)
    py = next((c for c in candidates if c and Path(c).exists()), None)
    if not py:
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    reference = video_path.stem  # 任务标识符：run_pipeline 按此建 data_dir 子目录
    try:
        p1 = subprocess.run(
            [py, "run_pipeline.py", "--videofile", str(video_path),
             "--reference", reference, "--data_dir", str(work_dir),
             "--min_track", "40", "--min_face_size", "60", "--overwrite"],
            cwd=str(root), capture_output=True, text=True, timeout=900)
        if p1.returncode != 0 or not (work_dir / "pycrop" / reference).exists():
            return None
        p2 = subprocess.run(
            [py, "run_syncnet.py", "--data_dir", str(work_dir),
             "--videofile", str(video_path), "--reference", reference],
            cwd=str(root), capture_output=True, text=True, timeout=900)
        if p2.returncode != 0:
            return None
        # SyncNetInstance 经 logging 输出（默认 stderr），合并双流解析
        # 注意：行首含时间戳「18:53:25,480」——必须 rsplit 最后一个冒号，
        # 否则切在时间戳上 float 抛 ValueError（冒烟实测踩坑）
        text = (p2.stdout or "") + (p2.stderr or "")
        conf = off = dist = None
        for line in text.splitlines():
            if "Confidence:" in line:
                conf = float(line.rsplit(":", 1)[1].strip())
            elif "AV offset:" in line:
                off = float(line.rsplit(":", 1)[1].strip())
            elif "Min dist:" in line:
                dist = float(line.rsplit(":", 1)[1].strip())
        if conf is None:
            return None
        return {
            "backend": "syncnet",
            "confidence": round(conf, 4),
            "av_offset": int(off) if off is not None else 0,
            "min_dist": round(dist, 3) if dist is not None else None,
            "score_norm": round(conf / (abs(conf) + 5.0), 4),
        }
    except Exception:
        return None
