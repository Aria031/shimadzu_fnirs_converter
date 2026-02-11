from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import math
from datetime import datetime
from typing import (Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union)

import numpy as np
import pandas as pd
import h5py


# mne 相关（可能在你的环境是 mne 或 mne_nirs）
try:
    import mne
except Exception as e:
    raise RuntimeError("未检测到 mne，请先安装 mne (pip install mne mne-nirs)") from e

from mne.transforms import get_ras_to_neuromag_trans, apply_trans




def _h5_str(x) -> np.bytes_:
    if x is None:
        x = ""
    if isinstance(x, bytes):
        return np.bytes_(x)
    return np.bytes_(str(x).encode("utf-8"))


# -------------------------
# Logger 配置（可通过 CLI 调整等级）
# -------------------------
logger = logging.getLogger("snirf_converter")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
_fmt = "%(asctime)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(_fmt)
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)

# -------------------------
# 常量与单位换算
# -------------------------
UNIT_SCALE_FACTORS: Dict[str, float] = {
    "mm": 1.0 / 1000.0,
    "millimeter": 1.0 / 1000.0,
    "millimeters": 1.0 / 1000.0,
    "cm": 1.0 / 100.0,
    "centimeter": 1.0 / 100.0,
    "centimeters": 1.0 / 100.0,
    "m": 1.0,
    "meter": 1.0
}

# -------------------------
# 辅助/通用函数
# -------------------------
def ensure_file(path: str, what: str = "file") -> None:
    """检查文件是否存在，若不存在则抛错。"""
    if not path:
        raise ValueError(f"{what} 路径为空")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} 不存在: {path}")


def safe_float(x: Any, default: float = float("nan")) -> float:
    """尝试将 x 转为 float，失败返回 default（通常是 np.nan）。"""
    try:
        return float(x)
    except Exception:
        return default


def get_scale(unit: str) -> float:
    """返回给定单位到米的缩放因子（单位字符串不区分大小写）。"""
    if not unit:
        return 1.0
    unit = str(unit).strip().lower()
    return UNIT_SCALE_FACTORS.get(unit, 1.0)


def find_column_by_prefix(columns: Sequence[str], prefix: str) -> Optional[str]:
    """
    在列名序列中查找以 prefix 开头的列（不区分大小写），返回第一个匹配列名或 None。
    prefix 比如 'x'、'X'、'Label' 等。
    """
    for c in columns:
        if c is None:
            continue
        try:
            if str(c).lower().startswith(prefix.lower()):
                return c
        except Exception:
            continue
    return None


def normalize_label(label: str) -> str:
    """对标签做标准化（去首尾空格并统一大小写），用于匹配。"""
    if label is None:
        return ""
    return str(label).strip()


# -------------------------
# 1) 解析 Shimadzu TXT
# -------------------------
def parse_shimadzu_txt(txt_path: str, max_header_lines: int = 400) -> Tuple[
    np.ndarray,                  # time (N,)
    np.ndarray,                  # data_matrix (N, n_sigcols)
    List[str],                   # raw_task (N,)
    List[str],                   # raw_mark (N,)
    List[str],                   # raw_count (N,)
    List[Tuple[int, int]]        # channel_pairs ([(s,d), ...]) 供后续测量映射使用
]:
    """
    解析岛津导出的 TXT 文件（文本格式，带头部）。返回：
      - time: shape (N_time,)
      - data_matrix: shape (N_time, n_sigcols)  # n_sigcols = n_channels * 3 (oxy,deoxy,total)
      - raw_task, raw_mark, raw_count 为原始列内容的字符串数组（便于调试）
      - channel_pairs: header中解析出的 source-detector pair 列表（若 header 含有 (s,d) 对）；若未提供则自动生成占位对
    要点：
      - 稳健查找 header 行（匹配 'Time'、'oxyHb' 等关键词）
      - 解析数据区块（从第一个以数字开头的行开始）
      - 容忍 Task/Mark 列中的非数字（例如 '0Z'）
      - 若数据列数必须是3的倍数，会抛出友好错误或警告
    """
    ensure_file(txt_path, "Shimadzu TXT 文件")
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 1) 在前 max_header_lines 行中查找 header 行（包含 Time & Hb 关键字）
    header_idx = None
    for i, L in enumerate(lines[:max_header_lines]):
        if re.search(r'\bTime\b', L, flags=re.I) and re.search(r'oxy|deoxy|hb', L, flags=re.I):
            header_idx = i
            break

    # 备用查找策略：查找 'Time(' 或 'Time(sec)'
    if header_idx is None:
        for i, L in enumerate(lines[:max_header_lines]):
            if 'Time(' in L or 'Time' in L and 'sec' in L.lower():
                header_idx = i
                break

    # 最后备用：如果都没找到，从前 max_header_lines 中找第一行第一个 token 可以解析为 float 的行（该行前一行为 header）
    if header_idx is None:
        for i, L in enumerate(lines[:max_header_lines]):
            toks = re.split(r'\s+', L.strip())
            if len(toks) == 0:
                continue
            try:
                float(toks[0])
                header_idx = max(0, i - 1)
                logger.warning("未显式找到含 'Time' 的 header 行，采用近似策略定位 header。")
                break
            except Exception:
                continue

    if header_idx is None:
        raise RuntimeError("无法定位 TXT 文件中的 header（未找到 'Time' 行或首个数字行）。请检查文件格式。")

    logger.info(f"定位到 header 行 (索引从0计)：{header_idx}；内容示例：{lines[header_idx].strip()[:120]}")

    # 2) 尝试从 header blob 中提取 channel pair（格式如 (1,1)(2,1)）
    header_blob = "".join(lines[:header_idx + 1])
    pairs = re.findall(r'\((\d+)\s*,\s*(\d+)\)', header_blob)
    channel_pairs: Optional[List[Tuple[int, int]]] = None
    if pairs:
        channel_pairs = [(int(s), int(d)) for s, d in pairs]
        logger.info(f"在 header 中解析到 {len(channel_pairs)} 个 (source,detector) 对（示例前 5 个）：{channel_pairs[:5]}")
    else:
        logger.debug("header 未找到 (s,d) 对的信息。")

    # 3) 找到数据行起点（header 后第一个以数字开头的行）
    data_start_idx = None
    for j in range(header_idx + 1, min(len(lines), header_idx + max_header_lines + 1)):
        toks = re.split(r'\s+', lines[j].strip())
        if len(toks) < 4:
            continue
        try:
            float(toks[0])
            data_start_idx = j
            break
        except Exception:
            continue

    if data_start_idx is None:
        raise RuntimeError("未能在 header 后找到以时间（数字）开头的数据行。请确认 TXT 文件内容。")

    logger.info(f"数据起始行索引 (0-based) = {data_start_idx}; 行示例：{lines[data_start_idx].strip()[:200]}")

    # 4) 解析数据区
    time_list: List[float] = []
    raw_task: List[str] = []
    raw_mark: List[str] = []
    raw_count: List[str] = []
    sig_rows: List[List[float]] = []

    for idx in range(data_start_idx, len(lines)):
        line = lines[idx].strip()
        if not line:
            continue
        toks = re.split(r'\s+', line)
        # 至少要有 Time, Task, Mark, Count + 一列 signal
        if len(toks) < 5:
            continue
        # 首 token 应为时间
        try:
            t = float(toks[0])
        except Exception:
            # 如果不是数字，跳过（可能是 footer 或注释）
            continue
        time_list.append(t)
        raw_task.append(toks[1] if len(toks) > 1 else "")
        raw_mark.append(toks[2] if len(toks) > 2 else "")
        raw_count.append(toks[3] if len(toks) > 3 else "")

        # 其余 tokens 视为信号列：若无法转为 float，设为 nan
        row_vals: List[float] = []
        for tok in toks[4:]:
            try:
                row_vals.append(float(tok))
            except Exception:
                row_vals.append(float("nan"))
        sig_rows.append(row_vals)

    if len(sig_rows) == 0:
        raise RuntimeError("从 TXT 中未解析到任何数据行（signal_rows 为空）。请确认文件。")

    data_matrix = np.array(sig_rows, dtype=float)  # shape (N_time, n_sigcols)
    time_arr = np.array(time_list, dtype=float)
    logger.info(f"解析到时间点数: {len(time_arr)}, 信号列数: {data_matrix.shape[1]}")

    # 5) 校验信号列是否为 channels * 3 (oxy,deoxy,total)
    n_sigcols = data_matrix.shape[1]
    if n_sigcols % 3 != 0:
        logger.warning(f"信号列数 {n_sigcols} 不是 3 的整数倍，请确认数据列。(仍继续，但后续映射可能不准确)")
    n_channels_inferred = n_sigcols // 3

    # 如果 header 中有 channel_pairs，但与实际列数不一致则抛出警告并忽略 header pairs
    if channel_pairs is not None:
        if len(channel_pairs) != n_channels_inferred:
            logger.warning(
                f"header 中解析到 {len(channel_pairs)} 个 channel pairs，但数据列对应 {n_channels_inferred} 个通道。"
                "将忽略 header pairs 并自动生成占位 pairs。")
            channel_pairs = None

    # 如果未找到 channel_pairs，则生成占位 (1..n_channels) 的 (s,d) 对
    if channel_pairs is None:
        channel_pairs = [(i + 1, i + 1) for i in range(n_channels_inferred)]
        logger.info(f"自动生成占位 channel_pairs，count={len(channel_pairs)} (示例前5): {channel_pairs[:5]}")

    # 返回：time, data_matrix, task, mark, count, channel_pairs
    return time_arr, data_matrix, raw_task, raw_mark, raw_count, channel_pairs

# -------------------------
# 2) 解析 origin (4 个点) 和 others 文件
# -------------------------
def parse_coords_file(coord_path: str, unit: str = "mm", scale_to_m: bool = True) -> Dict[str, np.ndarray]:
    """
    通用坐标文件解析器：支持 header 或无 header，分隔符支持逗号/空白。
    返回 dict: label -> np.array([x,y,z])，若 scale_to_m=True 则转换为米。
    预期：
        Label,X,Y,Z
      或:
        T1 14.47 7.09 4.87
    """
    ensure_file(coord_path, "坐标文件")
    logger.info(f"解析坐标文件: {coord_path}")

    # 统一按“无表头”读取，避免第一行被吃掉（你之前的 T1 就是这么丢的）
    try:
        df = pd.read_csv(coord_path, sep=None, engine="python", header=None)
    except Exception:
        df = pd.read_csv(coord_path, sep=r'[\s,]+', engine="python", header=None)

    if df.shape[0] == 0 or df.shape[1] < 4:
        raise RuntimeError("坐标文件至少需要 4 列（Label, X, Y, Z）且不能是空文件。")

    # ---- 判断第一行是否是表头：两种策略 ----
    first_row = df.iloc[0].astype(str).str.strip().str.lower().tolist()

    def _can_float(x: Any) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    looks_like_header = False

    # 1) 语义判断：出现 label/name/id 或 x/y/z
    if any(("label" in x) or ("name" in x) or ("id" in x) for x in first_row):
        looks_like_header = True
    if (any(x.startswith("x") for x in first_row) and
        any(x.startswith("y") for x in first_row) and
        any(x.startswith("z") for x in first_row)):
        looks_like_header = True

    # 2) 数值判断：如果第 2-4 列都不能转 float，也基本是表头
    if not (_can_float(df.iloc[0, 1]) and _can_float(df.iloc[0, 2]) and _can_float(df.iloc[0, 3])):
        # 注意：这里不直接覆盖语义判断，而是作为补强
        # 避免极端情况第一行 label 是 "T1" 但坐标是空字符串
        looks_like_header = True

    if looks_like_header:
        df = df.iloc[1:].reset_index(drop=True)

    coords: Dict[str, np.ndarray] = {}

    arr = df.values
    for row in arr:
        try:
            lab = normalize_label(row[0])
            x = safe_float(row[1], float("nan"))
            y = safe_float(row[2], float("nan"))
            z = safe_float(row[3], float("nan"))
            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                continue
            coords[lab] = np.array([x, y, z], dtype=float)
        except Exception:
            continue

    # 单位转换
    if scale_to_m:
        sf = get_scale(unit)
        for k in list(coords.keys()):
            coords[k] = coords[k] * sf

    logger.info(f"解析到坐标点数量: {len(coords)} (示例 keys 前 10): {list(coords.keys())[:10]}")
    return coords


# -------------------------
# 3) 解析 events CSV（onset,duration,value,condition）
# -------------------------
def parse_events_csv(events_path: str) -> pd.DataFrame:
    """
    读取事件 CSV 并返回 DataFrame。要求至少包含 onset、duration、value 列。
    如果包含 condition 列，会使用它作为描述字段。
    """
    ensure_file(events_path, "事件文件")
    try:
        df = pd.read_csv(events_path)
    except Exception as e:
        raise RuntimeError(f"无法读取事件文件 {events_path}: {e}")

    req_cols = {'onset', 'duration', 'value'}
    if not req_cols.issubset(set(df.columns)):
        raise RuntimeError(f"事件 CSV 必须包含列 {req_cols}，当前文件列为: {list(df.columns)}")

    # 规范化列类型
    df['onset'] = df['onset'].astype(float)
    df['duration'] = df['duration'].astype(float)
    # value 可能是 int 或 str，但我们保留原样
    if 'condition' in df.columns:
        df['condition'] = df['condition'].astype(str)
    else:
        # 当 condition 缺失时使用 value 作为 condition（字符串形式）
        df['condition'] = df['value'].astype(str)
        logger.warning("事件 CSV 中未包含 'condition' 列，已用 'value' 填充 condition 字段（字符串形式）。")

    logger.info(f"解析到事件数: {len(df)}，条件种类: {df['condition'].nunique()}")
    return df

def infer_sfreq_from_time(time_arr: np.ndarray) -> float:
    """从 time 列推断采样率：sfreq ≈ 1 / median(diff(time))."""
    t = np.asarray(time_arr, dtype=float)
    if t.ndim != 1 or t.size < 3:
        raise ValueError("time_arr 太短，无法推断 sfreq。")
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if dt.size < 2:
        raise ValueError("time_arr 的差分无有效正值，无法推断 sfreq。")

    med = float(np.median(dt))
    sfreq = 1.0 / med if med > 0 else float("nan")

    # 简单 sanity：如果 time 本来就是以“采样点序号”写的，会导致 dt≈1 -> sfreq≈1Hz
    # 你也可以在这里加更强的检查
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"推断得到非法 sfreq={sfreq}")

    return sfreq


def parse_code_maybe(x: Any) -> int:
    """
    从 Task/Mark 字符串解析 code：
    - "0" -> 0
    - "01" -> 1
    - "8Z" -> 8
    - ""/None/非数字开头 -> 0
    """
    if x is None:
        return 0
    s = str(x).strip()
    if s == "":
        return 0
    m = re.match(r"^\s*([+-]?\d+)", s)  # 取开头的整数部分
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def infer_events_from_task_mark(
    time_arr: np.ndarray,
    raw_task: List[str],
    raw_mark: List[str],
    prefer: str = "auto",            # "auto"|"mark"|"task"
    min_code: int = 1,               # 忽略 < min_code 的码（通常忽略0）
    duration_mode: str = "segment",  # "segment"|"zero"
    min_gap_s: float = 0.0,          # 相邻片段合并阈值（秒）
) -> pd.DataFrame:
    """
    从 TXT 的 Task/Mark 推断 events：
    - 先把 Task/Mark 解析成整数 code 序列
    - 找到 rising edge: code 从 0 -> 非0 的时刻为 onset
    - duration：
        - segment: 如果 code 连续保持非0，duration=持续时长；如果都是单点脉冲则 duration=0
        - zero: 全部 duration=0
    返回 DataFrame: onset, duration, value, condition
    """
    t = np.asarray(time_arr, dtype=float)
    if t.size == 0:
        return pd.DataFrame(columns=["onset", "duration", "value", "condition"])

    task_codes = np.array([parse_code_maybe(x) for x in raw_task], dtype=int)
    mark_codes = np.array([parse_code_maybe(x) for x in raw_mark], dtype=int)

    # 选择使用 mark/task
    use = prefer.lower().strip()
    if use not in ("auto", "mark", "task"):
        use = "auto"

    if use == "mark":
        codes = mark_codes
        src = "Mark"
    elif use == "task":
        codes = task_codes
        src = "Task"
    else:
        # auto: Mark 有非0就用 Mark，否则用 Task
        if np.any(mark_codes >= min_code):
            codes = mark_codes
            src = "Mark"
        else:
            codes = task_codes
            src = "Task"

    logger.info(f"[events-auto] prefer={prefer} -> using {src}; unique codes={sorted(set(codes.tolist()))[:20]}")

    # 把 < min_code 的都当作 0
    codes = codes.copy()
    codes[codes < min_code] = 0

    # 找 rising edge: prev==0 & curr!=0
    prev = np.r_[0, codes[:-1]]
    rising_idx = np.where((prev == 0) & (codes != 0))[0]

    if rising_idx.size == 0:
        return pd.DataFrame(columns=["onset", "duration", "value", "condition"])

    # segment 提取：把连续非0 的区间当片段
    segments = []
    in_seg = False
    seg_start = None
    seg_code = None

    for i in range(codes.size):
        c = codes[i]
        if not in_seg and c != 0:
            in_seg = True
            seg_start = i
            seg_code = c
        elif in_seg:
            # 片段结束：回到0，或者 code 发生变化（岛津可能会直接跳码）
            if c == 0 or c != seg_code:
                seg_end = i - 1
                segments.append((seg_start, seg_end, int(seg_code)))
                in_seg = False
                seg_start = None
                seg_code = None
                # 如果是跳码且新码非0，立刻开始新片段
                if c != 0:
                    in_seg = True
                    seg_start = i
                    seg_code = c

    if in_seg and seg_start is not None:
        segments.append((seg_start, codes.size - 1, int(seg_code)))

    # 生成 events：默认每个 segment 产生 1 个 event（onset = segment start）
    events = []
    for s_i, e_i, c in segments:
        onset = float(t[s_i])

        if duration_mode == "zero":
            dur = 0.0
        else:
            # segment duration: t[e_i] - t[s_i] (+ 一个采样间隔会更像“持续到最后采样点”)
            if e_i > s_i:
                dur = float(t[e_i] - t[s_i])
            else:
                dur = 0.0

        events.append((onset, dur, c, str(c)))

    # 可选：按 min_gap_s 合并相邻同码事件（你现在是单点脉冲时一般不合并）
    if min_gap_s > 0 and len(events) > 1:
        merged = [list(events[0])]
        for onset, dur, val, cond in events[1:]:
            last = merged[-1]
            last_on, last_dur, last_val, last_cond = last
            last_end = last_on + last_dur
            if (val == last_val) and (onset - last_end <= float(min_gap_s)):
                # merge
                new_end = max(last_end, onset + dur)
                last[1] = float(new_end - last_on)
            else:
                merged.append([onset, dur, val, cond])
        events = [tuple(x) for x in merged]

    df = pd.DataFrame(events, columns=["onset", "duration", "value", "condition"])
    df = df.sort_values("onset").reset_index(drop=True)
    return df


# -------------------------
# Part 2
# Montage / Raw 构建 / 保存
# -------------------------
# -------------------------
# map origins -> nasion/lpa/rpa/cz
# -------------------------
def map_origins_to_mne(orig_coords: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float, float]]:
    """
    更严格的 origin 映射：
    - 只在 key “等于/以这些 token 结尾/开头”时匹配，避免 'al'/'ar' 这种误命中
    - 兼容：Nz, NzHS, Nasion, AL, ALHS, LPA, AR, ARHS, RPA, Cz, CzHS
    """
    mapping: Dict[str, Tuple[float, float, float]] = {}
    if not orig_coords:
        return mapping

    def _norm(k: str) -> str:
        # 去空格、下划线、连字符，统一小写
        k = str(k).strip().lower()
        k = re.sub(r'[\s_\-]+', '', k)
        return k

    # 允许的“规范 token”
    NASION_KEYS = {"nz", "nzhs", "nasion", "nasionhs"}
    LPA_KEYS    = {"al", "alhs", "lpa", "lpahs", "left", "leftear", "lpaear"}
    RPA_KEYS    = {"ar", "arhs", "rpa", "rpahs", "right", "rightear", "rpaear"}
    CZ_KEYS     = {"cz", "czhs"}

    for key, coord in orig_coords.items():
        k = _norm(key)
        xyz = tuple(np.asarray(coord, float).tolist())

        # 用“等于”或“明确前后缀”匹配，避免误命中
        if (k in NASION_KEYS) or k.startswith("nz") or k.startswith("nasion"):
            mapping["nasion"] = xyz
            continue

        if (k in LPA_KEYS) or k.startswith("lpa") or k.endswith("lpa") or k.startswith("al"):
            mapping["lpa"] = xyz
            continue

        if (k in RPA_KEYS) or k.startswith("rpa") or k.endswith("rpa") or k.startswith("ar"):
            mapping["rpa"] = xyz
            continue

        if (k in CZ_KEYS) or k.startswith("cz"):
            mapping["cz"] = xyz
            continue

    logger.info(f"Origin mapping -> MNE landmarks found: {list(mapping.keys())}")

    # 关键提示：如果缺 fiducials，直接报警
    needed = {"nasion", "lpa", "rpa"}
    miss = needed - set(mapping.keys())
    if miss:
        logger.warning(f"[Fiducials missing] 缺少 {sorted(miss)}，2D/3D 可能会漂。请检查 origin 文件 label。")

    return mapping

# -------------------------
# build channel positions (ch_pos dict) for montage
# -------------------------
def build_channel_positions(channel_pairs, others_coords):
    """
    生成测量通道（S#_D# hbo/hbr）的中点坐标。
    注意：这里不要塞 S# / D# optode 点位！optode 另外加。
    """
    ch_pos = {}

    for (s_idx, d_idx) in channel_pairs:
        t_key = f"T{s_idx}"
        r_key = f"R{d_idx}"

        if t_key in others_coords and r_key in others_coords:
            pos = 0.5 * (others_coords[t_key] + others_coords[r_key])
        else:
            pos = np.array([0.0, 0.0, 0.0])
            logger.warning(f"找不到 {t_key}/{r_key} 坐标，使用 0,0,0")

        ch_pos[f"S{s_idx}_D{d_idx} hbo"] = pos
        ch_pos[f"S{s_idx}_D{d_idx} hbr"] = pos

    logger.info(f"montage 测量通道数: {len(ch_pos)}")
    return ch_pos

# -------------------------
# build Raw from data_matrix
# -------------------------
def build_raw_from_matrix(
    time: np.ndarray,
    data_matrix: np.ndarray,
    channel_pairs: List[Tuple[int, int]],
    others_coords: Dict[str, np.ndarray],
    origin_coords: Dict[str, np.ndarray],
    sfreq: float,
    subject: Optional[str] = None,
) -> Tuple[mne.io.Raw, Dict[str, np.ndarray], mne.channels.DigMontage]:

    sources_dict, detectors_dict, _ = split_others_into_t_r_ch(others_coords)

    # 1) RawArray：每个 (S,D) 两个通道（hbo/hbr）
    ch_names: List[str] = []
    ch_types: List[str] = []
    rows: List[np.ndarray] = []

    for ch_i, (s_idx, d_idx) in enumerate(channel_pairs):
        base = ch_i * 3
        oxy = data_matrix[:, base]
        deoxy = data_matrix[:, base + 1]

        name_hbo = f"S{s_idx}_D{d_idx} hbo"
        name_hbr = f"S{s_idx}_D{d_idx} hbr"

        ch_names += [name_hbo, name_hbr]
        ch_types += ["hbo", "hbr"]
        rows += [oxy, deoxy]

    data_for_mne = np.vstack(rows)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data_for_mne, info)

    # 2) montage：optodes + fiducials（nasion/lpa/rpa）
        # 2) montage：用 fiducials 把坐标变换到 Neuromag head frame
    #    然后把“测量通道 Sx_Dy hbo/hbr”的中点坐标写进 montage（raw.ch_names 才能对上）

    # ---- helper：对 dict(label->xyz) 应用变换 ----
    def _apply_trans_dict(trans, dct):
        out = {}
        for k, v in dct.items():
            out[k] = apply_trans(trans, np.asarray(v, float))
        return out

    # ---- 2.1 fiducials ----
    lm = map_origins_to_mne(origin_coords)
    if not all(k in lm for k in ("nasion", "lpa", "rpa")):
        raise RuntimeError("缺少 nasion/lpa/rpa，无法对齐到 head frame。请检查 origin 文件标签（Nz/AL/AR 等）。")

    nasion = np.array(lm["nasion"], float)
    lpa    = np.array(lm["lpa"], float)
    rpa    = np.array(lm["rpa"], float)

    # 计算从当前坐标系 -> Neuromag head 坐标系的刚体变换
    trans = get_ras_to_neuromag_trans(nasion=nasion, lpa=lpa, rpa=rpa)

    # 变换后的 fiducials（再传入 montage）
    nasion_h = apply_trans(trans, nasion)
    lpa_h    = apply_trans(trans, lpa)
    rpa_h    = apply_trans(trans, rpa)

    # ---- 2.2 测量通道位置（关键：名字必须匹配 raw.ch_names）----
        # ---- 2.2 测量通道位置（关键：名字必须匹配 raw.ch_names）----
    ch_pos_meas = build_channel_positions(channel_pairs, others_coords)  # dict: "Sx_Dy hbo/hbr" -> xyz(np.ndarray)
    ch_pos_meas = _apply_trans_dict(trans, ch_pos_meas)
    ch_pos_meas = {k: tuple(v.tolist()) for k, v in ch_pos_meas.items()}

    # ---- 2.3 optodes 位置（S#, D#）作为额外 dig 点保留 ----
    ch_pos_optodes = {}
    for s_idx, xyz in sources_dict.items():
        ch_pos_optodes[f"S{s_idx}"] = xyz
    for d_idx, xyz in detectors_dict.items():
        ch_pos_optodes[f"D{d_idx}"] = xyz
    ch_pos_optodes = _apply_trans_dict(trans, ch_pos_optodes)
    ch_pos_optodes = {k: tuple(v.tolist()) for k, v in ch_pos_optodes.items()}

    # -------------------------
    # ✅ 关键校验：meas keys 必须能对上 raw.ch_names
    # -------------------------
    raw_name_set = set(raw.ch_names)
    meas_key_set = set(ch_pos_meas.keys())

    missing_in_montage = sorted(raw_name_set - meas_key_set)
    extra_in_montage   = sorted(meas_key_set - raw_name_set)

    logger.info(f"[montage] raw channels = {len(raw.ch_names)}")
    logger.info(f"[montage] meas ch_pos  = {len(ch_pos_meas)} (expect {len(raw.ch_names)})")
    logger.info(f"[montage] optodes      = {len(ch_pos_optodes)}")
    if missing_in_montage:
        logger.warning(f"[montage] meas coords missing for {len(missing_in_montage)} raw channels. Example: {missing_in_montage[:10]}")
    if extra_in_montage:
        logger.warning(f"[montage] meas coords has extra keys not in raw: {extra_in_montage[:10]}")

    # ✅ 如果 meas 坐标缺失很多，直接报错（比 silent failure 强）
    if len(ch_pos_meas) == 0:
        raise RuntimeError("ch_pos_meas 是空的：build_channel_positions 没生成任何测量通道坐标。请检查 T#/R# 标签是否匹配。")

    # ✅ 如果 meas key 对不上 raw.ch_names（比如大小写/空格差异），给出硬提示
    # 这里用“必须完全覆盖”策略（fNIRS 画3D/拓扑需要）
    if len(raw_name_set - meas_key_set) > 0:
        raise RuntimeError(
            "meas 通道坐标 key 没有覆盖 raw.ch_names。"
            "请检查 build_channel_positions() 生成的 key 是否严格等于 raw.ch_names（例如 'S1_D2 hbo'）。"
        )

    # ---- 2.4 一次性 montage：meas + optodes + fiducials ----
    ch_pos_all = {}
    ch_pos_all.update(ch_pos_meas)      # ✅ meas: 必须匹配 raw.ch_names
    ch_pos_all.update(ch_pos_optodes)   # optodes: S#/D# 额外点

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos_all,
        nasion=tuple(nasion_h.tolist()),
        lpa=tuple(lpa_h.tolist()),
        rpa=tuple(rpa_h.tolist()),
        coord_frame="head",
    )
    raw.set_montage(montage, on_missing="warn")


    # 3) 可选：存 cz（不参与 montage 计算）
    if "cz" in lm:
        desc0 = raw.info.get("description") or ""
        raw.info["description"] = (desc0 + f" | cz={lm['cz']}").strip(" |")

    logger.info(f"Raw shape: {raw._data.shape}")
    logger.info(f"optodes: S={len(sources_dict)}, D={len(detectors_dict)}")
    logger.info(f"fiducials present: {list(k for k in ['nasion','lpa','rpa'] if k in lm)}")

    return raw, {k: np.asarray(v, float) for k, v in ch_pos_optodes.items()}, montage

# -------------------------
# save functions: FIF and SNIRF (snirf via mne_nirs if available else None)
# -------------------------
def save_raw_fif(raw: mne.io.Raw, out_fif: str, overwrite: bool = True) -> None:
    """保存 Raw 为 FIF (MNE 格式)"""
    out_dir = os.path.dirname(out_fif)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    logger.info(f"写入 FIF: {out_fif}")
    raw.save(out_fif, overwrite=overwrite)
    logger.info("FIF 保存完成。")

# -------------------------
# Part 3
# Events -> Annotations / events array, OD/Hb conversion, export helpers
# -------------------------

# -------------------------
# 1) events DataFrame -> Annotations and MNE events array
# -------------------------
def events_df_to_annotations(df_events: pd.DataFrame) -> mne.Annotations:
    """
    将包含 onset (s), duration (s), condition (str) 的 events DataFrame 转换为 MNE Annotations。
    返回 mne.Annotations 对象。
    """
    if 'onset' not in df_events.columns or 'duration' not in df_events.columns:
        raise RuntimeError("events DataFrame 必须包含 'onset' 和 'duration' 列。")

    onsets = df_events['onset'].astype(float).tolist()
    durs = df_events['duration'].astype(float).tolist()
    if 'condition' in df_events.columns:
        descs = df_events['condition'].astype(str).tolist()
    else:
        # fallback to value
        descs = df_events['value'].astype(str).tolist()

    ann = mne.Annotations(onset=onsets, duration=durs, description=descs)
    logger.info(f"转换为 Annotations：{len(ann)} 条目（示例前三）：{list(zip(onsets[:3], durs[:3], descs[:3]))}")
    return ann


def events_df_to_mne_events(df_events: pd.DataFrame, sfreq: float) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    将 events DataFrame 转换为 MNE events 数组 (samples, 0, event_id) 与 event_id 字典。
    event_id 使用 'condition' 列（如有），否则使用 'value' 列。
    """
    if 'onset' not in df_events.columns or 'value' not in df_events.columns:
        raise RuntimeError("events DataFrame 必须包含 'onset' 和 'value' 列以生成 events array。")

    if 'condition' in df_events.columns:
        cond_col = 'condition'
    else:
        cond_col = 'value'

    # map unique conditions to ints (respect given value if numeric)
    unique = df_events[cond_col].unique()
    event_id: Dict[str, int] = {}
    # If 'value' already numeric mapping, we prefer that numeric value when condition==value
    if cond_col == 'value':
        # use the numeric values directly
        for val in sorted(df_events['value'].unique()):
            event_id[str(val)] = int(val)
    else:
        # assign sequential integers starting at 1 for each condition
        for i, cond in enumerate(unique, start=1):
            event_id[str(cond)] = int(i)

    # build events array
    onsets_s = df_events['onset'].astype(float).values
    samples = np.round(onsets_s * float(sfreq)).astype(int)
    values = df_events['value'].astype(int).values  # user provided numeric value
    # if 'condition' used for mapping, use event_id[condition]
    if cond_col == 'condition':
        values = df_events['condition'].apply(lambda c: event_id[str(c)]).astype(int).values

    events = np.column_stack([samples, np.zeros_like(samples, dtype=int), values]).astype(int)
    logger.info(f"生成 MNE events 数组：形状 {events.shape}，event_id keys: {list(event_id.keys())[:10]}")
    return events, event_id


# -------------------------
# 2) Optical Density & MBLL (Intensity -> OD -> HbO/HbR)
# -------------------------
def load_extinction_coeffs_from_csv(path: str) -> np.ndarray:
    """
    读取一个 CSV 文件，其中每行对应一个波长（或列），并包含 HbO 和 HbR 的消光系数（单位自行保证）。
    预期格式（无 header 或带 header）：
        wavelength, ext_hbo, ext_hbr
    返回 shape: (n_wavelengths, 2)
    """
    ensure_file(path, "extinction coeffs file")
    df = pd.read_csv(path)
    # try to detect columns
    cols = list(df.columns)
    if len(cols) >= 3:
        ext = df.iloc[:, 1:3].values.astype(float)
        return ext
    else:
        raise RuntimeError("消光系数文件格式不正确，预期包含 wavelength, ext_hbo, ext_hbr 三列（或至少两列 ext_hbo/ext_hbr）。")


def get_default_extinction(wavelengths: Sequence[float]) -> np.ndarray:
    """
    返回一个 placeholder 的消光系数矩阵 (n_wavelengths, 2) (hbo, hbr)。
    **注意**：这些仅为占位示例，强烈建议使用实验室提供或文献中准确的消光系数。
    """
    wl_list = [float(w) for w in wavelengths]
    ext = []
    logger.warning("使用默认占位消光系数。建议提供真实的消光系数文件以获得准确的 Hb 计算结果。")
    # create simple synthetic values that vary with wavelength (not real)
    for wl in wl_list:
        # these are synthetic placeholders, not for publication use
        ext_hbo = 1.0 / (1.0 + (wl - 800.0) * 0.01) + 1.0
        ext_hbr = 1.2 / (1.0 + (wl - 800.0) * 0.012) + 0.5
        ext.append([ext_hbo, ext_hbr])
    return np.array(ext, dtype=float)


def intensity_to_od(intensity: np.ndarray, baseline: Optional[Union[float, np.ndarray]] = None, baseline_mode: str = "first") -> np.ndarray:
    """
    将强度 (I) 转换为光学密度 OD = -ln(I / I0)
    intensity: shape (n_wavelengths, n_time) OR (n_time, n_wavelengths) — 强度排列需明确由调用者保证
    baseline: 如果为 None，根据 baseline_mode 计算 I0:
        - 'first': 使用第一时间点作为 I0（每波长）
        - 'mean': 使用整段信号均值作为 I0（每波长）
        - 数组: 指定 I0 数组（长度 = n_wavelengths）
    返回 OD (same shape as intensity)
    """
    I = np.array(intensity, dtype=float)
    # determine axis: prefer shape (n_wl, n_time)
    if I.ndim == 1:
        I = I[np.newaxis, :]
    if I.shape[0] < I.shape[1] and (baseline is None and I.shape[0] <= 3):
        # ambiguous orientation: assume it's (n_time, n_wl) -> transpose to (n_wl, n_time)
        I = I.T

    n_wl, n_time = I.shape

    if baseline is None:
        if baseline_mode == "first":
            I0 = I[:, 0].copy()
        elif baseline_mode == "mean":
            I0 = I.mean(axis=1)
        else:
            raise ValueError("baseline_mode must be 'first' or 'mean' or provide baseline array.")
    else:
        I0 = np.array(baseline, dtype=float)
        if I0.ndim == 0:
            I0 = np.full((n_wl,), float(I0))
        if I0.shape[0] != n_wl:
            # attempt broadcast from single value
            if I0.size == 1:
                I0 = np.full((n_wl,), float(I0))
            else:
                raise ValueError("baseline length must equal number of wavelengths (or single scalar).")

    # avoid division by zero
    I0_safe = I0.copy()
    I0_safe[I0_safe == 0] = np.finfo(float).eps

    # compute OD
    with np.errstate(divide='ignore', invalid='ignore'):
        od = -np.log(I / I0_safe[:, np.newaxis])
    return od


def od_to_hb(od: np.ndarray, extinction: np.ndarray, dpf: Optional[float] = None) -> np.ndarray:
    """
    根据 MBLL 将 OD 转换为 HbO / HbR 变化：
      extinction: shape (n_wavelengths, 2) (ext_hbo, ext_hbr)
      od: shape (n_wavelengths, n_time)
    结果返回 conc: shape (2, n_time) -> [hbo, hbr]
    说明：
      conc = pseudo_inv(extinction) @ (od / d)   (若提供 dpf 则用于修正)
    注意：
      单位取决于 extinction 的单位；用户需保证一致性（常见为 molar extinction）。
    """
    ext = np.array(extinction, dtype=float)
    if ext.ndim != 2 or ext.shape[1] != 2:
        raise ValueError("extinction must have shape (n_wavelengths, 2) corresponding to [hbo, hbr].")

    od = np.array(od, dtype=float)
    # ensure od shape (n_wl, n_time)
    if od.ndim == 1:
        od = od[:, np.newaxis]
    if od.shape[0] != ext.shape[0]:
        raise ValueError(f"OD 第一维 ({od.shape[0]}) 与 extinction 行数 ({ext.shape[0]}) 不匹配。")

    # optionally divide by differential pathlength factor (dpf). If None, assume d=1
    if dpf is None:
        scale = 1.0
    else:
        scale = 1.0 / float(dpf)

    # pseudo-inverse
    pinv = np.linalg.pinv(ext)
    # conc shape (2, n_time)
    conc = pinv.dot(od) * scale
    # conc[0,:] = HbO, conc[1,:] = HbR
    return conc

# -------------------------
# 3) Export intermediate artifacts
# -------------------------
def export_channel_positions_csv(ch_pos: Dict[str, np.ndarray], out_csv: str) -> None:
    """把 ch_pos 字典写成 CSV (Label,X,Y,Z)."""
    rows = []
    for k, v in ch_pos.items():
        rows.append((k, float(v[0]), float(v[1]), float(v[2])))
    df = pd.DataFrame(rows, columns=['Label', 'X', 'Y', 'Z'])
    df.to_csv(out_csv, index=False)
    logger.info(f"导出 channel positions 到: {out_csv}")


def export_data_matrix_csv(time: np.ndarray, data_matrix: np.ndarray, out_csv: str) -> None:
    """
    导出时间与原始数据矩阵为 CSV。
    如果 data_matrix shape (N_time, n_sigcols) 会生成 columns: time, col01, col02...
    """
    n_time = time.shape[0]
    n_cols = data_matrix.shape[1]
    cols = ['time'] + [f'col{i+1:03d}' for i in range(n_cols)]
    arr = np.hstack([time.reshape(n_time, 1), data_matrix])
    df = pd.DataFrame(arr, columns=cols)
    df.to_csv(out_csv, index=False)
    logger.info(f"导出原始数据矩阵到: {out_csv}")

# -------------------------
# 4) sanity checks: matching channels vs coords
# -------------------------
def sanity_check_channel_coord_matching(channel_pairs: List[Tuple[int, int]], others_coords: Dict[str, np.ndarray]) -> None:
    """
    简单检查：others_coords 是否包含 CHxx 或 T#/R# 标签来匹配 channel_pairs。
    打印若干提示帮助用户修复定位文件（比如 CH 数量 vs channels 数量不匹配）。
    """
    n_ch = len(channel_pairs)
    # count CH labels present
    ch_present = [k for k in others_coords.keys() if re.match(r'^(ch)\d+', str(k), flags=re.I)]
    t_present = [k for k in others_coords.keys() if re.match(r'^(t)\d+', str(k), flags=re.I)]
    r_present = [k for k in others_coords.keys() if re.match(r'^(r)\d+', str(k), flags=re.I)]
    logger.info(f"检测到 CH 标签数: {len(ch_present)}, T 标签数: {len(t_present)}, R 标签数: {len(r_present)} (期望通道数: {n_ch})")
    if len(ch_present) < n_ch and (len(t_present) < n_ch or len(r_present) < n_ch):
        logger.warning("定位文件中 CH/T/R 标签数量少于通道数量，后续将采用可用的中点或填 0,0,0。请检查 others CSV 是否包含完整的 CH01..CHxx 或 T1/R1 列表。")


# -------------------------
# Part 4
# Final pipeline, SNIRF writer fallback, main CLI
# -------------------------
# -------------------------
# Helper: extract T/R and CH dicts from others_coords (label->coord)
# -------------------------
def split_others_into_t_r_ch(others_coords: Dict[str, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[str, np.ndarray]]:
    """
    从 others_coords 字典中分离出 sources (T# -> coord), detectors (R# -> coord), and channels (CHxx -> coord).
    返回 (sources_dict, detectors_dict, ch_coords_dict)
    """
    sources: Dict[int, np.ndarray] = {}
    detectors: Dict[int, np.ndarray] = {}
    ch_coords: Dict[str, np.ndarray] = {}

    for k, v in others_coords.items():
        key = str(k).strip()
        lower = key.lower()
        # source T#
        m = re.match(r'^[tT]\s*(\d+)$', key)
        if m:
            idx = int(m.group(1))
            sources[idx] = v
            continue
        m = re.match(r'^t(\d+)$', lower)
        if m:
            idx = int(m.group(1))
            sources[idx] = v
            continue
        # detector R#
        m = re.match(r'^[rR]\s*(\d+)$', key)
        if m:
            idx = int(m.group(1))
            detectors[idx] = v
            continue
        m = re.match(r'^r(\d+)$', lower)
        if m:
            idx = int(m.group(1))
            detectors[idx] = v
            continue
        # channel CHxx
        if lower.startswith("ch"):
            ch_coords[key] = v
            continue
        # also accept CH01 style with leading zeros
        m = re.match(r'^ch0*(\d+)$', lower)
        if m:
            ch_coords[key] = v
            continue
        # fallback: any other label put into ch_coords too
        ch_coords[key] = v

    return sources, detectors, ch_coords
def filter_pairs_by_optodes(
    data_matrix: np.ndarray,
    channel_pairs: List[Tuple[int, int]],
    sources_dict: Dict[int, np.ndarray],
    detectors_dict: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    keep_idx = []
    new_pairs = []
    for i, (s, d) in enumerate(channel_pairs):
        if (s in sources_dict) and (d in detectors_dict):
            keep_idx.append(i)
            new_pairs.append((s, d))

    if len(new_pairs) == 0:
        raise RuntimeError("过滤后没有任何通道可用：others.csv 与 TXT 的 probe 不匹配。")

    cols = []
    for i in keep_idx:
        base = i * 3
        cols.extend([base, base + 1, base + 2])

    data_matrix2 = data_matrix[:, cols]

    dropped = len(channel_pairs) - len(new_pairs)
    if dropped > 0:
        logger.warning(f"[pairs-filter] dropped {dropped}/{len(channel_pairs)} pairs due to missing optodes.")
        logger.warning(f"[pairs-filter] examples dropped: {[(channel_pairs[i]) for i in range(len(channel_pairs)) if i not in keep_idx][:10]}")
    return data_matrix2, new_pairs


# -------------------------
# Fallback SNIRF writer (h5py) — adapted/cleaned from earlier write_snirf_h5
# -------------------------
def write_snirf_h5(
    out_path,
    time,
    data_matrix,
    channel_pairs,
    sources,
    detectors,
    landmark_labels,
    landmark_coords,
    subject="sub-01",
    wavelengths=(760, 830),
    events_dict: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,  # ✅ 新增
):

    """
    写 HbO/HbR (processed) 的 SNIRF，使得 mne.io.read_raw_snirf 可读
    关键：measurementList*/dataType 必须是 int 标量 99999
         measurementList*/dataTypeLabel 必须是 HbO/HbR（MNE 读取后转成 hbo/hbr）
    """
    import numpy as np
    import h5py

    time = np.asarray(time, dtype=float)
    n_time = len(time)
    n_channels = len(channel_pairs)

    # --- dataTimeSeries: (n_time, 2*n_channels) 依次 HbO, HbR ---
    hb_cols = []
    for ch in range(n_channels):
        base = ch * 3
        hb_cols.append(data_matrix[:, base])       # oxy -> HbO
        hb_cols.append(data_matrix[:, base + 1])   # deoxy -> HbR
    hb_ts = np.column_stack(hb_cols).astype(float)


    with h5py.File(out_path, "w") as f:
        f["formatVersion"] = _h5_str("1.0")
        nirs = f.create_group("nirs")

        # ---------- data ----------
        data1 = nirs.create_group("data1")
        data1.create_dataset("time", data=time)
        data1.create_dataset("dataTimeSeries", data=hb_ts)

        # ---------- measurementList ----------
        # MNE: snirf_data_type = dat[".../measurementList1/dataType"].item()
        # 必须是 1 或 99999；HbO/HbR 属于 processed => 99999
        meas_idx = 0
        for ch_idx, (s, d) in enumerate(channel_pairs, start=1):
            # HbO
            meas_idx += 1
            g = data1.create_group(f"measurementList{meas_idx}")
            g.create_dataset("dataType", data=np.int64(99999))
            g.create_dataset("dataTypeIndex", data=np.int64(1))
            g.create_dataset("dataTypeLabel", data=_h5_str("hbO"))
            g.create_dataset("sourceIndex", data=np.int64(s))
            g.create_dataset("detectorIndex", data=np.int64(d))
            g.create_dataset("channelIndex", data=np.int64(ch_idx))

            # HbR
            meas_idx += 1
            g = data1.create_group(f"measurementList{meas_idx}")
            g.create_dataset("dataType", data=np.int64(99999))
            g.create_dataset("dataTypeIndex", data=np.int64(2))
            g.create_dataset("dataTypeLabel", data=_h5_str("hbr"))
            g.create_dataset("sourceIndex", data=np.int64(s))
            g.create_dataset("detectorIndex", data=np.int64(d))
            g.create_dataset("channelIndex", data=np.int64(ch_idx))

        # ---------- probe ----------

        def _as_bytes_list(str_list):
          return np.asarray([s.encode("utf-8") for s in str_list], dtype="S")

        max_s = int(max(sources.keys())) if sources else 0
        max_d = int(max(detectors.keys())) if detectors else 0

        src_pos = np.full((max_s, 3), np.nan, dtype=float)
        det_pos = np.full((max_d, 3), np.nan, dtype=float)

        for k, xyz in sources.items():      # k 是 1-based
          src_pos[int(k)-1, :] = np.asarray(xyz, float)

        for k, xyz in detectors.items():
          det_pos[int(k)-1, :] = np.asarray(xyz, float)

        probe = nirs.create_group("probe")
        probe.create_dataset("sourcePos3D", data=src_pos)
        probe.create_dataset("detectorPos3D", data=det_pos)

        probe.create_dataset("sourceLabels", data=_as_bytes_list([f"S{i}" for i in range(1, max_s+1)]))
        probe.create_dataset("detectorLabels", data=_as_bytes_list([f"D{i}" for i in range(1, max_d+1)]))

        probe.create_dataset("wavelengths", data=np.asarray(wavelengths, dtype=float))

        # ---------- metadata ----------
        meta = nirs.create_group("metaDataTags")
        meta.create_dataset("SubjectID", data=_h5_str(subject))
        meta.create_dataset("MeasurementType", data=_h5_str("nirs"))
        meta.create_dataset("LengthUnit", data=_h5_str("m"))
        meta.create_dataset("TimeUnit", data=_h5_str("s"))
        from datetime import datetime

        now = datetime.now()
        meta["MeasurementDate"] = _h5_str(now.strftime("%Y-%m-%d"))   # 例如 2026-01-27
        meta["MeasurementTime"] = _h5_str(now.strftime("%H:%M:%S"))   # 例如 15:31:05
        meta["HbUnit"] = _h5_str("M")
        meta["HbScaling"] = _h5_str("oxyHb/deoxyHb assumed uM; scaled x1e-6 to M before writing")

        # ---------- stim (markers/events) ----------
        if events_dict:
            stim_i = 0
            total = 0
            for name, items in events_dict.items():
                stim_i += 1
                stim = nirs.create_group(f"stim{stim_i}")

                stim.create_dataset("name", data=_h5_str(str(name)))

                arr = np.array(
                    [[float(on), float(dur), float(val)] for (on, dur, val) in items],
                    dtype=float
                )
                stim.create_dataset("data", data=arr)
                total += arr.shape[0]

            logger.info(f"写入 stim: {stim_i} 组, markers 总数: {total}")
        else:
            logger.info("events_dict 为空：不写 stim（所以 SNIRF 不会有 marker）")


    logger.info(f"SNIRF saved: {out_path}")






from dataclasses import dataclass

@dataclass
class BuildArtifacts:
    raw: mne.io.Raw
    time_arr: np.ndarray
    data_matrix: np.ndarray
    channel_pairs: List[Tuple[int, int]]
    sources_dict: Dict[int, np.ndarray]
    detectors_dict: Dict[int, np.ndarray]
    ch_coords_dict: Dict[str, np.ndarray]
    origin_coords: Dict[str, np.ndarray]
    sfreq: float
    events_dict_for_snirf: Optional[Dict[str, List[Tuple[float, float, float]]]]


def build_artifacts(
    txt_path: str,
    origin_path: str,
    others_path: str,
    events_path: Optional[str],
    sfreq: float,
    subject: str,
    scale_to_m: bool = True,
    length_unit: str = "m",
) -> BuildArtifacts:
    """
    只做：parse -> build raw -> events -> annotations -> 产出保存所需的所有中间产物
    （不写文件，不 plot）
    """
    logger.info("======== SNIRF 转换流程开始 ========")

    # validate inputs
    ensure_file(txt_path, "Shimadzu TXT")
    ensure_file(origin_path, "origin CSV")
    ensure_file(others_path, "others CSV")
    if events_path:
        ensure_file(events_path, "events CSV")

    # 1) parse TXT
    time_arr, data_matrix, raw_task, raw_mark, raw_count, channel_pairs = parse_shimadzu_txt(txt_path)

    # ---- 1.1) auto infer sfreq if needed ----
    # sfreq 允许为 None / 0 / 负数：都视为需要自动推断
    if (sfreq is None) or (float(sfreq) <= 0):
        sfreq = infer_sfreq_from_time(time_arr)
        logger.info(f"[sfreq-auto] inferred sfreq = {sfreq:.6f} Hz")
    else:
        sfreq = float(sfreq)
        logger.info(f"[sfreq] user provided sfreq = {sfreq:.6f} Hz")

    # 2) parse coords
    origin_coords = parse_coords_file(origin_path, unit=length_unit, scale_to_m=scale_to_m)
    others_coords = parse_coords_file(others_path, unit=length_unit, scale_to_m=scale_to_m)
    sources_dict, detectors_dict, ch_coords_dict = split_others_into_t_r_ch(others_coords)

    # ✅ 2.1) filter pairs by optodes actually present in others.csv
    data_matrix, channel_pairs = filter_pairs_by_optodes(
        data_matrix=data_matrix,
        channel_pairs=channel_pairs,
        sources_dict=sources_dict,
        detectors_dict=detectors_dict,
    )

    # 3) sanity checks
    sanity_check_channel_coord_matching(channel_pairs, others_coords)

    #4) 这里加单位缩放
    data_matrix_raw = data_matrix.astype(float).copy()   # original (assumed uM)
    data_matrix = data_matrix_raw * 1e-6                 # convert uM -> M
    logger.info("[unit] Hb assumed uM -> scaled to M (x1e-6) for BOTH FIF & SNIRF")

    # 5) build MNE Raw
    raw, ch_pos, montage = build_raw_from_matrix(
        time=time_arr,
        data_matrix=data_matrix,
        channel_pairs=channel_pairs,
        others_coords=others_coords,
        origin_coords=origin_coords,
        sfreq=sfreq,          # ✅ 这里现在一定是 float 了
        subject=subject,
    )

    # 6) events -> annotations + events array
    events_df: Optional[pd.DataFrame] = None
    events_dict_for_snirf: Optional[Dict[str, List[Tuple[float, float, float]]]] = None

    if events_path:
        # ---- 6.1) user provided events.csv ----
        events_df = parse_events_csv(events_path)
        logger.info(f"[events] loaded from CSV: n={len(events_df)}")
    else:
        # ---- 6.2) auto infer from TXT Task/Mark ----
        events_df = infer_events_from_task_mark(
            time_arr=time_arr,
            raw_task=raw_task,
            raw_mark=raw_mark,
            prefer="auto",              # Mark优先；若 Mark 全0 自动回退 Task
            min_code=1,                 # 忽略 0
            duration_mode="segment",    # 如果都是单点脉冲会得到 duration=0
            min_gap_s=0.0,              # 片段合并阈值（你现在是单点，所以无所谓）
        )
        logger.info(f"[events-auto] inferred events from TXT: n={len(events_df)}")

    # ---- 6.3) attach annotations and build snirf stim dict ----
    if (events_df is not None) and (len(events_df) > 0):
        # 规范化列：确保 onset/duration/value/condition 存在且类型对
        events_df = events_df.copy()
        if "condition" not in events_df.columns:
            events_df["condition"] = events_df["value"].astype(str)

        # annotations
        ann = events_df_to_annotations(events_df)
        raw.set_annotations(ann)

        # (可选) events array，后面你如果不用可以删掉
        try:
            events_arr, event_id = events_df_to_mne_events(events_df, sfreq)
            logger.info(f"[events] mne events array shape={events_arr.shape}, event_id={event_id}")
        except Exception as e:
            logger.warning(f"[events] 生成 MNE events array 失败（不影响 annotations）：{e}")

        # stim dict for SNIRF: {cond: [(onset,dur,val), ...]}
        events_dict_for_snirf = {}
        for _, row in events_df.iterrows():
            cond = str(row["condition"])
            events_dict_for_snirf.setdefault(cond, []).append(
                (float(row["onset"]), float(row["duration"]), float(row["value"]))
            )
    else:
        logger.warning("[events] events_df is empty -> no annotations, no stim will be written.")
        events_dict_for_snirf = None

    return BuildArtifacts(
        raw=raw,
        time_arr=time_arr,
        data_matrix=data_matrix,
        channel_pairs=channel_pairs,
        sources_dict=sources_dict,
        detectors_dict=detectors_dict,
        ch_coords_dict=ch_coords_dict,
        origin_coords=origin_coords,
        sfreq=sfreq,
        events_dict_for_snirf=events_dict_for_snirf,
    )


# -------------------------
# main pipeline
# -------------------------
def main_pipeline(
    txt_path: str,
    origin_path: str,
    others_path: str,
    events_path: Optional[str],
    out_fif: str,
    out_snirf: Optional[str],
    sfreq: float,
    subject: str,
    scale_to_m: bool = True,
    length_unit: str = "m",
    store_mode: str = "hbo_hbr",
    no_plot: bool = False,
    export_csv: bool = False
) -> None:
    """
    高级主流程：解析所有输入，构建 MNE Raw，添加注释/事件，保存 FIF，并可选写 SNIRF。
    （结构重构版：核心构建逻辑抽到 build_artifacts()；其余行为不变）
    """
    # 1) build all artifacts (parse/build/events/annotations)
    art = build_artifacts(
        txt_path=txt_path,
        origin_path=origin_path,
        others_path=others_path,
        events_path=events_path,
        sfreq=sfreq,
        subject=subject,
        scale_to_m=scale_to_m,
        length_unit=length_unit,
    )

    raw = art.raw
    time_arr = art.time_arr
    data_matrix = art.data_matrix
    channel_pairs = art.channel_pairs
    sources_dict = art.sources_dict
    detectors_dict = art.detectors_dict
    ch_coords_dict = art.ch_coords_dict
    origin_coords = art.origin_coords
    sfreq = art.sfreq
    events_dict_for_snirf = art.events_dict_for_snirf

    # 4) optional exports of intermediates
    base_dir = os.path.dirname(out_fif) or "."
    base_name = os.path.splitext(os.path.basename(out_fif))[0]

    if export_csv:
        try:
            export_data_matrix_csv(time_arr, data_matrix_raw, os.path.join(base_dir, base_name + "_matrix_raw_uM.csv"))
            export_data_matrix_csv(time_arr, data_matrix, os.path.join(base_dir, base_name + "_matrix_SI_M.csv"))
            export_channel_positions_csv(ch_coords_dict, os.path.join(base_dir, base_name + "_chpos.csv"))
        except Exception as e:
            logger.warning(f"导出中间 CSV 时发生错误: {e}")

    # 7) quick preview
    if not no_plot:
        try:
            logger.info("打开快速预览窗口 (block=True)。关闭窗口后将继续保存。")
            raw.plot(n_channels=min(40, len(raw.ch_names)), duration=30, show=True, block=True)
        except Exception as e:
            logger.warning(f"可视化预览失败（可能为无图形界面）：{e}")

    # 8) save FIF
    save_raw_fif(raw, out_fif, overwrite=True)

    # 9) save SNIRF if requested
    if out_snirf:
        logger.info("使用自定义 HbO/HbR writer 写入 SNIRF（跳过 mne_nirs writer）")

        write_snirf_h5(
            out_path=out_snirf,
            time=time_arr,
            data_matrix=data_matrix,
            channel_pairs=channel_pairs,
            sources=sources_dict,
            detectors=detectors_dict,
            landmark_labels=list(origin_coords.keys()),
            landmark_coords=np.array(list(origin_coords.values())) if origin_coords else np.zeros((0, 3)),
            subject=subject,
            wavelengths=(760, 830),                 # ✅ 真实波长就填这里
            events_dict=events_dict_for_snirf       # ✅ 自动 or csv 都会写进 stim
        )

        logger.info("SNIRF 写入完成（custom writer）")

    # 10) summary & tips
    logger.info("======== 转换完成 ========")
    logger.info(f"Saved FIF: {out_fif}")
    if out_snirf:
        logger.info(f"Saved SNIRF: {out_snirf}")
    logger.info("下一步建议：在 Python 中使用 mne.io.read_raw_fif(<out_fif>, preload=True) 检查 raw，对 events 使用 mne.events_from_annotations(raw) 以获得 events array。")
    logger.info("如果通道位置不对，请检查 others CSV 中 CH/T/R 标签是否与 channel 配对一致。")


# -------------------------
# CLI entrypoint
# -------------------------
def build_cli_and_run():
    p = argparse.ArgumentParser(prog="snirf.py", description="Convert Shimadzu TXT + origin/others + events -> MNE Raw (.fif) and optional SNIRF")
    p.add_argument("--txt", required=True, help="Shimadzu TXT (raw) file path")
    p.add_argument("--origin", required=True, help="Origin CSV path (Cz,Nz,AL,AR)")
    p.add_argument("--others", required=True, help="Others CSV path (T*,R*,CH*)")
    p.add_argument("--events", required=False, help="Events CSV path (onset,duration,value,condition)")
    p.add_argument("--out", required=True, help="Output MNE Raw .fif path")
    p.add_argument("--snirf", required=False, help="Optional output SNIRF path")
    p.add_argument("--sfreq", type=float, required=False, default=None, help="Sampling frequency (Hz). If omitted, infer from TXT time column.")
    p.add_argument("--subject", type=str, required=True, help="Subject ID")
    p.add_argument("--length-unit", choices=["mm", "cm", "m"], default="mm", help="Unit used in coord CSV (default mm)")
    p.add_argument("--scale-to-m", action="store_true", help="Scale coordinates to meters (recommended)")
    p.add_argument("--store-mode", choices=["hbo_hbr", "intensity"], default="hbo_hbr", help="How to store data in Raw")
    p.add_argument("--no-plot", action="store_true", help="Skip preview plotting")
    p.add_argument("--export-csv", action="store_true", help="Export intermediate CSVs (matrix, chpos)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging DEBUG")
    args = p.parse_args()

    # set logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled (DEBUG)")

    # run pipeline
    main_pipeline(
        txt_path=args.txt,
        origin_path=args.origin,
        others_path=args.others,
        events_path=args.events,
        out_fif=args.out,
        out_snirf=args.snirf,
        sfreq=args.sfreq,
        subject=args.subject,
        scale_to_m=args.scale_to_m,
        length_unit=args.length_unit if args.length_unit else "mm",
        no_plot=args.no_plot,
        export_csv=args.export_csv
    )

