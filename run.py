#!/usr/bin/env python3
"""DeepFaceLab Torch 版本启动器。

目标：提供一个接近原版 DeepFaceLab 的交互式入口（不包含任何推理/演示菜单）。

说明：需要可脚本化/全参数控制时，优先使用 `python main.py ...`（与原版一致）。
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

import torch


def _has_accelerator() -> bool:
    if torch.cuda.is_available():
        return True
    try:
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except Exception:
        return False


def print_banner():
    """打印欢迎横幅"""
    print("=" * 70)
    print(" " * 15 + "DeepFaceLab PyTorch 版本")
    print(" " * 20 + "启动器")
    print("=" * 70)
    print()


def print_menu():
    """打印主菜单"""
    print("\n" + "=" * 70)
    print("请选择功能：")
    print("=" * 70)
    print()
    print("  1. 提取（自动 S3FD，推荐）")
    print("  2. 提取（手动，排错）")
    print("  3. Sorter（人脸排序）")
    print("  4. FacesetResizer（重采样/改 face_type）")
    print("  5. FacesetEnhancer（细节增强）")
    print("  6. 训练（选择/新建模型）")
    print("  7. 训练（继续最新模型）")
    print("  8. 合成 (Merger)")
    print("  9. 导出 DFM (ExportDFM)")
    print(" 10. Util（工具集）")
    print(" 11. VideoEd（视频/序列处理）")
    print(" 12. XSeg（编辑器/应用/移除/抓取）")
    print(" 13. 开发测试（遮罩工具 dev_test）")
    print(" 14. 系统信息")
    print("  0. 退出")
    print()
    print("=" * 70)


def _input_str(prompt: str, default: str | None = None) -> str:
    if default is None:
        return input(f"{prompt}：").strip()
    s = input(f"{prompt} [{default}]：").strip()
    return s if s != "" else default


def _input_bool(prompt: str, default: bool = False) -> bool:
    default_str = "y" if default else "n"
    while True:
        s = input(f"{prompt} [y/n] (默认 {default_str})：").strip().lower()
        if s == "":
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False


def _input_path(prompt: str, default: Path | None = None, create: bool = False) -> Path:
    default_str = str(default) if default is not None else None
    s = _input_str(prompt, default_str)
    p = Path(s).expanduser().resolve()
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def _list_model_class_names() -> list[str]:
    root = Path(__file__).resolve().parent / 'models'
    if not root.exists():
        return []
    out: list[str] = []
    for d in sorted(root.glob('Model_*')):
        if not d.is_dir():
            continue
        if not (d / '__init__.py').exists():
            continue
        out.append(d.name.split('_', 1)[1])
    return out


def _choose_from_list(title: str, items: list[str], default_index: int = 0) -> str:
    if not items:
        raise RuntimeError(f"没有可选项：{title}")
    print(f"\n{title}：")
    for i, it in enumerate(items):
        mark = " (默认)" if i == default_index else ""
        print(f"  {i}. {it}{mark}")
    while True:
        s = input(f"请选择 [0-{len(items)-1}] (默认 {default_index})：").strip()
        if s == "":
            return items[default_index]
        try:
            idx = int(s)
            if 0 <= idx < len(items):
                return items[idx]
        except ValueError:
            pass


def _default_workspace_paths() -> tuple[Path, Path, Path]:
    # 对齐原版 DFL 常见的 workspace 目录布局
    ws = Path.cwd() / 'workspace'
    model = ws / 'model'
    src = ws / 'data_src' / 'aligned'
    dst = ws / 'data_dst' / 'aligned'
    return model, src, dst


def _find_latest_saved_model_base_name(saved_models_path: Path, model_class_name: str) -> str | None:
    """返回最新的模型 base 名（不包含 _ModelClass 后缀）。

        DFL 命名约定：
      <base>_<ModelClass>_data.dat
    """
    bases = _list_saved_model_base_names(saved_models_path, model_class_name)
    return bases[0] if bases else None


def _list_saved_model_base_names(saved_models_path: Path, model_class_name: str) -> list[str]:
    """列出可用的模型 base 名（按时间倒序）。

        DFL 命名约定：
      <base>_<ModelClass>_data.dat
    Where <base> itself may contain underscores.
    """
    try:
        suffix = f"_{model_class_name}_data.dat"
        candidates: list[tuple[float, str]] = []
        for fp in saved_models_path.glob(f"*{suffix}"):
            name = fp.name
            if not name.endswith(suffix):
                continue
            base = name[: -len(suffix)]
            try:
                mtime = fp.stat().st_mtime
            except Exception:
                mtime = 0.0
            candidates.append((mtime, base))

        if not candidates:
            return []

        # newest first
        candidates.sort(key=lambda x: x[0], reverse=True)

        # de-dup while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for _, base in candidates:
            if base in seen:
                continue
            seen.add(base)
            out.append(base)
        return out
    except Exception:
        return []


def _list_saved_models(saved_models_path: Path) -> list[tuple[float, str, str]]:
    """返回已保存模型列表：(mtime, model_class_name, base)，按时间倒序。"""
    try:
        out: list[tuple[float, str, str]] = []
        for fp in saved_models_path.glob('*_data.dat'):
            name = fp.name
            if not name.endswith('_data.dat'):
                continue
            stem = name[:-len('_data.dat')]
            if '_' not in stem:
                continue
            base, model_class_name = stem.rsplit('_', 1)
            if base.strip() == '' or model_class_name.strip() == '':
                continue
            try:
                mtime = fp.stat().st_mtime
            except Exception:
                mtime = 0.0
            out.append((mtime, model_class_name, base))
        out.sort(key=lambda x: x[0], reverse=True)
        return out
    except Exception:
        return []


def _choose_workspace_side() -> tuple[str, Path, Path]:
    ws = Path.cwd() / 'workspace'
    side = _choose_from_list('选择数据侧', ['SRC', 'DST'], default_index=0)
    if side == 'SRC':
        return side, ws / 'data_src', ws / 'data_src' / 'aligned'
    return side, ws / 'data_dst', ws / 'data_dst' / 'aligned'


def _choose_model_class_name() -> str:
    model_names = _list_model_class_names()
    if not model_names:
        raise RuntimeError('在 ./models/Model_* 下未找到任何模型')
    default_model = 'SAEHD' if 'SAEHD' in model_names else model_names[0]
    return _choose_from_list(
        '选择模型',
        model_names,
        default_index=(model_names.index(default_model) if default_model in model_names else 0),
    )


def run_trainer():
    """运行 mainscripts/Trainer.py（训练界面）。"""
    from core.leras import nn
    nn.initialize_main_env()

    from mainscripts import Trainer

    model_class_name = _choose_model_class_name()

    default_saved_models_path, default_src_path, default_dst_path = _default_workspace_paths()

    ws = Path.cwd() / 'workspace'

    saved_models_path = _input_path('模型目录', default_saved_models_path, create=True)
    training_data_src_path = _input_path('SRC aligned 目录', default_src_path, create=True)
    training_data_dst_path = _input_path('DST aligned 目录', default_dst_path, create=True)

    use_pretrain = _input_bool('启用预训练 (pretrain)?', default=False)
    pretraining_data_path = None
    if use_pretrain:
        pretraining_data_path = _input_path(
            '预训练 faceset 目录 pretraining_data_dir',
            ws / 'pretrain_faces',
            create=False,
        )

    # 快速自检：DFL 训练期望 aligned faces 为 DFLJPG/DFLPNG（包含元数据）。
    try:
        from core import pathex
        from DFLIMG import DFLIMG

        def _warn_if_not_dfl_aligned(p: Path, label: str):
            img_paths = pathex.get_image_paths(p)
            if len(img_paths) == 0:
                print(f"/!\\ {label}: {p} 为空。请先用 Extractor 生成 aligned faceset。")
                return
            checked = 0
            ok = 0
            for fp in img_paths[:10]:
                dfl = DFLIMG.load(Path(fp))
                checked += 1
                if dfl is not None and dfl.has_data():
                    ok += 1
            if ok == 0:
                print(f"/!\\ {label}: {p} 内前 {checked} 张都不是 DFL 图像(DFLJPG/DFLPNG)。")
                print("    训练需要包含 landmarks/face_type 等元数据的 aligned faceset；普通 jpg 不能直接训练。")

        _warn_if_not_dfl_aligned(training_data_src_path, 'SRC aligned')
        _warn_if_not_dfl_aligned(training_data_dst_path, 'DST aligned')
    except Exception:
        pass

    no_preview = _input_bool('不显示预览窗口？', default=False)
    cpu_only = _input_bool('强制 CPU 训练 (cpu_only)?', default=not _has_accelerator())
    silent_start = _input_bool('静默启动 (silent_start)?', default=False)

    force_model_name = _input_str('模型实例名 force_model_name (可空)', default='')
    force_model_name = force_model_name if force_model_name.strip() != '' else None

    # GPU idxs (optional)
    gpu_idxs_str = _input_str('指定 GPU idx (逗号分隔，可空)', default='')
    force_gpu_idxs = None
    if gpu_idxs_str.strip() != '':
        try:
            force_gpu_idxs = [int(x.strip()) for x in gpu_idxs_str.split(',') if x.strip() != '']
        except ValueError:
            print('GPU idx 解析失败，忽略该项。')
            force_gpu_idxs = None

    # macOS MPS: avoid double prompt by defaulting to device 0.
    if (not cpu_only) and force_gpu_idxs is None:
        try:
            if (not torch.cuda.is_available()) and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                force_gpu_idxs = [0]
        except Exception:
            pass

    Trainer.main(
        model_class_name=model_class_name,
        saved_models_path=Path(saved_models_path),
        training_data_src_path=Path(training_data_src_path),
        training_data_dst_path=Path(training_data_dst_path),
        pretraining_data_path=Path(pretraining_data_path) if pretraining_data_path is not None else None,
        no_preview=no_preview,
        force_model_name=force_model_name,
        force_gpu_idxs=force_gpu_idxs,
        cpu_only=cpu_only,
        silent_start=silent_start,
    )


def run_trainer_continue_latest():
    """继续训练（选择已有模型，默认选择最新）。

    Launcher UX 目标：不要重复询问 workspace 路径。
    """
    from core.leras import nn
    nn.initialize_main_env()

    from mainscripts import Trainer

    default_saved_models_path, default_src_path, default_dst_path = _default_workspace_paths()

    ws = Path.cwd() / 'workspace'

    saved_models_path = default_saved_models_path
    training_data_src_path = default_src_path
    training_data_dst_path = default_dst_path

    saved_models_path.mkdir(parents=True, exist_ok=True)

    saved = _list_saved_models(saved_models_path)
    if not saved:
        print(f"\n/!\\ {saved_models_path} 下未找到任何可继续训练的模型 (*_data.dat)。")
        print("    请先用‘训练（选择/新建模型）’创建并训练一次。\n")
        return

    items: list[str] = []
    for _, model_class, base in saved:
        items.append(f"{base} [{model_class}]")

    choice = _choose_from_list('继续训练：选择已有模型（默认最新）', items, default_index=0)
    # Parse back: "<base> [<ModelClass>]"
    if ' [' in choice and choice.endswith(']'):
        base = choice[: choice.rfind(' [')]
        model_class_name = choice[choice.rfind(' [') + 2 : -1]
    else:
        # 兜底：理论上不会发生
        base = choice
        model_class_name = _choose_model_class_name()

    # 这里尽量保持最少提示；预览默认仍可见。
    no_preview = _input_bool('不显示预览窗口？', default=False)

    use_pretrain = _input_bool('启用预训练 (pretrain)?', default=False)
    pretraining_data_path = None
    if use_pretrain:
        pretraining_data_path = _input_path(
            '预训练 faceset 目录 pretraining_data_dir',
            ws / 'pretrain_faces',
            create=False,
        )

    Trainer.main(
        model_class_name=model_class_name,
        saved_models_path=Path(saved_models_path),
        training_data_src_path=Path(training_data_src_path),
        training_data_dst_path=Path(training_data_dst_path),
        pretraining_data_path=Path(pretraining_data_path) if pretraining_data_path is not None else None,
        no_preview=no_preview,
        force_model_name=base,
        force_gpu_idxs=None,
        cpu_only=False,
        silent_start=True,
    )


def run_merger():
    from core.leras import nn
    nn.initialize_main_env()

    from mainscripts import Merger

    model_class_name = _choose_model_class_name()

    # 尽量对齐原版 DFL 的 Merger 体验：通常只需要选择 model 和 data_dst。
    # 其余路径（输出目录、mask 输出、aligned 目录）默认从 ./workspace 推导。
    ws = Path.cwd() / 'workspace'
    default_saved_models_path = ws / 'model'
    default_input_path = ws / 'data_dst'
    default_output_path = ws / 'data_dst' / 'merged'
    default_output_mask_path = ws / 'data_dst' / 'merged_mask'
    # Merger 需要读取“待合成帧”对应的对齐信息，因此默认使用 DST aligned。
    default_aligned_path = ws / 'data_dst' / 'aligned'

    saved_models_path = _input_path('模型目录', default_saved_models_path, create=False)
    input_path = _input_path('DST 帧目录（data_dst）', default_input_path, create=False)

    # 保持可脚本化：如需自定义路径，建议使用 `python main.py merge ...`
    output_path = default_output_path
    output_mask_path = default_output_mask_path
    aligned_path = default_aligned_path

    # 轻量自检：aligned 应当能匹配到 input_path 内的帧名（stem）。
    try:
        from core import pathex
        from DFLIMG import DFLIMG

        input_stems = {Path(p).stem for p in pathex.get_image_paths(input_path)}
        aligned_paths = pathex.get_image_paths(aligned_path)
        matched = 0
        checked = 0
        for ap in aligned_paths[:200]:
            dfl = DFLIMG.load(Path(ap))
            if dfl is None or not dfl.has_data():
                continue
            src_fn = dfl.get_source_filename()
            if src_fn is None:
                continue
            checked += 1
            if Path(src_fn).stem in input_stems:
                matched += 1
        if checked > 0 and matched == 0:
            print("/!\\ 提示：当前 aligned 目录似乎无法匹配到输入帧（可能选错了 aligned，通常应为 workspace/data_dst/aligned）。")
        print(f"将使用 aligned: {aligned_path}")
    except Exception:
        pass

    cpu_only = _input_bool('强制 CPU (cpu_only)?', default=not _has_accelerator())

    Merger.main(
        model_class_name=model_class_name,
        saved_models_path=saved_models_path,
        training_data_src_path=None,
        input_path=input_path,
        output_path=output_path,
        output_mask_path=output_mask_path,
        aligned_path=aligned_path,
        cpu_only=cpu_only,
    )


def run_extractor():
    from core.leras import nn
    nn.initialize_main_env()

    from mainscripts import Extractor

    side, input_default, output_default = _choose_workspace_side()

    input_path = _input_path(f'输入视频/图片目录 input_path ({side})', input_default, create=False)
    output_path = _input_path(f'输出 aligned 目录 output_path ({side})', output_default, create=True)

    manual_fix = _input_bool('自动切脸后，对缺脸帧进行手动修复 (manual_fix)?', default=False)

    Extractor.main(detector='s3fd', input_path=input_path, output_path=output_path, manual_fix=manual_fix)


def run_sorter():
    from mainscripts import Sorter
    _, _, aligned_default = _choose_workspace_side()
    input_path = _input_path('输入 faceset 目录 (aligned)', aligned_default, create=False)
    Sorter.main(input_path=input_path)


def run_faceset_resizer():
    from mainscripts import FacesetResizer
    _, _, aligned_default = _choose_workspace_side()
    input_path = _input_path('输入 faceset 目录 (aligned)', aligned_default, create=False)
    FacesetResizer.process_folder(input_path)


def run_faceset_enhancer():
    from core.leras import nn
    nn.initialize_main_env()

    from mainscripts import FacesetEnhancer
    _, _, aligned_default = _choose_workspace_side()
    input_path = _input_path('输入 faceset 目录 (aligned)', aligned_default, create=False)
    cpu_only = _input_bool('强制 CPU (cpu_only)?', default=not _has_accelerator())
    FacesetEnhancer.process_folder(input_path, cpu_only=cpu_only, force_gpu_idxs=None)


def run_util_menu():
    from mainscripts import Util
    from samplelib import PackedFaceset

    side, _, aligned_default = _choose_workspace_side()
    input_path = _input_path(f'输入目录（{side} faceset/aligned）', aligned_default, create=False)

    actions = [
        '添加 landmarks 调试图',
        '恢复 aligned 原始文件名',
        '保存 faceset 元数据 (meta.dat)',
        '恢复 faceset 元数据 (meta.dat)',
        '打包 faceset',
        '解包 faceset',
        '导出 faceset mask',
    ]
    action = _choose_from_list('Util', actions, default_index=0)

    if action == actions[0]:
        Util.add_landmarks_debug_images(input_path)
    elif action == actions[1]:
        Util.recover_original_aligned_filename(input_path)
    elif action == actions[2]:
        Util.save_faceset_metadata_folder(input_path)
    elif action == actions[3]:
        Util.restore_faceset_metadata_folder(input_path)
    elif action == actions[4]:
        PackedFaceset.pack(Path(input_path))
    elif action == actions[5]:
        PackedFaceset.unpack(Path(input_path))
    elif action == actions[6]:
        Util.export_faceset_mask(Path(input_path))


def run_videoed_menu():
    from mainscripts import VideoEd

    ws = Path.cwd() / 'workspace'
    actions = [
        'SRC：从视频提取帧（可设置 FPS）',
        'DST：从视频提取帧（全帧，不丢帧）',
        'SRC：裁剪视频',
        'DST：裁剪视频',
        'SRC：图片序列降噪',
        'DST：图片序列降噪',
        'SRC：由序列生成视频',
        'DST：由序列生成视频',
    ]
    action = _choose_from_list('VideoEd', actions, default_index=0)

    if action == actions[0]:
        input_file = _input_path('输入视频文件（SRC）', ws / 'data_src' / 'input.mp4', create=False)
        out_dir = _input_path('输出帧目录（SRC）', ws / 'data_src', create=True)
        VideoEd.extract_video(str(input_file), str(out_dir), fps=None)
    elif action == actions[1]:
        input_file = _input_path('输入视频文件（DST）', ws / 'data_dst' / 'input.mp4', create=False)
        out_dir = _input_path('输出帧目录（DST）', ws / 'data_dst', create=True)
        print('提示：DST 抽帧会强制使用原始全帧率（FPS=0），以保证合成不丢帧。')
        VideoEd.extract_video(str(input_file), str(out_dir), fps=0)
    elif action == actions[2]:
        input_file = _input_path('输入视频文件（SRC）', ws / 'data_src' / 'input.mp4', create=False)
        VideoEd.cut_video(str(input_file))
    elif action == actions[3]:
        input_file = _input_path('输入视频文件（DST）', ws / 'data_dst' / 'input.mp4', create=False)
        VideoEd.cut_video(str(input_file))
    elif action == actions[4]:
        in_dir = _input_path('输入序列目录（SRC）', ws / 'data_src', create=False)
        VideoEd.denoise_image_sequence(str(in_dir))
    elif action == actions[5]:
        in_dir = _input_path('输入序列目录（DST）', ws / 'data_dst', create=False)
        VideoEd.denoise_image_sequence(str(in_dir))
    elif action == actions[6]:
        in_dir = _input_path('输入序列目录（SRC）', ws / 'data_src', create=False)
        out_file = _input_path('输出视频文件（SRC）', ws / 'data_src' / 'output.mp4', create=True)
        VideoEd.video_from_sequence(str(in_dir), str(out_file))
    elif action == actions[7]:
        in_dir = _input_path('输入序列目录（DST）', ws / 'data_dst' / 'merged_out', create=False)
        out_file = _input_path('输出视频文件（DST）', ws / 'data_dst' / 'merged_out.mp4', create=True)
        VideoEd.video_from_sequence(str(in_dir), str(out_file))


def run_xseg_menu():
    ws = Path.cwd() / 'workspace'
    actions = [
        '训练 XSeg',
        'XSeg 编辑器',
        '应用已训练的 XSeg',
        '移除 XSeg 遮罩（masks）',
        '移除 XSeg 标注（labels）',
        '抓取带 XSeg 多边形标注的人脸',
    ]
    action = _choose_from_list('XSeg', actions, default_index=0)

    if action == actions[0]:
        from core.leras import nn
        nn.initialize_main_env()

        from mainscripts import Trainer

        # 使用工作区默认路径，与启动器其他入口保持一致。
        saved_models_path = ws / 'model'
        training_data_src_path = ws / 'data_src' / 'aligned'
        training_data_dst_path = ws / 'data_dst' / 'aligned'

        no_preview = _input_bool('不显示预览窗口？', default=False)

        Trainer.main(
            model_class_name='XSeg',
            saved_models_path=saved_models_path,
            training_data_src_path=training_data_src_path,
            training_data_dst_path=training_data_dst_path,
            no_preview=no_preview,
            force_gpu_idxs=None,
            cpu_only=False,
            silent_start=True,
        )
        return

    if action == actions[1]:
        try:
            from XSegEditor import XSegEditor
        except Exception as e:
            print('XSeg 编辑器需要 Qt 绑定（PyQt5 或 PySide6）。')
            print(f'当前环境导入失败：{type(e).__name__}: {e}')
            return
        side, _, aligned_default = _choose_workspace_side()
        input_dir = _input_path(f'输入 faceset 目录（{side} aligned）', aligned_default, create=False)
        XSegEditor.start(input_dir)
        return

    from mainscripts import XSegUtil
    side, _, aligned_default = _choose_workspace_side()
    input_dir = _input_path(f'输入 faceset 目录（{side} aligned）', aligned_default, create=False)

    if action == actions[2]:
        model_dir = _input_path('XSeg 模型目录', ws / 'model', create=False)
        XSegUtil.apply_xseg(input_dir, model_dir)
    elif action == actions[3]:
        XSegUtil.remove_xseg(input_dir)
    elif action == actions[4]:
        XSegUtil.remove_xseg_labels(input_dir)
    elif action == actions[5]:
        XSegUtil.fetch_xseg(input_dir)


def run_dev_test():
    from mainscripts import dev_misc
    ws = Path.cwd() / 'workspace'
    input_dir = _input_path('输入目录', ws / 'data_dst' / 'aligned', create=False)
    dev_misc.dev_gen_mask_files(str(input_dir))


def run_export_dfm():
    from core.leras import nn
    nn.initialize_main_env()

    from mainscripts import ExportDFM

    model_class_name = _choose_model_class_name()

    default_saved_models_path, _, _ = _default_workspace_paths()
    saved_models_path = _input_path('模型目录', default_saved_models_path, create=False)
    ExportDFM.main(model_class_name=model_class_name, saved_models_path=saved_models_path)


def show_system_info():
    """显示系统信息"""
    print("\n" + "=" * 70)
    print("系统信息")
    print("=" * 70)
    print()
    
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"PyTorch版本: {torch.__version__}")
    mps_available = False
    try:
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except Exception:
        mps_available = False

    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"MPS可用: {mps_available}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif mps_available:
        print("运行模式: Apple MPS")
    else:
        print("运行模式: CPU")
    
    print()
    print("=" * 70)


def main():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except Exception:
        pass

    print_banner()

    while True:
        print_menu()
        choice = input("请输入选项: ").strip()

        if choice == '0':
            print("\n再见！")
            break
        elif choice == '1':
            # 自动切脸（推荐）
            run_extractor()
        elif choice == '2':
            # 手动切脸（排错）
            from core.interact import interact as io
            io.log_info('进入手动切脸(排错)模式。')
            # run_extractor 内部会再次询问是否手动；这里强制走手动分支。
            # 为避免引入新 UX，这里复用 Extractor.main(detector='manual') 的路径。
            from core.leras import nn
            nn.initialize_main_env()
            from mainscripts import Extractor
            side, input_default, output_default = _choose_workspace_side()
            input_path = _input_path(f'输入视频/图片目录 input_path ({side})', input_default, create=False)
            output_path = _input_path(f'输出 aligned 目录 output_path ({side})', output_default, create=True)
            Extractor.main(detector='manual', input_path=input_path, output_path=output_path)
        elif choice == '3':
            run_sorter()
        elif choice == '4':
            run_faceset_resizer()
        elif choice == '5':
            run_faceset_enhancer()
        elif choice == '6':
            run_trainer()
        elif choice == '7':
            run_trainer_continue_latest()
        elif choice == '8':
            run_merger()
        elif choice == '9':
            run_export_dfm()
        elif choice == '10':
            run_util_menu()
        elif choice == '11':
            run_videoed_menu()
        elif choice == '12':
            run_xseg_menu()
        elif choice == '13':
            run_dev_test()
        elif choice == '14':
            show_system_info()
        else:
            print('无效选项，请重试。')


if __name__ == '__main__':
    main()
