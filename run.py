#!/usr/bin/env python3
"""DeepFaceLab Torch 版本启动器（按 bat/ 脚本重构）。

目标：参考 `bat/` 下的原版运行脚本，把其“命令编排”搬到跨平台 Python 启动器里。

设计原则：
- 菜单项 = `bat/` 目录中的 .bat 文件（保持与原流程同步）
- 执行逻辑 = 解析 .bat 中的 `mkdir/rmdir/python main.py/start` 等指令并执行等价操作
- 可脚本化：支持 `--list` / `--run` / `--workspace` / `--internal`

说明：`bat/` 里的 .bat 原本依赖 `_internal\\setenv.bat` 设置 `%WORKSPACE%/%DFL_ROOT%/%PYTHON_EXECUTABLE%/%INTERNAL%`。
本启动器在 macOS/Linux 上用同名变量的默认推导值替代，并允许通过参数/环境变量覆盖。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class LaunchContext:
    dfl_root: Path
    workspace: Path
    internal: Path
    python_executable: str

    def vars(self) -> dict[str, str]:
        # bat 变量名（不区分大小写）；为了简单全大写。
        return {
            'DFL_ROOT': str(self.dfl_root),
            'WORKSPACE': str(self.workspace),
            'INTERNAL': str(self.internal),
            'PYTHON_EXECUTABLE': self.python_executable,
        }


_VAR_PATTERN = re.compile(r'%(?P<name>[A-Za-z_][A-Za-z0-9_]*)%')


def _expand_bat_vars(s: str, ctx: LaunchContext) -> str:
    mapping = ctx.vars()

    def repl(m: re.Match[str]) -> str:
        name = m.group('name').upper()
        return mapping.get(name, m.group(0))

    out = _VAR_PATTERN.sub(repl, s)
    # bat 常用反斜杠路径；在 macOS/Linux 上统一替换为 POSIX。
    out = out.replace('\\', '/')
    return out


def _join_caret_lines(lines: Iterable[str]) -> list[str]:
    """合并 bat 的 ^ 行续写。"""
    out: list[str] = []
    buf = ''
    for raw in lines:
        line = raw.rstrip('\r\n')
        if buf:
            line = buf + line.lstrip()
            buf = ''

        stripped = line.rstrip()
        if stripped.endswith('^'):
            buf = stripped[:-1].rstrip() + ' '
            continue

        out.append(line)

    if buf:
        out.append(buf.rstrip())
    return out


def _tokenize_bat_command(line: str) -> list[str]:
    """非常轻量的 bat 命令拆词：支持双引号包裹。"""
    tokens: list[str] = []
    current: list[str] = []
    in_quotes = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"':
            in_quotes = not in_quotes
            i += 1
            continue
        if (not in_quotes) and ch.isspace():
            if current:
                tokens.append(''.join(current))
                current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    if current:
        tokens.append(''.join(current))
    return tokens


def _safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _open_path(path: Path) -> None:
    # macOS / Linux：尽量打开目录或文件。
    if sys.platform == 'darwin':
        subprocess.run(['open', str(path)], check=False)
        return
    subprocess.run(['xdg-open', str(path)], check=False)


def _run_python_main(args: list[str], ctx: LaunchContext) -> int:
    main_py = ctx.dfl_root / 'main.py'
    if not main_py.exists():
        raise FileNotFoundError(f'未找到 main.py: {main_py}')

    cmd = [ctx.python_executable, str(main_py), *args]
    print('\n$ ' + ' '.join(_quote_for_display(x) for x in cmd))
    try:
        p = subprocess.run(cmd)
        return int(p.returncode)
    except KeyboardInterrupt:
        return 130


def _quote_for_display(s: str) -> str:
    if re.search(r'\s', s):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _interpret_bat_file(bat_path: Path, ctx: LaunchContext) -> int:
    """执行一个 bat 脚本中“可跨平台等价”的子集。"""
    try:
        text = bat_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f'无法读取 {bat_path.name}: {type(e).__name__}: {e}')
        return 1

    lines = _join_caret_lines(text.splitlines(True))
    exit_code = 0

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        low = line.lower()

        # 常见无意义指令
        if low.startswith('@echo'):
            continue
        if low == 'pause':
            continue
        if low.startswith('call ') and 'setenv.bat' in low:
            continue

        # 处理形如: if %errorlevel% NEQ 0 ( pause )
        if low.startswith('if ') and 'errorlevel' in low:
            continue

        # mkdir / rmdir（只实现 bat 脚本里用到的基本形态）
        if low.startswith('mkdir '):
            tokens = _tokenize_bat_command(line)
            # mkdir "path" 2>nul
            if len(tokens) >= 2:
                p = Path(_expand_bat_vars(tokens[1], ctx)).expanduser()
                _mkdir(p)
            continue

        if low.startswith('rmdir '):
            tokens = _tokenize_bat_command(line)
            # rmdir "path" /s /q 2>nul
            if len(tokens) >= 2:
                p = Path(_expand_bat_vars(tokens[1], ctx)).expanduser()
                _safe_rmtree(p)
            continue

        # start "" ... <path>   （在 bat 中主要用于打开 aligned 目录）
        if low.startswith('start '):
            tokens = _tokenize_bat_command(line)
            if tokens:
                # 最后一个 token 往往是要打开的路径
                target = _expand_bat_vars(tokens[-1], ctx)
                _open_path(Path(target).expanduser())
            continue

        # 识别 python main.py 调用：
        # "%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" <args...>
        if 'main.py' in low:
            # bat 脚本通常不会写 python 字面量，而是写 "%PYTHON_EXECUTABLE%"。
            tokens = _tokenize_bat_command(line)
            tokens = [_expand_bat_vars(t, ctx) for t in tokens]
            # 在 tokens 中找 main.py 位置
            main_idx = None
            for i, t in enumerate(tokens):
                if t.endswith('main.py') or t.endswith('/main.py'):
                    main_idx = i
                    break
            if main_idx is None:
                continue

            args = tokens[main_idx + 1 :]
            # 过滤掉 bat 的重定向 "2>nul" 等
            args = [a for a in args if not a.startswith('2>') and a.lower() != '2>nul']

            rc = _run_python_main(args, ctx)
            if rc != 0:
                exit_code = rc
                break
            continue

        # 其他命令：不做等价执行，但保留可见性
        # （例如 xnviewmp 等 Windows 工具）
        # print(f"(跳过) {line}")

    return exit_code


def _list_bat_scripts(bat_dir: Path) -> list[Path]:
    if not bat_dir.exists():
        return []
    # 仅列出顶层 .bat 文件（与当前仓库结构一致）
    scripts = [p for p in bat_dir.iterdir() if p.is_file() and p.suffix.lower() == '.bat']
    scripts.sort(key=lambda p: p.name)
    return scripts


def _display_script_name(p: Path) -> str:
    return p.stem


def _choose_script_interactively(scripts: list[Path]) -> Path | None:
    if not scripts:
        return None

    print('=' * 70)
    print('DeepFaceLab Torch 启动器（按 bat/ 脚本）')
    print('=' * 70)
    print('选择一个流程脚本执行（与 Windows bat 名称一致）：')
    print()
    for i, p in enumerate(scripts, start=1):
        print(f' {i:2d}. {_display_script_name(p)}')
    print('  0. 退出')
    print()

    while True:
        s = input('请输入选项：').strip()
        if s == '0':
            return None
        try:
            idx = int(s)
            if 1 <= idx <= len(scripts):
                return scripts[idx - 1]
        except ValueError:
            pass


def _resolve_ctx(workspace: str | None, internal: str | None) -> LaunchContext:
    dfl_root = Path(__file__).resolve().parent

    ws = workspace or os.environ.get('DFL_WORKSPACE') or str(dfl_root / 'workspace')
    internal_default = None
    for cand in (dfl_root / 'bat' / '_internal', dfl_root / 'internal'):
        if cand.exists():
            internal_default = str(cand)
            break
    if internal_default is None:
        internal_default = str(dfl_root / 'internal')
    internal_val = internal or os.environ.get('DFL_INTERNAL') or internal_default

    return LaunchContext(
        dfl_root=dfl_root,
        workspace=Path(ws).expanduser().resolve(),
        internal=Path(internal_val).expanduser().resolve(),
        python_executable=sys.executable,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--list', action='store_true', help='列出可用的 bat 脚本并退出')
    parser.add_argument('--run', dest='run_name', default=None, help='执行指定 bat（文件名或包含其子串）')
    parser.add_argument('--workspace', default=None, help='覆盖 %%WORKSPACE%%（默认：./workspace 或 $DFL_WORKSPACE）')
    parser.add_argument('--internal', default=None, help='覆盖 %%INTERNAL%%（默认：./internal 或 $DFL_INTERNAL）')

    args = parser.parse_args(argv)

    ctx = _resolve_ctx(args.workspace, args.internal)
    bat_dir = ctx.dfl_root / 'bat'
    scripts = _list_bat_scripts(bat_dir)

    if args.list:
        for p in scripts:
            print(_display_script_name(p))
        return 0

    if args.run_name:
        needle_raw = args.run_name.strip().lower()
        needle = re.sub(r'\.bat$', '', needle_raw)
        picked = None
        for p in scripts:
            name_l = p.name.lower()
            stem_l = p.stem.lower()
            if (
                name_l == needle_raw
                or needle_raw in name_l
                or stem_l == needle
                or needle in stem_l
            ):
                picked = p
                break
        if picked is None:
            print(f'未找到脚本：{args.run_name}')
            return 2
        return _interpret_bat_file(picked, ctx)

    picked = _choose_script_interactively(scripts)
    if picked is None:
        return 0
    return _interpret_bat_file(picked, ctx)


if __name__ == '__main__':
    raise SystemExit(main())
