from __future__ import annotations

import ast
import os
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from typing import Any

from . import model_store


@dataclass
class UiState:
    workspace: Path
    model_dir: Path
    expert_mode: bool = False


class MainWindow(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("DeepFaceLab Torch - GUI")

        self._state = UiState(
            workspace=(Path(__file__).resolve().parents[1] / "workspace").resolve(),
            model_dir=(Path(__file__).resolve().parents[1] / "workspace" / "model").resolve(),
            expert_mode=False,
        )

        self._models: list[model_store.ModelInfo] = []
        self._current_model: model_store.ModelInfo | None = None
        self._current_options: dict[str, Any] = {}
        self._current_defaults: dict[str, Any] = {}

        self._build_layout()
        self._apply_responsive_geometry()
        self._refresh_models_async()

    # ---------------- geometry / layout ----------------
    def _apply_responsive_geometry(self) -> None:
        # Ensure default size is good and controls never get hidden.
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        width = min(1280, max(1100, screen_w - 80))
        height = min(820, max(700, screen_h - 120))

        min_w = min(1100, max(920, screen_w - 120))
        min_h = min(680, max(560, screen_h - 180))

        self.geometry(f"{width}x{height}")
        self.minsize(min_w, min_h)

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Top toolbar (always visible)
        toolbar = ttk.Frame(self, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(5, weight=1)

        self.expert_var = tk.BooleanVar(value=False)
        expert_cb = ttk.Checkbutton(toolbar, text="专家模式", variable=self.expert_var, command=self._on_toggle_expert)
        expert_cb.grid(row=0, column=0, padx=(0, 12))

        ttk.Label(toolbar, text="Workspace:").grid(row=0, column=1, sticky="w")
        self.workspace_var = tk.StringVar(value=str(self._state.workspace))
        workspace_entry = ttk.Entry(toolbar, textvariable=self.workspace_var, width=44)
        workspace_entry.grid(row=0, column=2, sticky="ew", padx=(6, 6))
        ttk.Button(toolbar, text="选择...", command=self._choose_workspace).grid(row=0, column=3)

        ttk.Label(toolbar, text="ModelDir:").grid(row=0, column=4, sticky="w", padx=(12, 0))
        self.model_dir_var = tk.StringVar(value=str(self._state.model_dir))
        model_entry = ttk.Entry(toolbar, textvariable=self.model_dir_var, width=44)
        model_entry.grid(row=0, column=5, sticky="ew", padx=(6, 6))
        ttk.Button(toolbar, text="选择...", command=self._choose_model_dir).grid(row=0, column=6)

        ttk.Button(toolbar, text="刷新模型", command=self._refresh_models_async).grid(row=0, column=7, padx=(12, 0))

        # Main area (paned) - left list, right parameters/log
        main = ttk.PanedWindow(self, orient="horizontal")
        main.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        left = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=8)
        main.add(left, weight=1)
        main.add(right, weight=3)

        # Left: model list
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="模型列表", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.model_tree = ttk.Treeview(left, columns=("name", "iter"), show="headings", height=12)
        self.model_tree.heading("name", text="模型")
        self.model_tree.heading("iter", text="iter")
        self.model_tree.column("name", width=260, anchor="w")
        self.model_tree.column("iter", width=70, anchor="e")
        self.model_tree.grid(row=1, column=0, sticky="nsew")

        model_scroll = ttk.Scrollbar(left, orient="vertical", command=self.model_tree.yview)
        model_scroll.grid(row=1, column=1, sticky="ns")
        self.model_tree.configure(yscrollcommand=model_scroll.set)
        self.model_tree.bind("<<TreeviewSelect>>", self._on_select_model)

        # Right: vertical paned: parameters (top) + log (bottom)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        vpaned = ttk.PanedWindow(right, orient="vertical")
        vpaned.grid(row=0, column=0, sticky="nsew")

        self.params_frame = ttk.Labelframe(vpaned, text="参数（默认：普通模式）", padding=8)
        log_frame = ttk.Labelframe(vpaned, text="日志", padding=8)
        vpaned.add(self.params_frame, weight=3)
        vpaned.add(log_frame, weight=1)

        # Params: treeview with scrollbars (no editing yet, stable layout first)
        self.params_frame.columnconfigure(0, weight=1)
        self.params_frame.rowconfigure(0, weight=1)

        params_container = ttk.Frame(self.params_frame)
        params_container.grid(row=0, column=0, sticky="nsew")
        params_container.columnconfigure(0, weight=1)
        params_container.rowconfigure(0, weight=1)

        self.param_tree = ttk.Treeview(params_container, columns=("key", "value", "default"), show="headings")
        self.param_tree.heading("key", text="key")
        self.param_tree.heading("value", text="保存值")
        self.param_tree.heading("default", text="默认值")
        self.param_tree.column("key", width=220, anchor="w")
        self.param_tree.column("value", width=360, anchor="w")
        self.param_tree.column("default", width=220, anchor="w")
        self.param_tree.grid(row=0, column=0, sticky="nsew")

        param_y = ttk.Scrollbar(params_container, orient="vertical", command=self.param_tree.yview)
        param_y.grid(row=0, column=1, sticky="ns")
        param_x = ttk.Scrollbar(params_container, orient="horizontal", command=self.param_tree.xview)
        param_x.grid(row=1, column=0, sticky="ew")
        self.param_tree.configure(yscrollcommand=param_y.set, xscrollcommand=param_x.set)
        self.param_tree.bind("<<TreeviewSelect>>", self._on_select_param)

        # Editor panel (always visible)
        editor = ttk.LabelFrame(self.params_frame, text="选中项编辑（输入 Python 字面量）", padding=8)
        editor.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        editor.columnconfigure(1, weight=1)

        self.selected_key_var = tk.StringVar(value="")
        self.saved_value_var = tk.StringVar(value="")
        self.default_value_var = tk.StringVar(value="")
        self.edit_value_var = tk.StringVar(value="")

        ttk.Label(editor, text="key:").grid(row=0, column=0, sticky="w")
        ttk.Entry(editor, textvariable=self.selected_key_var, state="readonly", width=26).grid(
            row=0, column=1, sticky="ew", padx=(6, 6)
        )

        ttk.Label(editor, text="保存值:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(editor, textvariable=self.saved_value_var, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=(6, 6), pady=(6, 0)
        )

        ttk.Label(editor, text="默认值:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(editor, textvariable=self.default_value_var, state="readonly").grid(
            row=2, column=1, sticky="ew", padx=(6, 6), pady=(6, 0)
        )

        ttk.Label(editor, text="新值:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.edit_entry = ttk.Entry(editor, textvariable=self.edit_value_var)
        self.edit_entry.grid(row=3, column=1, sticky="ew", padx=(6, 6), pady=(6, 0))
        self.edit_entry.bind("<Return>", self._apply_edit_value)

        btns = ttk.Frame(editor)
        btns.grid(row=0, column=2, rowspan=4, sticky="ns", padx=(8, 0))

        self.apply_btn = ttk.Button(btns, text="应用(回车)", command=self._apply_edit_value)
        self.apply_btn.grid(row=0, column=0, sticky="ew")
        self.reset_btn = ttk.Button(btns, text="恢复默认", command=self._reset_to_default)
        self.reset_btn.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.save_btn = ttk.Button(btns, text="写回模型(备份)", command=self._save_options_to_model)
        self.save_btn.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.reload_btn = ttk.Button(btns, text="重新读取", command=self._reload_current_model)
        self.reload_btn.grid(row=3, column=0, sticky="ew", pady=(6, 0))

        self.params_frame.rowconfigure(1, weight=0)

        # Log: text with scrollbar, always visible
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=8, wrap="none")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_y = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_y.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_y.set)

        # Status bar (always visible)
        status = ttk.Frame(self, padding=(10, 6))
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(1, weight=1)

        ttk.Label(status, text="状态:").grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status, textvariable=self.status_var).grid(row=0, column=1, sticky="w")

    # ---------------- actions ----------------
    def _log(self, s: str) -> None:
        self.log_text.insert("end", s + "\n")
        self.log_text.see("end")

    def _on_toggle_expert(self) -> None:
        self._state.expert_mode = bool(self.expert_var.get())
        mode = "专家模式" if self._state.expert_mode else "普通模式"
        self.status_var.set(f"切换为：{mode}")
        self.params_frame.configure(text=f"参数（默认：{'专家模式' if self._state.expert_mode else '普通模式'}）")
        self._update_editor_state()

    def _choose_workspace(self) -> None:
        p = filedialog.askdirectory(initialdir=self.workspace_var.get() or os.getcwd())
        if not p:
            return
        self._state.workspace = Path(p).expanduser().resolve()
        self.workspace_var.set(str(self._state.workspace))
        # If model_dir is under workspace/model, update it automatically.
        guess_model = self._state.workspace / "model"
        if guess_model.exists():
            self._state.model_dir = guess_model
            self.model_dir_var.set(str(self._state.model_dir))
        self._refresh_models_async()

    def _choose_model_dir(self) -> None:
        p = filedialog.askdirectory(initialdir=self.model_dir_var.get() or os.getcwd())
        if not p:
            return
        self._state.model_dir = Path(p).expanduser().resolve()
        self.model_dir_var.set(str(self._state.model_dir))
        self._refresh_models_async()

    def _refresh_models_async(self) -> None:
        # Avoid freezing UI when scanning lots of files.
        self.status_var.set("正在扫描模型...")
        self._log("[GUI] 扫描模型目录...")

        def worker() -> None:
            try:
                model_dir = Path(self.model_dir_var.get()).expanduser().resolve()
                models = model_store.scan_models(model_dir)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("扫描失败", str(e)))
                models = []

            self.after(0, lambda: self._apply_models(models))

        threading.Thread(target=worker, name="scan-models", daemon=True).start()

    def _apply_models(self, models: list[model_store.ModelInfo]) -> None:
        self._models = models

        # Clear list
        for item in self.model_tree.get_children(""):
            self.model_tree.delete(item)

        for m in models:
            self.model_tree.insert("", "end", iid=m.data_path.as_posix(), values=(m.display_name, m.iter))

        self.status_var.set(f"模型数：{len(models)}")
        self._log(f"[GUI] 发现模型：{len(models)}")

        # Auto-select first model if any
        if models:
            first_id = models[0].data_path.as_posix()
            try:
                self.model_tree.selection_set(first_id)
                self.model_tree.focus(first_id)
                self.model_tree.see(first_id)
                self._load_model_by_path(Path(first_id))
            except Exception:
                pass
        else:
            self._current_model = None
            self._render_params({}, {})

    def _on_select_model(self, _event: object | None = None) -> None:
        sel = self.model_tree.selection()
        if not sel:
            return
        p = Path(sel[0])
        self._load_model_by_path(p)

    def _load_model_by_path(self, data_path: Path) -> None:
        try:
            m = next((x for x in self._models if x.data_path.resolve() == data_path.resolve()), None)
            if m is None:
                return
            self._current_model = m
            model_dir = Path(self.model_dir_var.get()).expanduser().resolve()
            self._current_options = model_store.read_options(m.data_path)
            self._current_defaults = model_store.read_default_options(model_dir, m.model_class)
            self._render_params(self._current_options, self._current_defaults)
            self.status_var.set(f"已加载：{m.display_name}  iter={m.iter}")
            self._clear_editor_fields()
            self._update_editor_state()
        except Exception as e:
            messagebox.showerror("读取模型失败", str(e))

    def _render_params(self, options: dict[str, Any], defaults: dict[str, Any]) -> None:
        # Clear
        for item in self.param_tree.get_children(""):
            self.param_tree.delete(item)

        keys = sorted(set(options.keys()) | set(defaults.keys()))
        for k in keys:
            v = options.get(k, "")
            d = defaults.get(k, "")
            self.param_tree.insert("", "end", values=(str(k), self._format_value(v), self._format_value(d)))

    # ---------------- param editing ----------------
    _LOCKED_KEYS_NORMAL_MODE = {"pretrain"}

    def _clear_editor_fields(self) -> None:
        self.selected_key_var.set("")
        self.saved_value_var.set("")
        self.default_value_var.set("")
        self.edit_value_var.set("")

    def _get_selected_param_key(self) -> str | None:
        sel = self.param_tree.selection()
        if not sel:
            return None
        item = sel[0]
        vals = self.param_tree.item(item, "values")
        if not vals:
            return None
        return str(vals[0])

    def _on_select_param(self, _event: object | None = None) -> None:
        key = self._get_selected_param_key()
        if not key:
            self._clear_editor_fields()
            self._update_editor_state()
            return

        saved = self._current_options.get(key, "")
        default = self._current_defaults.get(key, "")
        self.selected_key_var.set(key)
        self.saved_value_var.set(self._format_value(saved))
        self.default_value_var.set(self._format_value(default))

        # Prefill edit with repr-like value so user sees type intent.
        self.edit_value_var.set(self._format_edit_prefill(saved if key in self._current_options else default))
        self._update_editor_state()

    def _update_editor_state(self) -> None:
        has_model = self._current_model is not None
        key = self.selected_key_var.get().strip()
        locked = (not self._state.expert_mode) and (key in self._LOCKED_KEYS_NORMAL_MODE)

        state_normal = "normal" if (has_model and key and not locked) else "disabled"
        self.edit_entry.configure(state=state_normal)
        self.apply_btn.configure(state=state_normal)
        self.reset_btn.configure(state="normal" if (has_model and key) else "disabled")
        self.save_btn.configure(state="normal" if has_model else "disabled")
        self.reload_btn.configure(state="normal" if has_model else "disabled")

        if has_model and key and locked:
            self.status_var.set(f"普通模式下禁止修改：{key}（切换专家模式可强改）")

    @staticmethod
    def _format_edit_prefill(v: Any) -> str:
        if v == "":
            return ""
        if v is None:
            return "None"
        if isinstance(v, str):
            # Let user decide if they want quotes; keep raw string as-is.
            return v
        try:
            return repr(v)
        except Exception:
            return str(v)

    @staticmethod
    def _parse_value(text: str) -> Any:
        s = text.strip()
        if s == "":
            return ""
        low = s.lower()
        if low in {"true", "false"}:
            return low == "true"
        if low in {"none", "null"}:
            return None
        # Fast numeric path
        try:
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                return int(s)
        except Exception:
            pass
        try:
            if any(c in s for c in (".", "e", "E")):
                return float(s)
        except Exception:
            pass

        try:
            return ast.literal_eval(s)
        except Exception:
            return s

    def _apply_edit_value(self, _event: object | None = None) -> None:
        if self._current_model is None:
            return
        key = self.selected_key_var.get().strip()
        if not key:
            return
        if (not self._state.expert_mode) and (key in self._LOCKED_KEYS_NORMAL_MODE):
            messagebox.showwarning("已锁定", f"普通模式下不能修改：{key}\n\n如确需修改，请切换专家模式。")
            return

        raw = self.edit_value_var.get()
        new_val = self._parse_value(raw)
        self._current_options[key] = new_val
        self._render_params(self._current_options, self._current_defaults)
        self.status_var.set(f"已应用：{key}")
        self._log(f"[GUI] 已应用参数：{key} = {self._format_value(new_val)}")

    def _reset_to_default(self) -> None:
        if self._current_model is None:
            return
        key = self.selected_key_var.get().strip()
        if not key:
            return
        if key not in self._current_defaults:
            messagebox.showinfo("无默认值", f"该项没有默认值：{key}")
            return
        if (not self._state.expert_mode) and (key in self._LOCKED_KEYS_NORMAL_MODE):
            messagebox.showwarning("已锁定", f"普通模式下不能修改：{key}\n\n如确需修改，请切换专家模式。")
            return

        self._current_options[key] = self._current_defaults[key]
        self._render_params(self._current_options, self._current_defaults)
        self.status_var.set(f"已恢复默认：{key}")
        self._log(f"[GUI] 已恢复默认：{key}")

    def _reload_current_model(self) -> None:
        if self._current_model is None:
            return
        self._load_model_by_path(self._current_model.data_path)

    def _save_options_to_model(self) -> None:
        if self._current_model is None:
            return
        m = self._current_model

        if self._state.expert_mode:
            ok = messagebox.askokcancel(
                "确认写回（专家模式）",
                "你正在专家模式下写回模型参数。\n"
                "这可能导致训练无法继续或结果异常。\n\n"
                "将自动创建 .bak 备份。\n\n"
                "确定写回吗？",
            )
            if not ok:
                return

        try:
            backup = model_store.write_options(m.data_path, self._current_options, make_backup=True)
        except Exception as e:
            messagebox.showerror("写回失败", str(e))
            return

        self.status_var.set("写回成功（已备份）")
        if backup is not None:
            self._log(f"[GUI] 写回成功，备份：{backup}")
        else:
            self._log("[GUI] 写回成功")
        # Reload to reflect persisted state
        self._reload_current_model()

    @staticmethod
    def _format_value(v: Any) -> str:
        try:
            if isinstance(v, (dict, list, tuple)):
                s = str(v)
            else:
                s = "" if v is None else str(v)
        except Exception:
            s = "<unrepr>"
        if len(s) > 260:
            return s[:260] + "..."
        return s


def main() -> int:
    try:
        app = MainWindow()
        app.mainloop()
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
