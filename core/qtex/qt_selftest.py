def selftest() -> str:
    try:
        from . import qt_compat
    except Exception as e:
        return f"qt_import_error={type(e).__name__}: {e}"

    qt_binding = getattr(qt_compat, "QT_BINDING", "unknown")
    qt_version = getattr(qt_compat, "QT_VERSION_STR", None)
    pyqt_version = getattr(qt_compat, "PYQT_VERSION_STR", None)

    pyside_version = None
    if qt_binding == "PySide6":
        try:
            import PySide6  # type: ignore

            pyside_version = getattr(PySide6, "__version__", None)
        except Exception:
            pyside_version = None

    qapp_class = getattr(getattr(qt_compat, "QApplication", None), "__name__", None)

    return (
        f"qt_binding={qt_binding} "
        f"qt_version={qt_version} "
        f"pyqt_version={pyqt_version} "
        f"pyside_version={pyside_version} "
        f"qapp_class={qapp_class}"
    )
