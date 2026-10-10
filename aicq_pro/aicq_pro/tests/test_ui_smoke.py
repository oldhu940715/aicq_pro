"""以『假 streamlit／假 plotly』執行 app.py，檢查各分頁程式路徑沒有執行期例外。
這不能證明畫面在真實 Streamlit 顯示正確（API 簽章差異、版面都未驗證），只能抓出 NameError/KeyError/型別錯誤。
"""
import os
import runpy
import sys
import tempfile
from unittest.mock import MagicMock


class _Ctx:
    """context manager；欄位物件的 widget 方法委派回主 stub，行為與真實 st.columns 回傳物件一致。"""
    def __init__(self, parent=None):
        self._parent = parent

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        if name.startswith("_") or self._parent is None:
            return lambda *a, **k: None
        return getattr(self._parent, name)


class _SS(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class _Cache:
    def __call__(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

    def clear(self):
        pass


class StubST:
    def __init__(self, radio_idx=0, press=()):
        self.session_state, self.radio_idx, self.press = _SS(), radio_idx, press
        self.sidebar = _Ctx(self)
        self.cache_data = _Cache()
        self.calls = []

    def radio(self, label, options, index=0, **k):
        return options[self.radio_idx]

    def selectbox(self, label, options, index=0, **k):
        return list(options)[index]

    def slider(self, label, mn, mx, value=None, **k):
        return value

    def checkbox(self, label, value=False, **k):
        return value

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, **k):
        return value

    def button(self, label, **k):
        return any(p in label for p in self.press)

    def columns(self, n):
        return [_Ctx(self) for _ in range(n if isinstance(n, int) else len(n))]

    def tabs(self, labels):
        return [_Ctx(self) for _ in labels]

    def spinner(self, *a, **k):
        return _Ctx(self)

    def expander(self, *a, **k):
        return _Ctx(self)

    def stop(self):
        raise RuntimeError("st.stop() 被呼叫")

    def rerun(self):
        pass

    def __getattr__(self, name):
        def f(*a, **k):
            self.calls.append(name)
        return f


def _run(radio_idx, press=()):
    stub = StubST(radio_idx, press)
    saved = {k: sys.modules.get(k) for k in ("streamlit", "plotly", "plotly.graph_objects", "plotly.subplots")}
    sys.modules.update({"streamlit": stub, "plotly": MagicMock(), "plotly.graph_objects": MagicMock(),
                        "plotly.subplots": MagicMock()})
    d = tempfile.mkdtemp()
    os.environ["AICQ_DB_PATH"] = os.path.join(d, "ui.db")
    try:
        runpy.run_path("app.py", run_name="__main__")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        os.environ.pop("AICQ_DB_PATH", None)
    return stub


def test_ui_live_mode_without_network_shows_offline_and_does_not_crash():
    s = _run(radio_idx=0)           # 沙盒沒有 ccxt／網路 → 必須顯示離線，而不是改用假資料
    assert "error" in s.calls and "plotly_chart" not in s.calls


def test_ui_synthetic_mode_all_tabs_and_buttons():
    s = _run(radio_idx=1, press=("執行回測", "更新持倉", "建立模擬單", "套用設定"))
    assert "warning" in s.calls and "dataframe" in s.calls and "plotly_chart" in s.calls
