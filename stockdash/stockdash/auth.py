"""Cổng mật khẩu tùy chọn cho Dashboard.

Chỉ bật khi có DASHBOARD_PASSWORD (biến môi trường hoặc st.secrets). Chạy trên máy cá nhân
mà không đặt biến này thì không hiện màn hình đăng nhập. BẮT BUỘC đặt khi mở ra Internet/LAN:
trang chứa danh mục cá nhân và cho phép lưu API key.
"""
from __future__ import annotations

import hmac
import os
import time

import streamlit as st


def _configured_password() -> str:
    pw = os.getenv("DASHBOARD_PASSWORD", "").strip()
    if pw:
        return pw
    try:
        return str(st.secrets.get("DASHBOARD_PASSWORD", "") or "").strip()
    except Exception:
        return ""


def require_password() -> None:
    expected = _configured_password()
    if not expected or st.session_state.get("_auth_ok"):
        return

    st.markdown(
        """<div class="nm-header" style="max-width:460px;margin:12vh auto 1rem">
          <div class="nm-logo">SR</div>
          <div><div class="nm-title">Stock Radar</div>
          <div class="nm-sub">Nhập mật khẩu để tiếp tục</div></div></div>""",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        with st.form("login_form", clear_on_submit=True):
            pw = st.text_input("Mật khẩu", type="password")
            ok = st.form_submit_button("Đăng nhập", type="primary", width="stretch")
        if ok:
            fails = int(st.session_state.get("_auth_fails", 0))
            if fails:
                time.sleep(min(2 ** fails, 10))  # làm chậm dò mật khẩu
            if hmac.compare_digest(pw.encode("utf-8"), expected.encode("utf-8")):
                st.session_state["_auth_ok"] = True
                st.session_state["_auth_fails"] = 0
                st.rerun()
            st.session_state["_auth_fails"] = fails + 1
            st.error("Mật khẩu chưa đúng.")
    st.stop()
