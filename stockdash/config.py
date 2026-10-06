from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

APP_NAME = "Stock Radar"

# Mọi đường dẫn dữ liệu neo theo thư mục cài đặt, không phụ thuộc thư mục đang đứng
# khi chạy streamlit. Có thể chuyển dữ liệu sang ổ cục bộ bằng STOCKDASH_DATA_DIR.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("STOCKDASH_DATA_DIR", "") or (BASE_DIR / "data"))
PORTFOLIO_CSV = DATA_DIR / "portfolio.csv"
RUNTIME_CACHE_PATH = DATA_DIR / "runtime_state.pkl"
CATALOG_CSV = DATA_DIR / "symbol_catalog.csv"
VNSTOCK_CACHE_DIR = Path(os.getenv("VNSTOCK_CACHE_DIR", "") or (DATA_DIR / "cache_vnstock"))

# Danh sách mã thanh khoản / được theo dõi rộng rãi dùng cho bản thử nghiệm.
DEFAULT_UNIVERSE = [
    ("VCB", "HOSE", "Ngân hàng"), ("BID", "HOSE", "Ngân hàng"),
    ("CTG", "HOSE", "Ngân hàng"), ("TCB", "HOSE", "Ngân hàng"),
    ("MBB", "HOSE", "Ngân hàng"), ("VPB", "HOSE", "Ngân hàng"),
    ("ACB", "HOSE", "Ngân hàng"), ("STB", "HOSE", "Ngân hàng"),
    ("HDB", "HOSE", "Ngân hàng"), ("LPB", "HOSE", "Ngân hàng"),
    ("FPT", "HOSE", "Công nghệ"), ("CMG", "HOSE", "Công nghệ"),
    ("HPG", "HOSE", "Vật liệu"), ("HSG", "HOSE", "Vật liệu"),
    ("NKG", "HOSE", "Vật liệu"), ("DGC", "HOSE", "Hóa chất"),
    ("VNM", "HOSE", "Tiêu dùng"), ("MSN", "HOSE", "Tiêu dùng"),
    ("MWG", "HOSE", "Bán lẻ"), ("PNJ", "HOSE", "Bán lẻ"),
    ("SAB", "HOSE", "Tiêu dùng"), ("VIC", "HOSE", "Bất động sản"),
    ("VHM", "HOSE", "Bất động sản"), ("VRE", "HOSE", "Bất động sản"),
    ("KBC", "HOSE", "Bất động sản"), ("DXG", "HOSE", "Bất động sản"),
    ("DIG", "HOSE", "Bất động sản"), ("NVL", "HOSE", "Bất động sản"),
    ("GAS", "HOSE", "Năng lượng"), ("PLX", "HOSE", "Năng lượng"),
    ("PVD", "HOSE", "Năng lượng"), ("POW", "HOSE", "Điện - tiện ích"),
    ("GMD", "HOSE", "Logistics"), ("VSC", "HOSE", "Logistics"),
    ("SSI", "HOSE", "Chứng khoán"), ("HCM", "HOSE", "Chứng khoán"),
    ("VCI", "HOSE", "Chứng khoán"), ("VIX", "HOSE", "Chứng khoán"),
    ("VND", "HOSE", "Chứng khoán"), ("SHS", "HNX", "Chứng khoán"),
    ("PVS", "HNX", "Năng lượng"), ("IDC", "HNX", "BĐS khu công nghiệp"),
    ("CEO", "HNX", "Bất động sản"), ("MBS", "HNX", "Chứng khoán"),
    ("BSR", "UPCOM", "Năng lượng"), ("ACV", "UPCOM", "Hàng không"),
    ("VGI", "UPCOM", "Công nghệ"), ("OIL", "UPCOM", "Năng lượng"),
]

INDEX_SYMBOLS = {
    "VNINDEX": "VN-Index",
    "HNXINDEX": "HNX-Index",
    "UPCOMINDEX": "UPCoM-Index",
}

@dataclass(frozen=True)
class ScoreWeights:
    market: int = 20
    trend: int = 25
    momentum: int = 20
    volume: int = 15
    risk: int = 10
    attention: int = 10

DEFAULT_WEIGHTS = ScoreWeights()

SIGNAL_THRESHOLDS = {
    "MUA MẠNH": 82,
    "MUA": 72,
    "THEO DÕI": 62,
    "NẮM GIỮ": 45,
    "GIẢM TỶ TRỌNG": 32,
    "BÁN": 0,
}
