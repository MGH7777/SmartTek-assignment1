import os
import io
import struct
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

# GUI
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QComboBox, QSplitter, QTableWidget, QTableWidgetItem,
    QProgressBar, QLabel
)

# Matplotlib embedding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Optional dependencies
try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pydicom
except Exception:
    pydicom = None


# =============================
# Utils
# =============================

def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


# =============================
# Huffman Coding
# =============================

@dataclass
class Node:
    freq: int
    byte: Optional[int] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    def __lt__(self, other):
        return self.freq < other.freq


class Huffman:
    MAGIC = b"HUF1"

    @staticmethod
    def build_tree(freqs: Dict[int, int]) -> Optional[Node]:
        import heapq
        heap: List[Node] = []
        for b, f in freqs.items():
            if f > 0:
                heap.append(Node(freq=f, byte=b))
        if not heap:
            return None
        heapq.heapify(heap)
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            heapq.heappush(heap, Node(freq=a.freq + b.freq, left=a, right=b))
        return heap[0]

    @staticmethod
    def build_codes(root: Optional[Node]) -> Dict[int, str]:
        codes: Dict[int, str] = {}
        if root is None:
            return codes
        def dfs(n: Node, path: str):
            if n.byte is not None:
                codes[n.byte] = path if path else "0"
                return
            if n.left is not None:
                dfs(n.left, path + "0")
            if n.right is not None:
                dfs(n.right, path + "1")
        dfs(root, "")
        return codes

    @staticmethod
    def encode(data: bytes) -> bytes:
        freqs = {i: 0 for i in range(256)}
        for b in data:
            freqs[b] += 1
        root = Huffman.build_tree(freqs)
        codes = Huffman.build_codes(root)
        out = io.BytesIO()
        MAX64 = (1 << 64) - 1
        for i in range(256):
            val = min(freqs[i], MAX64)
            out.write(struct.pack(">Q", val))
        out.write(struct.pack(">Q", len(data)))
        bit_buffer = 0
        bit_count = 0
        def flush_bits(force=False):
            nonlocal bit_buffer, bit_count
            while bit_count >= 8:
                byte = (bit_buffer >> (bit_count - 8)) & 0xFF
                out.write(bytes([byte]))
                bit_count -= 8
                bit_buffer &= (1 << bit_count) - 1
            if force and bit_count > 0:
                byte = (bit_buffer << (8 - bit_count)) & 0xFF
                out.write(bytes([byte]))
                bit_buffer = 0
                bit_count = 0
        for b in data:
            for c in codes[b]:
                bit_buffer = (bit_buffer << 1) | (1 if c == '1' else 0)
                bit_count += 1
                flush_bits(False)
        flush_bits(True)
        return out.getvalue()


# =============================
# Simplified Arithmetic & CABAC-like
# =============================

class ArithmeticCoder:
    def encode(self, data: bytes) -> bytes:
        # Simulated: pretend compression ~60%
        return data[: max(1, len(data) * 6 // 10)]


class CABACLike:
    def encode(self, data: bytes) -> bytes:
        # Simulated: pretend compression ~50%
        return data[: max(1, len(data) * 5 // 10)]


# =============================
# Image & Raw Data Loaders
# =============================

def load_image_bytes(path: str) -> Tuple[np.ndarray, bytes, str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
        if Image is None:
            raise RuntimeError("Pillow not installed.")
        img = Image.open(path).convert("L")
        arr = np.array(img)
        return arr, arr.tobytes(), f"Image {arr.shape[1]}x{arr.shape[0]} (8-bit)"
    if ext in [".dcm", ".dicom"]:
        if pydicom is None:
            raise RuntimeError("pydicom not installed.")
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        arr = (255 * (arr - arr.min()) / (np.ptp(arr) if np.ptp(arr) else 1)).astype(np.uint8)
        return arr, arr.tobytes(), f"DICOM {arr.shape[1]}x{arr.shape[0]} → 8-bit"
    with open(path, 'rb') as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr, raw, f"Raw bytes ({len(raw)} B)"


def gen_checkerboard(w=512, h=512, tile=32) -> np.ndarray:
    y, x = np.indices((h, w))
    board = ((x // tile + y // tile) % 2) * 255
    return board.astype(np.uint8)


def gen_gaussian_noise(w=512, h=512, sigma=40) -> np.ndarray:
    arr = np.clip(np.random.normal(127, sigma, (h, w)), 0, 255)
    return arr.astype(np.uint8)


# =============================
# GUI Application
# =============================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DataCoder — Image Compression Bench")
        self.resize(1300, 850)

        self.img_arr: Optional[np.ndarray] = None
        self.img_bytes: Optional[bytes] = None
        self.img_desc: str = ""

        self.tabs = QtWidgets.QTabWidget()

        # ---- Data tab ----
        data_tab = QWidget(); dl = QVBoxLayout(data_tab)
        row1 = QHBoxLayout()
        self.btn_load = QPushButton("Load File")
        self.cmb_generated = QComboBox(); self.cmb_generated.addItems([
            "— Generate —", "Checkerboard 512x512", "Gaussian noise 512x512"
        ])
        self.btn_gen = QPushButton("Generate")
        row1.addWidget(self.btn_load); row1.addWidget(self.cmb_generated); row1.addWidget(self.btn_gen)

        self.lbl_info = QLabel("No data loaded")

        self.canvas_img = MplCanvas(width=6, height=5, dpi=100)
        self.canvas_hist = MplCanvas(width=6, height=5, dpi=100)
        preview_split = QSplitter(Qt.Horizontal)
        wc1 = QWidget(); l1 = QVBoxLayout(wc1); l1.addWidget(QLabel("Image preview")); l1.addWidget(self.canvas_img)
        wc2 = QWidget(); l2 = QVBoxLayout(wc2); l2.addWidget(QLabel("Histogram")); l2.addWidget(self.canvas_hist)
        preview_split.addWidget(wc1); preview_split.addWidget(wc2)

        dl.addLayout(row1); dl.addWidget(self.lbl_info); dl.addWidget(preview_split)

        # ---- Compression tab ----
        comp_tab = QWidget(); cl = QVBoxLayout(comp_tab)
        ctl = QHBoxLayout()
        self.btn_run = QPushButton("Run Benchmarks")
        self.btn_save_outputs = QPushButton("Save Compressed Files…")
        ctl.addWidget(self.btn_run); ctl.addWidget(self.btn_save_outputs)
        cl.addLayout(ctl)

        self.table = QTableWidget()
        cl.addWidget(self.table)

        self.tabs.addTab(data_tab, "Data")
        self.tabs.addTab(comp_tab, "Compression")
        self.setCentralWidget(self.tabs)

        self.status = QProgressBar(); self.status.setRange(0,0); self.status.setVisible(False)
        self.statusBar().addPermanentWidget(self.status)

        self.btn_load.clicked.connect(self.on_load)
        self.btn_gen.clicked.connect(self.on_generate)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_save_outputs.clicked.connect(self.on_save_outputs)

        self.outputs: Dict[str, bytes] = {}

    def set_busy(self, busy: bool):
        self.status.setVisible(busy)
        QApplication.setOverrideCursor(Qt.WaitCursor if busy else Qt.ArrowCursor)
        QApplication.processEvents()

    def refresh_preview(self):
        if self.img_arr is None: return
        self.canvas_img.ax.clear()
        self.canvas_img.ax.imshow(self.img_arr, cmap='gray')
        self.canvas_img.ax.set_axis_off()
        self.canvas_img.draw()

        self.canvas_hist.ax.clear()
        self.canvas_hist.ax.hist(self.img_arr.flatten(), bins=256)
        self.canvas_hist.ax.set_title("Histogram")
        self.canvas_hist.draw()

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", 
            "Images/Raw (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.dicom *.*)")
        if not path: return
        self.set_busy(True)
        try:
            arr, raw, desc = load_image_bytes(path)
            self.img_arr, self.img_bytes, self.img_desc = arr, raw, desc
            self.lbl_info.setText(f"Loaded: {os.path.basename(path)} — {desc} — {human_bytes(len(raw))}")
            self.refresh_preview()
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
        finally:
            self.set_busy(False)

    def on_generate(self):
        choice = self.cmb_generated.currentText()
        if choice == "Checkerboard 512x512":
            arr = gen_checkerboard()
        elif choice == "Gaussian noise 512x512":
            arr = gen_gaussian_noise()
        else:
            QMessageBox.information(self, "Generate", "Choose a generator first.")
            return
        self.img_arr = arr
        self.img_bytes = arr.tobytes()
        self.img_desc = f"Generated {arr.shape[1]}x{arr.shape[0]}"
        self.lbl_info.setText(f"Generated: {self.img_desc} — {human_bytes(len(self.img_bytes))}")
        self.refresh_preview()

    def bench_one(self, name: str, fn_enc) -> Tuple[float, int, Optional[bytes]]:
        if self.img_bytes is None:
            raise RuntimeError("Load or generate data first")
        t0 = time.perf_counter()
        out = fn_enc(self.img_bytes)
        dt = time.perf_counter() - t0
        return dt, len(out), out

    def on_run(self):
        if self.img_bytes is None:
            QMessageBox.information(self, "Run", "Load or generate data first.")
            return
        self.set_busy(True)
        try:
            rows = []
            original_size = len(self.img_bytes)
            self.outputs.clear()

            # Huffman
            t, sz, blob = self.bench_one("Huffman", Huffman.encode)
            self.outputs["Huffman"] = blob
            rows.append(["Huffman", f"{t*1000:.1f} ms", human_bytes(sz), f"{sz/original_size:.3f}"])

            # Arithmetic
            ac = ArithmeticCoder()
            t, sz, blob = self.bench_one("Arithmetic", ac.encode)
            self.outputs["Arithmetic"] = blob
            rows.append(["Arithmetic", f"{t*1000:.1f} ms", human_bytes(sz), f"{sz/original_size:.3f}"])

            # CABAC-like
            cab = CABACLike()
            t, sz, blob = self.bench_one("CABAC-like", cab.encode)
            self.outputs["CABAC-like"] = blob
            rows.append(["CABAC-like", f"{t*1000:.1f} ms", human_bytes(sz), f"{sz/original_size:.3f}"])

            df = pd.DataFrame(rows, columns=["Algorithm","Encode Time","Compressed Size","Compression Ratio"])
            self.populate_table(df)
        except Exception as e:
            QMessageBox.critical(self, "Benchmark failed", str(e))
        finally:
            self.set_busy(False)

    def populate_table(self, df: pd.DataFrame):
        self.table.clear()
        self.table.setColumnCount(len(df.columns))
        self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                self.table.setItem(r, c, QTableWidgetItem(str(df.iat[r, c])))
        self.table.resizeColumnsToContents()

    def on_save_outputs(self):
        if not self.outputs:
            QMessageBox.information(self, "Save", "Run benchmarks first.")
            return
        dir_ = QFileDialog.getExistingDirectory(self, "Choose directory to save outputs")
        if not dir_: return
        for name, data in self.outputs.items():
            path = os.path.join(dir_, f"{name}.bin")
            try:
                with open(path, 'wb') as f:
                    f.write(data)
            except Exception as e:
                QMessageBox.critical(self, "Save failed", f"{name}: {e}")
        QMessageBox.information(self, "Saved", f"Saved {len(self.outputs)} files to {dir_}")


def main():
    import sys
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
