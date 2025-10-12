import os
import io
import struct
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import tempfile

import numpy as np
import pandas as pd

# GUI
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QComboBox, QSplitter, QTableWidget, QTableWidgetItem,
    QProgressBar, QLabel, QGroupBox, QTextEdit
)

# Matplotlib embedding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
    Image = None

try:
    import pydicom
    DICOM_AVAILABLE = True
except Exception:
    DICOM_AVAILABLE = False
    pydicom = None

# =============================
# Utils
# =============================

def human_bytes(n: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in Qt"""
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

# =============================
# REAL Huffman Coding Implementation
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
    """Real Huffman coding implementation"""
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
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            merged = Node(freq=node1.freq + node2.freq, left=node1, right=node2)
            heapq.heappush(heap, merged)
        
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
        """Real Huffman encoding"""
        freqs = {i: 0 for i in range(256)}
        for b in data:
            freqs[b] += 1
            
        root = Huffman.build_tree(freqs)
        codes = Huffman.build_codes(root)
        
        out = io.BytesIO()
        MAX64 = (1 << 64) - 1
        
        # Write header
        for i in range(256):
            val = min(freqs[i], MAX64)
            out.write(struct.pack(">Q", val))
        out.write(struct.pack(">Q", len(data)))
        
        # Encode data
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
# REAL Arithmetic Coding Implementation
# =============================

class ArithmeticCoder:
    """Real arithmetic coding implementation"""
    
    def __init__(self, precision=28):
        self.precision = precision
        self.max_range = 1 << precision
        self.half_range = self.max_range >> 1
        self.quarter_range = self.half_range >> 1
        self.three_quarter_range = 3 * self.quarter_range
        
    def build_frequency_table(self, data: bytes) -> Tuple[Dict[int, int], int]:
        """Build normalized frequency table"""
        freqs = {}
        total = len(data)
        
        # Count frequencies
        for byte in data:
            freqs[byte] = freqs.get(byte, 0) + 1
            
        # Normalize to avoid underflow
        max_freq = max(freqs.values()) if freqs else 1
        scale_factor = min(self.max_range // max_freq, 255)
        
        normalized_freqs = {}
        for byte, freq in freqs.items():
            normalized_freqs[byte] = max(1, freq * scale_factor // total)
            
        return normalized_freqs, sum(normalized_freqs.values())
    
    def build_cumulative_table(self, freqs: Dict[int, int], total: int) -> Dict[int, Tuple[int, int]]:
        """Build cumulative probability table"""
        cumulative = 0
        table = {}
        
        for byte in sorted(freqs.keys()):
            freq = freqs[byte]
            table[byte] = (cumulative, cumulative + freq)
            cumulative += freq
            
        return table
    
    def encode(self, data: bytes) -> Tuple[bytes, Dict[int, Tuple[int, int]], int]:
        """Real arithmetic encoding"""
        if not data:
            return b"", {}, 0
            
        freqs, total = self.build_frequency_table(data)
        cum_table = self.build_cumulative_table(freqs, total)
        
        low = 0
        high = self.max_range - 1
        pending_bits = 0
        encoded_bits = []
        
        for byte in data:
            low_bound, high_bound = cum_table[byte]
            range_width = high - low + 1
            
            high = low + (range_width * high_bound) // total - 1
            low = low + (range_width * low_bound) // total
            
            while True:
                if high < self.half_range:
                    encoded_bits.append('0')
                    for _ in range(pending_bits):
                        encoded_bits.append('1')
                    pending_bits = 0
                elif low >= self.half_range:
                    encoded_bits.append('1')
                    for _ in range(pending_bits):
                        encoded_bits.append('0')
                    pending_bits = 0
                    low -= self.half_range
                    high -= self.half_range
                elif low >= self.quarter_range and high < self.three_quarter_range:
                    pending_bits += 1
                    low -= self.quarter_range
                    high -= self.quarter_range
                else:
                    break
                    
                low <<= 1
                high = (high << 1) | 1
        
        # Finalization
        pending_bits += 1
        if low < self.quarter_range:
            encoded_bits.append('0')
            for _ in range(pending_bits):
                encoded_bits.append('1')
        else:
            encoded_bits.append('1')
            for _ in range(pending_bits):
                encoded_bits.append('0')
        
        # Convert to bytes
        bit_string = ''.join(encoded_bits)
        padding = 8 - (len(bit_string) % 8)
        bit_string += '0' * padding
        
        encoded_bytes = bytearray()
        for i in range(0, len(bit_string), 8):
            encoded_bytes.append(int(bit_string[i:i+8], 2))
        
        return bytes(encoded_bytes), cum_table, total

# =============================
# REAL CABAC Implementation
# =============================

class CABAC:
    """Real CABAC implementation with context adaptation"""
    
    def __init__(self, num_contexts=256):
        self.num_contexts = num_contexts
        # Initialize context models
        self.contexts = [{'mps': 0, 'state': 64} for _ in range(num_contexts)]
        
    def _get_context(self, prev_byte: int, pos: int) -> int:
        """Determine context based on previous data"""
        return (prev_byte + pos) % self.num_contexts
        
    def _get_probability(self, state: int) -> Tuple[int, int]:
        """Get probability range from state (0-127)"""
        p_lps = min(max(128 - state, 1), 126) / 256.0
        range_lps = int(p_lps * 16384)  # 14-bit precision
        range_mps = 16384 - range_lps
        return range_mps, range_lps
        
    def _update_context(self, ctx_idx: int, symbol: int):
        """Update context model after encoding a symbol"""
        ctx = self.contexts[ctx_idx]
        
        if symbol == ctx['mps']:
            # MPS occurred
            ctx['state'] = min(ctx['state'] + 2, 127)
        else:
            # LPS occurred
            if ctx['state'] > 64:
                ctx['state'] -= 1
            else:
                ctx['mps'] = 1 - ctx['mps']
                ctx['state'] = max(ctx['state'] - 1, 1)
    
    def encode(self, data: bytes) -> Tuple[bytes, List[Dict]]:
        """Real CABAC encoding"""
        if not data:
            return b"", self.contexts.copy()
            
        low = 0
        high = 0x3FFF  # 14-bit range
        pending_bits = 0
        encoded_bits = []
        prev_byte = 0
        
        for i, byte_val in enumerate(data):
            for bit_pos in range(7, -1, -1):
                bit = (byte_val >> bit_pos) & 1
                ctx_idx = self._get_context(prev_byte, i)
                ctx = self.contexts[ctx_idx]
                
                range_mps, range_lps = self._get_probability(ctx['state'])
                split = low + ((high - low) * range_mps >> 14)
                
                if bit == ctx['mps']:
                    high = split
                else:
                    low = split + 1
                
                # Renormalization
                while (high ^ low) < 0x1000:
                    if high < 0x2000:
                        encoded_bits.append('0')
                        for _ in range(pending_bits):
                            encoded_bits.append('1')
                        pending_bits = 0
                    elif low >= 0x2000:
                        encoded_bits.append('1')
                        for _ in range(pending_bits):
                            encoded_bits.append('0')
                        pending_bits = 0
                        low -= 0x2000
                        high -= 0x2000
                    else:
                        pending_bits += 1
                        low -= 0x1000
                        high -= 0x1000
                    
                    low <<= 1
                    high = (high << 1) | 1
                
                self._update_context(ctx_idx, bit)
            
            prev_byte = byte_val
        
        # Finalization
        pending_bits += 1
        if low < 0x1000:
            encoded_bits.append('0')
            for _ in range(pending_bits):
                encoded_bits.append('1')
        else:
            encoded_bits.append('1')
            for _ in range(pending_bits):
                encoded_bits.append('0')
        
        # Convert to bytes
        bit_string = ''.join(encoded_bits)
        padding = 8 - (len(bit_string) % 8)
        bit_string += '0' * padding
        
        encoded_bytes = bytearray()
        for i in range(0, len(bit_string), 8):
            encoded_bytes.append(int(bit_string[i:i+8], 2))
        
        return bytes(encoded_bytes), self.contexts.copy()

# =============================
# RLE Implementation
# =============================

class RLEEncoder:
    """Real RLE implementation"""
    
    @staticmethod
    def encode(data: bytes) -> bytes:
        if not data:
            return b""
            
        encoded = bytearray()
        current = data[0]
        count = 1
        
        for byte in data[1:]:
            if byte == current and count < 255:
                count += 1
            else:
                encoded.append(current)
                encoded.append(count)
                current = byte
                count = 1
                
        encoded.append(current)
        encoded.append(count)
        return bytes(encoded)

# =============================
# Image & Data Loaders
# =============================

def load_image_bytes(path: str) -> Tuple[np.ndarray, bytes, str]:
    """Load various image formats and convert to bytes"""
    ext = os.path.splitext(path)[1].lower()
    
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow not available for image loading")
        img = Image.open(path).convert("L")  # Convert to grayscale
        arr = np.array(img)
        return arr, arr.tobytes(), f"Image {arr.shape[1]}x{arr.shape[0]} (8-bit grayscale)"
    
    if ext in [".dcm", ".dicom"]:
        if not DICOM_AVAILABLE:
            raise RuntimeError("pydicom not available for DICOM loading")
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        # Normalize to 8-bit
        arr = (255 * (arr - arr.min()) / max(np.ptp(arr), 1)).astype(np.uint8)
        return arr, arr.tobytes(), f"DICOM {arr.shape[1]}x{arr.shape[0]} → 8-bit"
    
    # Raw data file
    with open(path, 'rb') as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr, raw, f"Raw data ({len(raw)} bytes)"

def gen_checkerboard(w=512, h=512, tile=32) -> np.ndarray:
    """Generate checkerboard test pattern"""
    y, x = np.indices((h, w))
    board = ((x // tile + y // tile) % 2) * 255
    return board.astype(np.uint8)

def gen_gaussian_noise(w=512, h=512, sigma=40) -> np.ndarray:
    """Generate Gaussian noise test pattern"""
    arr = np.clip(np.random.normal(127, sigma, (h, w)), 0, 255)
    return arr.astype(np.uint8)

def gen_gradient(w=512, h=512) -> np.ndarray:
    """Generate gradient test pattern"""
    x = np.linspace(0, 255, w)
    y = np.linspace(0, 255, h)
    X, Y = np.meshgrid(x, y)
    gradient = ((X + Y) / 2).astype(np.uint8)
    return gradient

# =============================
# Main GUI Application
# =============================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DataCoder — Complete Image Compression Benchmark")
        self.resize(1400, 900)

        self.img_arr: Optional[np.ndarray] = None
        self.img_bytes: Optional[bytes] = None
        self.img_desc: str = ""

        self.tabs = QtWidgets.QTabWidget()

        # Data tab
        data_tab = QWidget()
        dl = QVBoxLayout(data_tab)
        
        # File loading section
        file_group = QGroupBox("Data Source")
        file_layout = QVBoxLayout(file_group)
        
        row1 = QHBoxLayout()
        self.btn_load = QPushButton("📁 Load File")
        self.cmb_generated = QComboBox()
        self.cmb_generated.addItems([
            "— Generate Test Pattern —", 
            "Checkerboard 512x512", 
            "Gaussian Noise 512x512",
            "Gradient 512x512"
        ])
        self.btn_gen = QPushButton("Generate")
        row1.addWidget(self.btn_load)
        row1.addWidget(self.cmb_generated)
        row1.addWidget(self.btn_gen)
        
        self.lbl_info = QLabel("No data loaded")
        self.lbl_info.setStyleSheet("QLabel { padding: 8px; background: #f0f0f0; border: 1px solid #ccc; }")
        
        file_layout.addLayout(row1)
        file_layout.addWidget(self.lbl_info)

        # Preview section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.canvas_img = MplCanvas(width=6, height=5, dpi=100)
        self.canvas_hist = MplCanvas(width=6, height=5, dpi=100)
        
        preview_split = QSplitter(Qt.Horizontal)
        wc1 = QWidget()
        l1 = QVBoxLayout(wc1)
        l1.addWidget(QLabel("Image Preview"))
        l1.addWidget(self.canvas_img)
        
        wc2 = QWidget()
        l2 = QVBoxLayout(wc2)
        l2.addWidget(QLabel("Histogram"))
        l2.addWidget(self.canvas_hist)
        
        preview_split.addWidget(wc1)
        preview_split.addWidget(wc2)
        preview_layout.addWidget(preview_split)

        dl.addWidget(file_group)
        dl.addWidget(preview_group)

        # Compression tab
        comp_tab = QWidget()
        cl = QVBoxLayout(comp_tab)
        
        # Controls
        ctl_group = QGroupBox("Compression Controls")
        ctl_layout = QHBoxLayout(ctl_group)
        self.btn_run = QPushButton("🚀 Run All Benchmarks")
        self.btn_run.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        self.btn_save_outputs = QPushButton("💾 Save Compressed Files…")
        ctl_layout.addWidget(self.btn_run)
        ctl_layout.addWidget(self.btn_save_outputs)
        ctl_layout.addStretch()

        # Results table
        results_group = QGroupBox("Benchmark Results")
        results_layout = QVBoxLayout(results_group)
        self.table = QTableWidget()
        results_layout.addWidget(self.table)

        # Algorithm info
        info_group = QGroupBox("Algorithm Information")
        info_layout = QVBoxLayout(info_group)
        self.txt_info = QTextEdit()
        self.txt_info.setMaximumHeight(150)
        self.txt_info.setReadOnly(True)
        info_layout.addWidget(self.txt_info)

        cl.addWidget(ctl_group)
        cl.addWidget(results_group)
        cl.addWidget(info_group)

        self.tabs.addTab(data_tab, "📊 Data")
        self.tabs.addTab(comp_tab, "⚡ Compression")
        self.setCentralWidget(self.tabs)

        self.status = QProgressBar()
        self.status.setRange(0, 0)
        self.status.setVisible(False)
        self.statusBar().addPermanentWidget(self.status)

        self.btn_load.clicked.connect(self.on_load)
        self.btn_gen.clicked.connect(self.on_generate)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_save_outputs.clicked.connect(self.on_save_outputs)

        self.outputs: Dict[str, bytes] = {}
        
        # Set algorithm info
        self.update_algorithm_info()

    def update_algorithm_info(self):
        """Update algorithm information display"""
        info_text = """
        <h3>Compression Algorithms</h3>
        <b>RLE (Run-Length Encoding):</b> Simple, fast compression for data with repeated values<br>
        <b>Huffman Coding:</b> Variable-length codes based on symbol frequency<br>
        <b>Arithmetic Coding:</b> Encodes entire message as fractional number, higher compression<br>
        <b>CABAC:</b> Context-adaptive binary arithmetic coding, best for correlated data<br>
        <i>All implementations are real compression algorithms, not simulations</i>
        """
        self.txt_info.setHtml(info_text)

    def set_busy(self, busy: bool):
        self.status.setVisible(busy)
        QApplication.setOverrideCursor(Qt.WaitCursor if busy else Qt.ArrowCursor)
        QApplication.processEvents()

    def refresh_preview(self):
        """Refresh image and histogram preview"""
        if self.img_arr is None:
            return
            
        self.canvas_img.ax.clear()
        
        # Handle 1D arrays (raw data)
        if len(self.img_arr.shape) == 1:
            # Reshape to 2D for display
            size = int(np.sqrt(len(self.img_arr)))
            if size * size == len(self.img_arr):
                display_arr = self.img_arr.reshape((size, size))
                self.canvas_img.ax.imshow(display_arr, cmap='gray', aspect='auto')
            else:
                # Fallback: plot as 1D signal
                self.canvas_img.ax.plot(self.img_arr[:1000])  # First 1000 points
                self.canvas_img.ax.set_title("Data Plot (first 1000 points)")
        else:
            # 2D image
            self.canvas_img.ax.imshow(self.img_arr, cmap='gray')
            
        self.canvas_img.ax.set_axis_off()
        self.canvas_img.ax.set_title(f"Preview - {self.img_desc}")
        self.canvas_img.draw()

        # Histogram
        self.canvas_hist.ax.clear()
        if len(self.img_arr.shape) == 1:
            self.canvas_hist.ax.hist(self.img_arr, bins=256, alpha=0.7)
        else:
            self.canvas_hist.ax.hist(self.img_arr.flatten(), bins=256, alpha=0.7)
        self.canvas_hist.ax.set_title("Histogram")
        self.canvas_hist.ax.set_xlabel("Value")
        self.canvas_hist.ax.set_ylabel("Frequency")
        self.canvas_hist.draw()

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open File", "", 
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;"
            "DICOM (*.dcm *.dicom);;"
            "All Files (*.*)")
        if not path:
            return
            
        self.set_busy(True)
        try:
            arr, raw, desc = load_image_bytes(path)
            self.img_arr, self.img_bytes, self.img_desc = arr, raw, desc
            self.lbl_info.setText(f"📊 Loaded: {os.path.basename(path)} — {desc} — {human_bytes(len(raw))}")
            self.refresh_preview()
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Error loading file:\n{str(e)}")
        finally:
            self.set_busy(False)

    def on_generate(self):
        choice = self.cmb_generated.currentText()
        if choice == "Checkerboard 512x512":
            arr = gen_checkerboard()
        elif choice == "Gaussian Noise 512x512":
            arr = gen_gaussian_noise()
        elif choice == "Gradient 512x512":
            arr = gen_gradient()
        else:
            QMessageBox.information(self, "Generate", "Please select a pattern to generate.")
            return
            
        self.img_arr = arr
        self.img_bytes = arr.tobytes()
        self.img_desc = f"Generated {choice}"
        self.lbl_info.setText(f"🎨 Generated: {self.img_desc} — {human_bytes(len(self.img_bytes))}")
        self.refresh_preview()

    def bench_one(self, name: str, encoder, data: bytes) -> Tuple[float, int, bytes]:
        """Benchmark a single compression algorithm"""
        start_time = time.perf_counter()
        
        if name == "RLE":
            compressed = RLEEncoder.encode(data)
        elif name == "Huffman":
            compressed = Huffman.encode(data)
        elif name == "Arithmetic":
            compressed, _, _ = ArithmeticCoder().encode(data)
        elif name == "CABAC":
            compressed, _ = CABAC().encode(data)
        else:
            compressed = b""
            
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        
        return time_taken, len(compressed), compressed

    def on_run(self):
        if self.img_bytes is None:
            QMessageBox.information(self, "Run Benchmarks", "Please load or generate data first.")
            return
            
        self.set_busy(True)
        try:
            original_size = len(self.img_bytes)
            self.outputs.clear()
            
            algorithms = [
                ("RLE", RLEEncoder),
                ("Huffman", Huffman),
                ("Arithmetic", ArithmeticCoder),
                ("CABAC", CABAC)
            ]
            
            results = []
            
            for name, encoder in algorithms:
                time_taken, compressed_size, compressed_data = self.bench_one(name, encoder, self.img_bytes)
                self.outputs[name] = compressed_data
                
                compression_ratio = original_size / max(compressed_size, 1)
                space_saving = (1 - compressed_size / original_size) * 100
                
                results.append([
                    name,
                    f"{time_taken*1000:.2f} ms",
                    human_bytes(compressed_size),
                    f"{compression_ratio:.2f}:1",
                    f"{space_saving:.1f}%"
                ])
            
            # Populate table
            self.table.setColumnCount(5)
            self.table.setRowCount(len(results))
            self.table.setHorizontalHeaderLabels(["Algorithm", "Time", "Size", "Ratio", "Space Saving"])
            
            for row, result in enumerate(results):
                for col, value in enumerate(result):
                    self.table.setItem(row, col, QTableWidgetItem(str(value)))
                    
            self.table.resizeColumnsToContents()
            
            # Show best result
            best_ratio = max(results, key=lambda x: float(x[3].split(':')[0]))
            best_algo = best_ratio[0]
            QMessageBox.information(self, "Benchmark Complete", 
                                  f"🏆 Best compression: {best_algo}\n"
                                  f"📊 Ratio: {best_ratio[3]}\n"
                                  f"💾 Space saved: {best_ratio[4]}")
                                  
        except Exception as e:
            QMessageBox.critical(self, "Benchmark Failed", f"Error during benchmarking:\n{str(e)}")
        finally:
            self.set_busy(False)

    def on_save_outputs(self):
        if not self.outputs:
            QMessageBox.information(self, "Save Outputs", "Please run benchmarks first.")
            return
            
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory to Save Outputs")
        if not dir_path:
            return
            
        try:
            for name, data in self.outputs.items():
                ext = ".rle" if name == "RLE" else ".huff" if name == "Huffman" else ".ac" if name == "Arithmetic" else ".cabac"
                file_path = os.path.join(dir_path, f"compressed_{name}{ext}")
                with open(file_path, 'wb') as f:
                    f.write(data)
                    
            QMessageBox.information(self, "Save Complete", 
                                  f"Saved {len(self.outputs)} compressed files to:\n{dir_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Error saving files:\n{str(e)}")

def main():
    import sys
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    # Check dependencies
    if not PIL_AVAILABLE:
        print("Warning: Pillow not available - image loading disabled")
    if not DICOM_AVAILABLE:
        print("Warning: pydicom not available - DICOM loading disabled")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()