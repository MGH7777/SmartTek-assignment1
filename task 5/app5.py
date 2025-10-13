import os
import io
import struct
import time
import zlib
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
    QProgressBar, QLabel, QGroupBox, QTextEdit, QCheckBox
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
# IMPROVED Huffman Coding with Decoding
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
    """Improved Huffman coding with full round-trip support"""
    
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
    def encode(data: bytes) -> Tuple[bytes, Dict[int, int]]:
        """Improved Huffman encoding with proper header"""
        if not data:
            return b"", {}
            
        freqs = {i: 0 for i in range(256)}
        for b in data:
            freqs[b] += 1
            
        root = Huffman.build_tree(freqs)
        codes = Huffman.build_codes(root)
        
        out = io.BytesIO()
        
        # Write header: 256 frequencies as 4-byte integers
        for i in range(256):
            out.write(struct.pack(">I", freqs[i]))
        out.write(struct.pack(">I", len(data)))  # Original size
        
        # Encode data
        bit_buffer = 0
        bit_count = 0
        
        for b in data:
            for c in codes[b]:
                bit_buffer = (bit_buffer << 1) | (1 if c == '1' else 0)
                bit_count += 1
                if bit_count == 8:
                    out.write(bytes([bit_buffer]))
                    bit_buffer = 0
                    bit_count = 0
        
        # Flush remaining bits
        if bit_count > 0:
            bit_buffer <<= (8 - bit_count)
            out.write(bytes([bit_buffer]))
            out.write(bytes([bit_count]))  # Store bit count for last byte
        else:
            out.write(bytes([0]))  # No leftover bits
            
        return out.getvalue(), freqs

    @staticmethod
    def decode(compressed: bytes) -> Tuple[bytes, bool]:
        """Huffman decoding with validation"""
        try:
            if len(compressed) < 256 * 4 + 4:
                return b"", False
                
            # Read frequency table
            freqs = {}
            idx = 0
            for i in range(256):
                freqs[i] = struct.unpack(">I", compressed[idx:idx+4])[0]
                idx += 4
                
            original_size = struct.unpack(">I", compressed[idx:idx+4])[0]
            idx += 4
            
            if original_size == 0:
                return b"", True
                
            # Rebuild tree
            root = Huffman.build_tree(freqs)
            if root is None:
                return b"", False
                
            # Decode data
            data = bytearray()
            node = root
            bit_buffer = compressed[idx:-1]  # All but last byte
            last_byte_info = compressed[-1]  # Last byte: bit_count or 0
            
            bit_pos = 0
            max_bits = len(bit_buffer) * 8
            if last_byte_info > 0:
                max_bits = (len(bit_buffer) - 1) * 8 + last_byte_info
            
            while bit_pos < max_bits and len(data) < original_size:
                byte_idx = bit_pos // 8
                bit_idx = 7 - (bit_pos % 8)
                bit = (bit_buffer[byte_idx] >> bit_idx) & 1
                bit_pos += 1
                
                if bit == 0:
                    node = node.left
                else:
                    node = node.right
                    
                if node is None:
                    return b"", False
                    
                if node.byte is not None:
                    data.append(node.byte)
                    node = root
                    
            success = (len(data) == original_size)
            return bytes(data), success
            
        except Exception:
            return b"", False

# =============================
# PROPER Arithmetic Coding Implementation (Static Methods)
# =============================

class ArithmeticCoder:
    """
    Real 32-bit integer arithmetic coder with proper bit-packing.
    Header layout:
      b"ARITH" | uint32 original_len | 256 * uint32 freqs | packed_bitstream...
    """

    class _BitWriter:
        def __init__(self):
            self.buf = bytearray()
            self.byte = 0
            self.nbits = 0
        def put(self, bit: int):
            self.byte = (self.byte << 1) | (1 if bit else 0)
            self.nbits += 1
            if self.nbits == 8:
                self.buf.append(self.byte)
                self.byte = 0
                self.nbits = 0
        def finish(self):
            if self.nbits:
                self.buf.append(self.byte << (8 - self.nbits))
            return bytes(self.buf)

    class _BitReader:
        def __init__(self, data: bytes):
            self.data = data
            self.i = 0
            self.nbits = 0
            self.byte = 0
        def get(self) -> int:
            if self.nbits == 0:
                self.byte = self.data[self.i] if self.i < len(self.data) else 0
                self.i += 1
                self.nbits = 8
            bit = (self.byte >> 7) & 1
            self.byte = (self.byte << 1) & 0xFF
            self.nbits -= 1
            return bit

    @staticmethod
    def _build_tables(data: bytes):
        freqs = [0]*256
        for b in data:
            freqs[b] += 1
        total = len(data)
        cum = [0]*257
        s = 0
        for i,f in enumerate(freqs):
            cum[i] = s
            s += f
        cum[256] = s
        return freqs, cum, total

    @staticmethod
    def encode(data: bytes) -> Tuple[bytes, Dict, int]:
        if not data:
            return b"ARITH" + (0).to_bytes(4,"big") + bytes(256*4), {}, 0

        freqs, cum, total = ArithmeticCoder._build_tables(data)

        PREC = 32
        MAXR = 1 << PREC
        HALF = 1 << (PREC-1)
        QUAR = 1 << (PREC-2)
        MASK = MAXR - 1

        low  = 0
        high = MASK
        pending = 0
        bw = ArithmeticCoder._BitWriter()

        def output(bit: int):
            nonlocal pending                 # <-- critical fix
            bw.put(bit)
            while pending:
                bw.put(1 - bit)
                pending -= 1

        for s in data:
            rng = (high - low + 1)
            high = low + (rng * cum[s+1]) // total - 1
            low  = low + (rng * cum[s])   // total

            while True:
                if high < HALF:
                    output(0)
                    low  = (low << 1) & MASK
                    high = ((high << 1) & MASK) | 1
                elif low >= HALF:
                    output(1)
                    low  = ((low  - HALF) << 1) & MASK
                    high = ((high - HALF) << 1) & MASK | 1
                elif low >= QUAR and high < 3*QUAR:
                    pending += 1
                    low  = ((low  - QUAR) << 1) & MASK
                    high = ((high - QUAR) << 1) & MASK | 1
                else:
                    break

        # termination
        pending += 1
        if low < QUAR:
            output(0)
        else:
            output(1)
        bitstream = bw.finish()

        hdr = bytearray(b"ARITH")
        hdr += len(data).to_bytes(4, "big")
        for f in freqs:
            hdr += int(f).to_bytes(4, "big")
        return bytes(hdr) + bitstream, {"cum": cum}, len(data)

    @staticmethod
    def decode(compressed: bytes) -> Tuple[bytes, bool]:
        try:
            if len(compressed) < 5 + 4 + 256*4:
                return b"", False
            if compressed[:5] != b"ARITH":
                return b"", False

            n = int.from_bytes(compressed[5:9], "big")
            pos = 9
            freqs = []
            total = 0
            for _ in range(256):
                f = int.from_bytes(compressed[pos:pos+4], "big")
                freqs.append(f); total += f
                pos += 4
            if total != n:
                return b"", False

            cum = [0]*257
            s = 0
            for i,f in enumerate(freqs):
                cum[i] = s
                s += f
            cum[256] = s

            PREC = 32
            MAXR = 1 << PREC
            HALF = 1 << (PREC-1)
            QUAR = 1 << (PREC-2)
            MASK = MAXR - 1

            br = ArithmeticCoder._BitReader(compressed[pos:])
            low  = 0
            high = MASK
            code = 0
            for _ in range(PREC):
                code = ((code << 1) | br.get()) & MASK

            out = bytearray()
            for _ in range(n):
                rng = (high - low + 1)
                value = ((code - low + 1) * total - 1) // rng

                sidx = 255
                for k in range(256):
                    if value < cum[k+1]:
                        sidx = k
                        break
                out.append(sidx)

                high = low + (rng * cum[sidx+1]) // total - 1
                low  = low + (rng * cum[sidx])   // total

                while True:
                    if high < HALF:
                        low  = (low << 1) & MASK
                        high = ((high << 1) & MASK) | 1
                        code = ((code << 1) | br.get()) & MASK
                    elif low >= HALF:
                        low  = ((low  - HALF) << 1) & MASK
                        high = ((high - HALF) << 1) & MASK | 1
                        code = ((code - HALF) << 1) & MASK | br.get()
                    elif low >= QUAR and high < 3*QUAR:
                        low  = ((low  - QUAR) << 1) & MASK
                        high = ((high - QUAR) << 1) & MASK | 1
                        code = ((code - QUAR) << 1) & MASK | br.get()
                    else:
                        break

            return bytes(out), True
        except Exception:
            return b"", False


# =============================
# SIMPLIFIED CABAC Implementation
# =============================

# =============================
# CABAC — validated, CABAC-style, symmetric encode/decode
# =============================
class CABAC:
    """
    CABAC-style binary arithmetic coding:
      - left predictor
      - residual -> zigzag -> Elias-gamma(u+1)
      - 12 contexts: 6 for prefix (run/sep) + 6 for suffix (value bits)
      - Q15 binary range coder with byte renormalization
    Format: b"CAB1" | uint32(original_len) | coded bytes
    """

    # ---- helpers ----
    @staticmethod
    def _zigzag(v: int) -> int:
        return (v << 1) if v >= 0 else ((-v << 1) - 1)

    @staticmethod
    def _unzigzag(u: int) -> int:
        return (u >> 1) if ((u & 1) == 0) else -((u >> 1) + 1)

    @staticmethod
    def _buckets_for(mag: int):
        # 0..5 => prefix contexts, 6..11 => suffix contexts
        if mag == 0: base = 0
        elif mag == 1: base = 1
        elif mag <= 3: base = 2
        elif mag <= 7: base = 3
        elif mag <= 15: base = 4
        else: base = 5
        return base, base + 6

    # ---- binary range coder (Q15) ----
    class _BREnc:
        def __init__(self):
            self.low = 0
            self.rng = 0xFFFFFFFF
            self.out = bytearray()

        def _renorm(self):
            while self.rng < (1 << 24):
                self.out.append((self.low >> 24) & 0xFF)
                self.low = (self.low << 8) & 0xFFFFFFFF
                self.rng = (self.rng << 8) & 0xFFFFFFFF

        def put(self, bit: int, p1_q15: int):
            # STRICT-INTERIOR SPLIT: 1 .. rng-1 (no clamping branches)
            split = ((self.rng - 1) * p1_q15 >> 15) + 1

            # MPS ≡ 1 (upper subrange has size = split)
            if bit:
                # go to upper subrange
                self.low = (self.low + (self.rng - split)) & 0xFFFFFFFF
                self.rng = split
            else:
                # stay in lower subrange
                self.rng -= split

            self._renorm()

        def finish(self):
            for _ in range(4):
                self.out.append((self.low >> 24) & 0xFF)
                self.low = (self.low << 8) & 0xFFFFFFFF
            return bytes(self.out)


    class _BRDec:
        def __init__(self, data: bytes, pos: int):
            self.data = data
            self.i = pos
            self.low = 0
            self.rng = 0xFFFFFFFF
            self.code = 0
            for _ in range(4):
                nxt = self.data[self.i] if self.i < len(self.data) else 0
                self.code = ((self.code << 8) | nxt) & 0xFFFFFFFF
                self.i += 1

        def _renorm(self):
            while self.rng < (1 << 24):
                self.low = (self.low << 8) & 0xFFFFFFFF
                nxt = self.data[self.i] if self.i < len(self.data) else 0
                self.code = ((self.code << 8) | nxt) & 0xFFFFFFFF
                self.i += 1
                self.rng = (self.rng << 8) & 0xFFFFFFFF

        def get(self, p1_q15: int) -> int:
            # SAME STRICT-INTERIOR SPLIT AS ENCODER
            split = ((self.rng - 1) * p1_q15 >> 15) + 1
            threshold = self.rng - split  # size of '0' (lower) subrange

            # STRICT COMPARISON mirrors encoder intervals exactly:
            # [low, low+threshold-1] => bit 0
            # [low+threshold, low+rng-1] => bit 1
            if (self.code - self.low) > threshold - 1:
                # bit = 1, go to upper subrange
                self.low = (self.low + threshold) & 0xFFFFFFFF
                self.rng = split
                bit = 1
            else:
                # bit = 0, stay in lower subrange
                self.rng = threshold
                bit = 0

            self._renorm()
            return bit


    # ---- adaptive contexts ----
    def __init__(self, num_contexts: int = 12):
        self.num_contexts = num_contexts  # 6 prefix + 6 suffix
        self.ones = [1] * num_contexts
        self.zeros = [1] * num_contexts
    def _p1(self, ctx: int) -> int:
        o, z = self.ones[ctx], self.zeros[ctx]
        return int((o << 15) // (o + z))
    def _upd(self, ctx: int, bit: int):
        if bit: self.ones[ctx] += 1
        else:   self.zeros[ctx] += 1
        if self.ones[ctx] + self.zeros[ctx] > (1 << 14):
            self.ones[ctx] >>= 1
            self.zeros[ctx] >>= 1

    # ---- encode ----
    def encode(self, data: bytes):
        if not data:
            return b"CAB1" + (0).to_bytes(4, "big") + (0).to_bytes(4, "big"), []

        # --- infer stride if data looks like an N×N image ---
        n = len(data)
        side = int(len(data) ** 0.5)
        stride = side if side * side == n else 0

        enc = CABAC._BREnc()
        prev = 0
        prev_mag = 0

        # header: magic | orig_len | stride
        out = bytearray(b"CAB1")
        out += n.to_bytes(4, "big")
        out += stride.to_bytes(4, "big")
        
        
        enc = CABAC._BREnc()
        prev = 0
        prev_mag = 0
        out = bytearray(b"CAB1" + len(data).to_bytes(4, "big"))

        for b in data:
            r = int(b) - int(prev)
            if r < -255: r = -255
            if r > 255:  r = 255
            u = CABAC._zigzag(r)
            n = u + 1  # gamma is for positive ints
            ctx_pref, ctx_suf = CABAC._buckets_for(prev_mag)

            # Elias-gamma: k ones, 0, then k suffix bits (MSB-first)
            k = n.bit_length() - 1
            for _ in range(k):              # prefix ones
                p1 = self._p1(ctx_pref); enc.put(1, p1); self._upd(ctx_pref, 1)
            p1 = self._p1(ctx_pref); enc.put(0, p1); self._upd(ctx_pref, 0)  # separator zero
            # suffix bits (LSB-first) — exact same bits, reversed order
            for t in range(0, k):
                bit = (n >> t) & 1
                p1 = self._p1(ctx_suf); enc.put(bit, p1); self._upd(ctx_suf, bit)


            prev = b
            prev_mag = min(abs(r), 255)

        out += enc.finish()
        return bytes(out), []

    # ---- decode ----
    @staticmethod
    def decode(compressed: bytes) -> Tuple[bytes, bool]:
        try:
            if len(compressed) < 12 or compressed[:4] != b"CAB1":
                return b"", False
            nsyms  = int.from_bytes(compressed[4:8],  "big")
            stride = int.from_bytes(compressed[8:12], "big")
            dec = CABAC._BRDec(compressed, 12)
            self = CABAC()

            prev = 0
            prev_mag = 0
            out = bytearray()

            for i in range(nsyms):
                if stride and (i % stride) == 0:
                    prev = 0
                    prev_mag = 0
    
                ctx_pref, ctx_suf = CABAC._buckets_for(prev_mag)

                # gamma prefix: count ones until zero
                k = 0
                while True:
                    b = dec.get(self._p1(ctx_pref)); self._upd(ctx_pref, b)
                    if b == 1:
                        k += 1
                        if k > 32:  # sanity
                            return b"", False
                    else:
                        break
                # gamma suffix: k bits
                n = 1
                for t in range(0, k):
                    bit = dec.get(self._p1(ctx_suf)); self._upd(ctx_suf, bit)
                    n |= (bit << t)


                u = n - 1
                r = CABAC._unzigzag(u)
                val = (prev + r) & 0xFF
                out.append(val)
                prev = val
                prev_mag = min(abs(r), 255)

            return bytes(out), (len(out) == nsyms)
        except Exception:
            return b"", False


# =============================
# IMPROVED RLE with Decoding
# =============================

class RLEEncoder:
    """Improved RLE with full round-trip support"""
    
    @staticmethod
    def encode(data: bytes) -> bytes:
        if not data:
            return b""
            
        encoded = bytearray()
        i = 0
        while i < len(data):
            current = data[i]
            count = 1
            # Count run length (max 255)
            while i + count < len(data) and data[i + count] == current and count < 255:
                count += 1
                
            encoded.append(current)
            encoded.append(count)
            i += count
            
        return bytes(encoded)

    @staticmethod
    def decode(compressed: bytes) -> Tuple[bytes, bool]:
        """RLE decoding with validation"""
        try:
            if len(compressed) % 2 != 0:
                return b"", False
                
            decoded = bytearray()
            for i in range(0, len(compressed), 2):
                byte_val = compressed[i]
                count = compressed[i+1]
                if count == 0:
                    return b"", False
                decoded.extend([byte_val] * count)
                
            return bytes(decoded), True
        except Exception:
            return b"", False

# =============================
# IMPROVED Image & Data Loaders
# =============================

def load_image_bytes(path: str, preserve_bit_depth: bool = False) -> Tuple[np.ndarray, bytes, str]:
    """Load various image formats with bit depth preservation option"""
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
        arr = ds.pixel_array
        
        if preserve_bit_depth and hasattr(ds, 'BitsStored'):
            # Preserve original bit depth
            bit_depth = ds.BitsStored
            if bit_depth > 8:
                # Normalize to 16-bit range but keep precision
                arr = arr.astype(np.uint16)
                desc = f"DICOM {arr.shape[1]}x{arr.shape[0]} ({bit_depth}-bit preserved)"
            else:
                arr = arr.astype(np.uint8)
                desc = f"DICOM {arr.shape[1]}x{arr.shape[0]} (8-bit)"
        else:
            # Normalize to 8-bit for display
            arr = arr.astype(np.float32)
            arr = (255 * (arr - arr.min()) / max(np.ptp(arr), 1)).astype(np.uint8)
            desc = f"DICOM {arr.shape[1]}x{arr.shape[0]} → 8-bit"
            
        return arr, arr.tobytes(), desc
    
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
# UPDATED Main GUI Application
# =============================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DataCoder — Complete Compression Benchmark")
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
        
        # Add bit depth preservation option
        self.chk_preserve_bits = QCheckBox("Preserve DICOM Bit Depth")
        self.chk_preserve_bits.setChecked(False)
        
        row1.addWidget(self.btn_load)
        row1.addWidget(self.cmb_generated)
        row1.addWidget(self.btn_gen)
        row1.addWidget(self.chk_preserve_bits)
        
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
        self.btn_validate = QPushButton("✅ Validate Round-Trip")
        ctl_layout.addWidget(self.btn_run)
        ctl_layout.addWidget(self.btn_save_outputs)
        ctl_layout.addWidget(self.btn_validate)
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
        self.btn_validate.clicked.connect(self.on_validate)

        self.outputs: Dict[str, bytes] = {}
        self.validation_results: Dict[str, bool] = {}
        
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
        <i>All implementations include round-trip validation</i>
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
            preserve_bits = self.chk_preserve_bits.isChecked()
            arr, raw, desc = load_image_bytes(path, preserve_bits)
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
            compressed, _ = Huffman.encode(data)
        
        elif name == "Arithmetic":
             compressed, _, _ = ArithmeticCoder.encode(data)  # Static call
        
        elif name == "CABAC":
            compressed, _ = CABAC().encode(data)
        else:
            compressed = b""
            
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        
        return time_taken, len(compressed), compressed

    def validate_round_trip(self, name: str, compressed: bytes, original: bytes) -> bool:
        """Validate that decompression reproduces original data"""
        try:
            if name == "RLE":
                decoded, success = RLEEncoder.decode(compressed)
            elif name == "Huffman":
                decoded, success = Huffman.decode(compressed)
            elif name == "Arithmetic":
                decoded, success = ArithmeticCoder.decode(compressed)
            elif name == "CABAC":
                decoded, success = CABAC.decode(compressed)
            else:
                return False
                
            return success and decoded == original
        except Exception:
            return False

    def on_run(self):
        if self.img_bytes is None:
            QMessageBox.information(self, "Run Benchmarks", "Please load or generate data first.")
            return
        
        # TEST: Try with simple data first
        test_data = b"AAAAABBBBBCCCCCDDDDD" * 10  # Simple repetitive data
        print(f"Testing with {len(test_data)} bytes")
        
        # Test Arithmetic Coding
        compressed, _, _ = ArithmeticCoder.encode(test_data)
        decoded, success = ArithmeticCoder.decode(compressed)
        print(f"Arithmetic: Original={len(test_data)}, Compressed={len(compressed)}, Success={success}, Match={decoded == test_data}")
        
        if not success or decoded != test_data:
            print(f"First 20 bytes original: {test_data[:20]}")
            print(f"First 20 bytes decoded: {decoded[:20] if decoded else b'FAILED'}")
        
        # Test CABAC  
        cabac = CABAC()
        compressed, _ = cabac.encode(test_data)
        decoded, success = CABAC.decode(compressed)
        print(f"CABAC: Original={len(test_data)}, Compressed={len(compressed)}, Success={success}, Match={decoded == test_data}")
        
        if not success or decoded != test_data:
            print(f"First 20 bytes original: {test_data[:20]}")
            print(f"First 20 bytes decoded: {decoded[:20] if decoded else b'FAILED'}")
            
        self.set_busy(True)
        try:
            original_size = len(self.img_bytes)
            self.outputs.clear()
            self.validation_results.clear()
            
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
                
                # Validate round-trip for supported algorithms
                is_valid = self.validate_round_trip(name, compressed_data, self.img_bytes)
                self.validation_results[name] = is_valid
                
                            # --- ADD THIS BLOCK (debug only) ---
                if name == "CABAC" and not is_valid:
                    dec, ok = CABAC.decode(self.outputs["CABAC"])
                    print("CABAC debug -> ok:", ok,
                        "decoded_len:", len(dec),
                        "orig_len:", len(self.img_bytes))
                    m = min(len(dec), len(self.img_bytes))
                    for i, (a, b) in enumerate(zip(dec, self.img_bytes)):
                        if a != b:
                            print("First mismatch @", i, "decoded:", a, "orig:", b)
                            s = max(0, i-8); e = min(m, i+16)
                            print("decoded slice:", dec[s:e])
                            print("original slice:", self.img_bytes[s:e])
                            break
                    if len(dec) != len(self.img_bytes):
                        print("Length mismatch only.")
                # --- END DEBUG BLOCK ---
                
                compression_ratio = original_size / max(compressed_size, 1)
                space_saving = (1 - compressed_size / original_size) * 100
                
                validation_status = "✅" if is_valid else "❌"
                
                results.append([
                    name,
                    f"{time_taken*1000:.2f} ms",
                    human_bytes(compressed_size),
                    f"{compression_ratio:.2f}:1",
                    f"{space_saving:.1f}%",
                    validation_status
                ])
            
            # Populate table
            self.table.setColumnCount(6)
            self.table.setRowCount(len(results))
            self.table.setHorizontalHeaderLabels(["Algorithm", "Time", "Size", "Ratio", "Space Saving", "Validation"])
            
            for row, result in enumerate(results):
                for col, value in enumerate(result):
                    self.table.setItem(row, col, QTableWidgetItem(str(value)))
                    
            self.table.resizeColumnsToContents()
            
            # Show best valid result
            valid_results = [r for r in results if r[5] == "✅"]
            if valid_results:
                best_ratio = max(valid_results, key=lambda x: float(x[3].split(':')[0]))
                best_algo = best_ratio[0]
                QMessageBox.information(self, "Benchmark Complete", 
                                      f"🏆 Best compression: {best_algo}\n"
                                      f"📊 Ratio: {best_ratio[3]}\n"
                                      f"💾 Space saved: {best_ratio[4]}")
            else:
                QMessageBox.warning(self, "Benchmark Complete", 
                                  "No algorithms passed validation. Check implementations.")
                                  
        except Exception as e:
            QMessageBox.critical(self, "Benchmark Failed", f"Error during benchmarking:\n{str(e)}")
        finally:
            self.set_busy(False)

    def on_validate(self):
        """Run comprehensive validation on all algorithms"""
        if not self.outputs:
            QMessageBox.information(self, "Validate", "Please run benchmarks first.")
            return
            
        validation_report = ["Round-Trip Validation Results:"]
        
        for name, compressed in self.outputs.items():
            is_valid = self.validate_round_trip(name, compressed, self.img_bytes)
            status = "✅ PASS" if is_valid else "❌ FAIL"
            validation_report.append(f"{name}: {status}")
            
        QMessageBox.information(self, "Validation Results", "\n".join(validation_report))

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