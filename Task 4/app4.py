import io
import math
import heapq
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time, tracemalloc

import pandas as pd
import streamlit as st
import numpy as np

# ---- Assignment benchmark requirements (tweak as needed) ----
BENCHMARK_REQUIREMENTS = {
    "speed": {       # seconds (<=)
        "RLE": 2.0,
        "Huffman": 5.0,
        "Arithmetic": 10.0,
        "CABAC": 15.0,
    },
    "memory": {      # MB (<=)
        "RLE": 50,
        "Huffman": 100,
        "Arithmetic": 200,
        "CABAC": 300,
    },
}

def check_benchmark_requirements(algo_name: str, time_taken_s: float, peak_bytes: int):
    """Return (speed_ok, memory_ok, overall_ok)."""
    speed_ok  = time_taken_s <= BENCHMARK_REQUIREMENTS["speed"][algo_name]
    mem_mb    = peak_bytes / (1024*1024)
    memory_ok = mem_mb <= BENCHMARK_REQUIREMENTS["memory"][algo_name]
    return speed_ok, memory_ok, (speed_ok and memory_ok)


def accurate_byte_len(obj: Any) -> int:
    """Calculate accurate byte length of objects"""
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    elif isinstance(obj, dict):
        size = 0
        for k, v in obj.items():
            size += len(str(k)) + len(str(v)) + 8
        return max(size, len(str(obj)))
    else:
        return len(str(obj).encode('utf-8'))

def to_bytes(data: Any) -> bytes:
    """Convert pandas series/list to bytes for compression"""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, pd.Series):
        s = "\n".join(map(lambda x: "" if pd.isna(x) else str(x), data.tolist()))
        return s.encode("utf-8")
    return str(data).encode("utf-8")

def calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of the data"""
    if not data:
        return 0.0
    
    counter = Counter(data)
    total = len(data)
    entropy = 0.0
    
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    
    return entropy

def benchmark(fn, data, *args):
    """Benchmark function with memory and timing"""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(data, *args) if args else fn(data)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    mem_used = peak if peak > 0 else len(data) * 3 + 1024 * 1024
    return result, (t1 - t0), int(mem_used)

def generate_large_test_data():
    """Generate synthetic large datasets for testing different algorithm strengths"""
    np.random.seed(42)  # For reproducible results
    
    # 1. Text-like data with skewed distribution (good for Huffman/Arithmetic)
    common_words = [b'ERROR', b'SUCCESS', b'PENDING', b'ACTIVE', b'INACTIVE']
    rare_words = [b'CRITICAL', b'URGENT', b'IMMEDIATE', b'SHUTDOWN']
    
    text_data = []
    for _ in range(20000):  # 20K elements
        if np.random.random() < 0.8:  # 80% common words
            text_data.append(np.random.choice(common_words))
        else:  # 20% rare words
            text_data.append(np.random.choice(rare_words))
    
    # 2. Data with long runs (good for RLE)
    runs_data = bytearray()
    for _ in range(10000):
        run_length = np.random.randint(5, 100)  # Longer runs
        byte_val = np.random.randint(0, 256)
        runs_data.extend([byte_val] * run_length)
    
    # 3. Highly correlated data (good for CABAC)
    correlated = bytearray()
    current = 128
    for _ in range(50000):  # Larger dataset
        # Strong correlation with previous values
        current = max(0, min(255, current + np.random.randint(-5, 6)))
        correlated.append(current)
    
    return b''.join(text_data), bytes(runs_data), bytes(correlated)

# =============================
# RLE (Run-Length Encoding) - REAL IMPLEMENTATION
# =============================
def rle_encode(data: bytes) -> bytes:
    """Real RLE implementation - replaces sequences with (value, count) pairs"""
    if not data:
        return b""
    
    encoded = bytearray()
    current_byte = data[0]
    count = 1
    
    for byte in data[1:]:
        if byte == current_byte and count < 255:
            count += 1
        else:
            encoded.append(current_byte)
            encoded.append(count)
            current_byte = byte
            count = 1
    
    encoded.append(current_byte)
    encoded.append(count)
    return bytes(encoded)

def rle_decode(data: bytes) -> bytes:
    """Real RLE decoding"""
    if len(data) % 2 != 0:
        raise ValueError("Invalid RLE data")
    
    decoded = bytearray()
    for i in range(0, len(data), 2):
        byte_val = data[i]
        count = data[i + 1]
        decoded.extend([byte_val] * count)
    
    return bytes(decoded)

# =============================
# Huffman Coding - REAL IMPLEMENTATION
# =============================
@dataclass(order=True)
class HuffNode:
    freq: int
    symbol: Optional[int] = field(default=None, compare=False)
    left: Optional["HuffNode"] = field(default=None, compare=False)
    right: Optional["HuffNode"] = field(default=None, compare=False)

def huffman_build_tree(freqs: Dict[int, int]) -> Optional["HuffNode"]:
    # Sort by (freq, symbol) so ties are resolved identically on both sides
    items = sorted(freqs.items(), key=lambda x: (x[1], x[0]))
    heap = [HuffNode(freq=f, symbol=s) for s, f in items]
    if not heap:
        return None

    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, parent)
    return heap[0]


def huffman_build_codes(root: Optional[HuffNode]) -> Dict[int, str]:
    """Generate Huffman codes from tree"""
    codes: Dict[int, str] = {}
    
    def traverse(node: HuffNode, code: str):
        if node.symbol is not None:
            codes[node.symbol] = code
        else:
            if node.left:
                traverse(node.left, code + "0")
            if node.right:
                traverse(node.right, code + "1")
    
    if root:
        traverse(root, "")
    return codes

def huffman_encode(data: bytes) -> Tuple[bytes, Dict[int, str]]:
    """Real Huffman encoding"""
    if not data:
        return b"", {}
    
    # Build frequency table
    freqs = Counter(data)
    
    # Build tree and codes
    root = huffman_build_tree(freqs)
    codes = huffman_build_codes(root)
    
    # Encode data
    encoded_bits = "".join(codes[byte] for byte in data)
    
    # Convert bits to bytes
    padding = 8 - (len(encoded_bits) % 8)
    encoded_bits += "0" * padding
    
    encoded_bytes = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte_val = int(encoded_bits[i:i+8], 2)
        encoded_bytes.append(byte_val)
    
    # Create header with frequencies
    header = bytearray()
    for i in range(256):
        freq = freqs.get(i, 0)
        header.extend(freq.to_bytes(4, 'big'))
    
    return bytes(header) + bytes(encoded_bytes), codes

def huffman_decode(encoded_data: bytes) -> bytes:
    """Real Huffman decoding"""
    if len(encoded_data) < 1024:  # 256 * 4 bytes for header
        return b""
    
    # Read frequency table from header
    freqs = {}
    for i in range(256):
        start = i * 4
        freq = int.from_bytes(encoded_data[start:start+4], 'big')
        if freq > 0:
            freqs[i] = freq
    
    # Rebuild tree
    root = huffman_build_tree(freqs)
    
    # Convert encoded bytes to bit string
    encoded_bits = ""
    for byte in encoded_data[1024:]:
        encoded_bits += f"{byte:08b}"
    
    # Decode using tree
    decoded = bytearray()
    current_node = root
    total_symbols = sum(freqs.values())   # total original symbol count
    symbols_out = 0

    for bit in encoded_bits:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.symbol is not None:
            decoded.append(current_node.symbol)
            symbols_out += 1
            if symbols_out == total_symbols:
                break  # stop once all original symbols are decoded
            current_node = root

    return bytes(decoded)


# =============================
# Arithmetic Coding - REAL IMPLEMENTATION
# =============================
class ArithmeticCoder:
    """Real arithmetic coding implementation"""
    
    def __init__(self, precision=32):
        self.precision = precision
        self.one = 1 << precision
        self.half = self.one >> 1
        self.quarter = self.half >> 1
        
    def build_probability_table(self, data: bytes) -> Tuple[Dict[int, Tuple[int, int]], int]:
        """Build cumulative probability table"""
        freqs = Counter(data)
        total = len(data)
        
        # Normalize frequencies to avoid underflow
        cum_prob = {}
        current_low = 0
        
        for symbol, freq in freqs.items():
            prob_range = (freq * self.one) // total
            cum_prob[symbol] = (current_low, current_low + prob_range)
            current_low += prob_range
        
        return cum_prob, total
    
    def encode(self, data: bytes) -> Tuple[bytes, Dict[int, Tuple[int, int]], int]:
        """Real arithmetic encoding"""
        if not data:
            return b"", {}, 0
            
        cum_prob, total = self.build_probability_table(data)
        
        low = 0
        high = self.one
        underflow_bits = 0
        encoded_bits = []
        
        for symbol in data:
            range_width = high - low
            symbol_low, symbol_high = cum_prob[symbol]
            
            high = low + (range_width * symbol_high) // self.one
            low = low + (range_width * symbol_low) // self.one
            
            while True:
                if high < self.half:
                    encoded_bits.append('0')
                    encoded_bits.extend(['1'] * underflow_bits)
                    underflow_bits = 0
                elif low >= self.half:
                    encoded_bits.append('1')
                    encoded_bits.extend(['0'] * underflow_bits)
                    underflow_bits = 0
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < 3 * self.quarter:
                    underflow_bits += 1
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break
                    
                low <<= 1
                high <<= 1
                high = min(high, self.one)
        
        # Finalization
        underflow_bits += 1
        if low < self.quarter:
            encoded_bits.append('0')
            encoded_bits.extend(['1'] * underflow_bits)
        else:
            encoded_bits.append('1')
            encoded_bits.extend(['0'] * underflow_bits)
        
        # Convert bits to bytes
        bit_string = ''.join(encoded_bits)
        padding = 8 - (len(bit_string) % 8)
        bit_string += '0' * padding
        
        encoded_bytes = bytearray()
        for i in range(0, len(bit_string), 8):
            encoded_bytes.append(int(bit_string[i:i+8], 2))
        
        return bytes(encoded_bytes), cum_prob, total

def arithmetic_encode(data: bytes) -> Tuple[bytes, Dict[int, Tuple[int, int]], int, int]:
    """Wrapper for arithmetic encoding"""
    coder = ArithmeticCoder()
    encoded, table, total = coder.encode(data)
    total_size = len(encoded) + accurate_byte_len(table) + 8  # +8 for total length
    return encoded, table, total, total_size

def arithmetic_decode(encoded_data: bytes, table: Dict[int, Tuple[int, int]], total: int, data_length: int) -> bytes:
    """Real arithmetic decoding"""
    if not encoded_data or not table:
        return b""
    
    coder = ArithmeticCoder()
    
    # Convert encoded bytes to bits
    bit_string = ''.join(f"{byte:08b}" for byte in encoded_data)
    
    # Initialize decoder state
    low, high = 0, coder.one
    value = 0
    
    # Read initial value bits
    for i in range(coder.precision):
        if i < len(bit_string):
            value = (value << 1) | int(bit_string[i])
    
    decoded = bytearray()
    bit_index = coder.precision
    
    for _ in range(data_length):
        range_width = high - low
        offset = ((value - low) * coder.one) // range_width
        
        # Find symbol
        found_symbol = None
        for symbol, (sym_low, sym_high) in table.items():
            if sym_low <= offset < sym_high:
                found_symbol = symbol
                break
        
        if found_symbol is None:
            break
            
        decoded.append(found_symbol)
        
        # Update ranges
        sym_low, sym_high = table[found_symbol]
        high = low + (range_width * sym_high) // coder.one
        low = low + (range_width * sym_low) // coder.one
        
        # Renormalize
        while True:
            if high < coder.half:
                # Do nothing
                pass
            elif low >= coder.half:
                low -= coder.half
                high -= coder.half
                value -= coder.half
            elif low >= coder.quarter and high < 3 * coder.quarter:
                low -= coder.quarter
                high -= coder.quarter
                value -= coder.quarter
            else:
                break
                
            low <<= 1
            high <<= 1
            value <<= 1
            if bit_index < len(bit_string):
                value |= int(bit_string[bit_index])
                bit_index += 1
            high = min(high, coder.one)
    
    return bytes(decoded)

# =============================
# CABAC - REAL IMPLEMENTATION
# =============================
class CABAC:
    """Real CABAC implementation with context modeling"""
    
    def __init__(self, num_contexts=64):
        self.num_contexts = num_contexts
        # Initialize context models with uniform distribution
        self.contexts = [{'mps': 0, 'state': 64} for _ in range(num_contexts)]  # state 64 = p(0) = 0.5
        
    def _get_context(self, data: bytes, position: int) -> int:
        """Simple context modeling based on previous bytes"""
        if position < 2:
            return 0
        # Use previous 2 bytes to determine context
        ctx = (data[position-1] ^ data[position-2]) % self.num_contexts
        return ctx
    
    def _get_probability(self, state: int) -> Tuple[int, int]:
        """Get probability range from state"""
        # State represents probability of LPS (Least Probable Symbol)
        p_lps = min(max(state, 1), 126) / 128.0
        p_mps = 1.0 - p_lps
        range_lps = int(p_lps * 4096)
        range_mps = 4096 - range_lps
        return range_mps, range_lps
    
    def _update_context(self, context_idx: int, symbol: int):
        """Update context model after encoding/decoding a symbol"""
        ctx = self.contexts[context_idx]
        
        if symbol == ctx['mps']:
            # MPS occurred - increase probability of MPS
            if ctx['state'] > 64:
                ctx['state'] -= 1
            else:
                ctx['state'] = max(ctx['state'] - 1, 1)
        else:
            # LPS occurred - decrease probability of MPS
            if ctx['state'] < 64:
                ctx['state'] += 1
            else:
                # Switch MPS
                ctx['mps'] = 1 - ctx['mps']
                ctx['state'] = min(ctx['state'] + 1, 126)
    
    def encode(self, data: bytes) -> Tuple[bytes, List[Dict]]:
        """CABAC encoding with symmetric integer range math (+1 width and -1 split)."""
        if not data:
            return b"", self.contexts.copy()

        low = 0
        high = 0xFFFF
        pending_bits = 0
        encoded_bits = []

        for i, byte_val in enumerate(data):
            for bit_pos in range(7, -1, -1):
                bit = (byte_val >> bit_pos) & 1

                # SAME CONTEXT RULE AS BEFORE: depends on already-known bytes in 'data'
                ctx_idx = self._get_context(data, i)
                ctx = self.contexts[ctx_idx]

                range_mps, range_lps = self._get_probability(ctx['state'])

                range_width = (high - low + 1)
                split = low + ((range_mps * range_width) >> 12) - 1

                if bit == ctx['mps']:
                    high = split
                else:
                    low = split + 1

                # Renormalization
                while (high ^ low) < 0x1000:
                    if high < 0x8000:
                        encoded_bits.append('0')
                        for _ in range(pending_bits):
                            encoded_bits.append('1')
                        pending_bits = 0
                    elif low >= 0x8000:
                        encoded_bits.append('1')
                        for _ in range(pending_bits):
                            encoded_bits.append('0')
                        pending_bits = 0
                        low -= 0x8000
                        high -= 0x8000
                    else:
                        pending_bits += 1
                        low  -= 0x4000
                        high -= 0x4000

                    low <<= 1
                    high = (high << 1) | 1

                # Update context after coding this bit
                self._update_context(ctx_idx, bit)

        # Finalization
        pending_bits += 1
        if low < 0x4000:
            encoded_bits.append('0')
            for _ in range(pending_bits):
                encoded_bits.append('1')
        else:
            encoded_bits.append('1')
            for _ in range(pending_bits):
                encoded_bits.append('0')

        # Bits -> bytes
        bit_string = ''.join(encoded_bits)
        padding = (8 - (len(bit_string) % 8)) % 8
        bit_string += '0' * padding

        encoded_bytes = bytearray()
        for i in range(0, len(bit_string), 8):
            encoded_bytes.append(int(bit_string[i:i+8], 2))

        return bytes(encoded_bytes), self.contexts.copy()


def cabac_encode(data: bytes) -> Tuple[bytes, List[Dict], int]:
    """Wrapper for CABAC encoding - IMPROVED"""
    cabac = CABAC()
    encoded, contexts = cabac.encode(data)
    # Only count the actual encoded bytes, not the context overhead
    total_size = len(encoded)
    return encoded, contexts, total_size


def cabac_decode(encoded_data: bytes, contexts: List[Dict], data_length: int) -> bytes:
    """CABAC decoding — mirrors encoder exactly: fresh contexts + same range math + same context rule."""
    if not encoded_data:
        return b""

    # Fresh, uniform contexts. If you want to honor the encoder's num_contexts, use len(contexts) if provided.
    num_ctx = len(contexts) if contexts else 64
    cabac = CABAC(num_contexts=num_ctx)   # DO NOT load encoder's final contexts

    # Bits
    bit_string = ''.join(f"{byte:08b}" for byte in encoded_data)
    bit_index = 0

    low = 0
    high = 0xFFFF
    code = 0

    # Initial code (16 bits)
    for _ in range(16):
        if bit_index < len(bit_string):
            code = (code << 1) | int(bit_string[bit_index])
            bit_index += 1

    decoded_bytes = bytearray()
    current_byte = 0
    bit_count = 0

    for byte_idx in range(data_length):
        for bit_pos in range(8):
            # SAME context rule as encoder: context depends on previously decoded BYTES
            ctx_idx = cabac._get_context(decoded_bytes, byte_idx)
            ctx = cabac.contexts[ctx_idx]

            range_mps, range_lps = cabac._get_probability(ctx['state'])

            # >>> EXACT SAME RANGE MATH AS ENCODER <<<
            range_width = (high - low + 1)
            split = low + ((range_mps * range_width) >> 12) - 1

            if code <= split:
                bit = ctx['mps']
                high = split
            else:
                bit = 1 - ctx['mps']
                low = split + 1

            # Emit bit into current output byte
            current_byte = (current_byte << 1) | bit
            bit_count += 1
            if bit_count == 8:
                decoded_bytes.append(current_byte)
                current_byte = 0
                bit_count = 0

            # Renormalization (mirror encoder)
            while (high ^ low) < 0x1000:
                if high < 0x8000:
                    pass
                elif low >= 0x8000:
                    code -= 0x8000
                    low  -= 0x8000
                    high -= 0x8000
                else:
                    code -= 0x4000
                    low  -= 0x4000
                    high -= 0x4000

                low <<= 1
                high = (high << 1) | 1
                code <<= 1
                if bit_index < len(bit_string):
                    code |= int(bit_string[bit_index])
                    bit_index += 1

            # Update context with the decoded bit (same as encoder)
            cabac._update_context(ctx_idx, bit)

    return bytes(decoded_bytes)



# =============================
# Streamlit UI - UPDATED WITH INDIVIDUAL ALGORITHM SELECTION
# =============================
def main():
    st.set_page_config(page_title="Task 4 — Lossless Coding Benchmark", layout="wide")
    st.title("🎯 Task 4 — Lossless Coding Benchmark")
    st.markdown("**Choose and compare: RLE • Huffman • Arithmetic Coding • CABAC**")
    
    # Algorithm explanations
    with st.expander("📚 Algorithm Explanations", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RLE (Run-Length Encoding)")
            st.markdown("""
            - **How it works**: Replaces sequences of repeated bytes with (value, count) pairs
            - **Best for**: Data with long runs of identical values (images with solid colors)
            - **Compression**: Fast but limited effectiveness
            - **Example**: `AAAABBBCC` → `A4B3C2`
            """)
            
            st.subheader("Huffman Coding")
            st.markdown("""
            - **How it works**: Creates variable-length codes based on symbol frequency
            - **Best for**: Text and general data with skewed frequency distributions
            - **Compression**: Good general-purpose compression
            - **Key feature**: Prefix-free codes guarantee unique decoding
            """)
        
        with col2:
            st.subheader("Arithmetic Coding")
            st.markdown("""
            - **How it works**: Encodes entire message as a single fractional number
            - **Best for**: High compression ratio needed, works well on all data types
            - **Compression**: Typically better than Huffman
            - **Key feature**: Can approach entropy limit more closely
            """)
            
            st.subheader("CABAC (Context-Adaptive Binary Arithmetic Coding)")
            st.markdown("""
            - **How it works**: Combines arithmetic coding with adaptive context modeling
            - **Best for**: Video compression (H.264/265), highly correlated data
            - **Compression**: Excellent for sequential/correlated data
            - **Key feature**: Adapts probability models based on context
            """)

    # File upload
    st.sidebar.header("1) Upload Excel File")
    uploaded = st.sidebar.file_uploader("Excel file (.xlsx)", type=["xlsx"]) 
    selected_col = None
    series = None

    if uploaded is not None:
        try:
            df = pd.read_excel(uploaded)
            st.sidebar.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            if not df.empty:
                selected_col = st.sidebar.selectbox("Select column to compress", df.columns.tolist())
                series = df[selected_col]
                
                st.subheader("📊 Data Preview")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.dataframe(series.to_frame().head(10), use_container_width=True)
                with col_b:
                    st.metric("Total Values", len(series))
                    st.metric("Unique Values", series.nunique())
                    st.metric("Data Type", str(series.dtype))
        except Exception as e:
            st.sidebar.error(f"Failed to read Excel: {e}")

    # Algorithm Selection
    st.sidebar.header("2) Choose Algorithms")
    st.sidebar.markdown("Select which compression algorithms to test:")
    
    algorithms = {
        "RLE": st.sidebar.checkbox("RLE Encoding", value=True),
        "Huffman": st.sidebar.checkbox("Huffman Coding", value=True),
        "Arithmetic": st.sidebar.checkbox("Arithmetic Coding", value=True),
        "CABAC": st.sidebar.checkbox("CABAC", value=True)
    }
    
    # Get selected algorithms
    selected_algorithms = [algo for algo, selected in algorithms.items() if selected]
    
    # Options
        # Options
    st.sidebar.header("3) Benchmark Options")
    force_text = st.sidebar.checkbox("Treat as text data", value=True)
    
    # Large Dataset Testing
    st.sidebar.header("4) Large Dataset Testing")
    run_large_test = st.sidebar.button("🧪 Run Large Synthetic Test", type="secondary")
    
    # Only show run button if algorithms are selected
    if selected_algorithms:
        run_bench = st.sidebar.button("🚀 Run Selected Benchmarks", type="primary")
    else:
        st.sidebar.warning("Please select at least one algorithm")
        run_bench = False

    # Main content
        # Main content
    colA, colB = st.columns(2)

    if run_bench:
        if series is None:
            st.error("Please upload an Excel file and select a column.")
        else:
            raw_bytes = to_bytes(series if force_text else series.astype("uint8", errors="ignore"))
            st.write(f"**Original data size:** {len(raw_bytes):,} bytes")
            
            if len(raw_bytes) == 0:
                st.warning("Selected data is empty after conversion.")
            else:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_algos = len(selected_algorithms)
                
                for i, algo in enumerate(selected_algorithms):
                    status_text.text(f"Running {algo} compression... ({i+1}/{total_algos})")
                    
                    if algo == "RLE":
                        # RLE Benchmark
                        enc_rle, t_rle, m_rle = benchmark(rle_encode, raw_bytes)
                        dec_rle = rle_decode(enc_rle)
                        rle_valid = dec_rle == raw_bytes
                        results.append(("RLE", len(enc_rle), t_rle, m_rle, rle_valid))
                        
                    elif algo == "Huffman":
                        try:
                            # Use consistent benchmarking for encode + memory
                            enc_result, t_huff, m_huff = benchmark(huffman_encode, raw_bytes)
                            enc_huff_data, codes = enc_result
                            
                            # Measure decode time separately but consistently
                            t_decode_start = time.perf_counter()
                            dec_huff = huffman_decode(enc_huff_data)
                            t_decode = time.perf_counter() - t_decode_start
                            
                            # Total time = encode + decode
                            total_time = t_huff + t_decode
                            
                            huff_valid = dec_huff == raw_bytes
                            huff_size = len(enc_huff_data)
                            
                            results.append(("Huffman", huff_size, total_time, m_huff, huff_valid))
                            
                        except Exception as e:
                            st.error(f"Huffman failed: {e}")
                            results.append(("Huffman", 0, 0, 0, False))
                        
                    elif algo == "Arithmetic":
                        enc_result, t_arith, m_arith = benchmark(arithmetic_encode, raw_bytes)
                        enc_arith, table, total, ac_size = enc_result
                        dec_arith = arithmetic_decode(enc_arith, table, total, len(raw_bytes))
                        arith_valid = dec_arith == raw_bytes
                        results.append(("Arithmetic", ac_size, t_arith, m_arith, arith_valid))
                        
                    elif algo == "CABAC":
                        enc_result, t_cabac, m_cabac = benchmark(cabac_encode, raw_bytes)
                        enc_cabac, contexts, cabac_size = enc_result
                        dec_cabac = cabac_decode(enc_cabac, contexts, len(raw_bytes))
                        cabac_valid = dec_cabac == raw_bytes
                        results.append(("CABAC", cabac_size, t_cabac, m_cabac, cabac_valid))
                    
                    progress_bar.progress((i + 1) / total_algos)
                
                status_text.text("Benchmark complete!")
                
                # Create results table
                report_rows = []
                overall_flags = []

                for name, size, t, mem, valid in results:
                    ratio  = len(raw_bytes) / max(1, size)
                    mem_mb = max(mem / (1024 * 1024), 0.000001)

                    # Benchmark pass/fail for this algorithm
                    speed_ok, memory_ok, overall_ok = check_benchmark_requirements(name, t, mem)
                    overall_flags.append(overall_ok)

                    report_rows.append({
                        "Algorithm": name,
                        "Compressed Size": f"{size:,} bytes",
                        "Compression Ratio": f"{ratio:.3f}:1",
                        "Time": f"{t:.6f}s",
                        "Memory": f"{mem_mb:.2f} MB",
                        "Valid": "✅" if valid else "❌",
                        "Speed Benchmark": "✅ PASS" if speed_ok  else "❌ FAIL",
                        "Memory Benchmark": "✅ PASS" if memory_ok else "❌ FAIL",
                        "Meets All Benchmarks": "✅" if overall_ok else "❌",
                    })

                
                rep_df = pd.DataFrame(report_rows)
                
                with colA:
                    st.subheader("📊 Benchmark Results")
                    st.dataframe(rep_df, use_container_width=True)
                    
                    # Show best algorithm
                    if results:
                        best_algo = max(results, key=lambda x: len(raw_bytes) / max(1, x[1]))
                        st.success(f"🏆 **Best compression**: {best_algo[0]} with {len(raw_bytes)/max(1, best_algo[1]):.2f}:1 ratio")
                    
                    # Data characteristics
                    st.subheader("📈 Data Characteristics")
                    entropy = calculate_entropy(raw_bytes)
                    st.metric("Data Entropy", f"{entropy:.3f} bits/byte")
                    st.metric("Theoretical Max Compression", f"{8/entropy:.2f}:1" if entropy > 0 else "N/A")
                
                with colB:
                    st.subheader("📈 Compression Ratio Comparison")
                    
                    # Prepare chart data
                    chart_data = {
                        "Algorithm": [r[0] for r in results],
                        "Compression Ratio": [len(raw_bytes) / max(1, r[1]) for r in results]
                    }
                    chart_df = pd.DataFrame(chart_data)
                    
                    if not chart_df.empty:
                        st.bar_chart(chart_df.set_index("Algorithm"))
                    
                    # Show efficiency analysis
                    st.subheader("⚡ Efficiency Analysis")
                    for name, size, t, mem, valid in results:
                        ratio = len(raw_bytes) / max(1, size)
                        efficiency = ratio / t if t > 0 else 0
                        st.write(f"**{name}**: {efficiency:.1f} ratio/sec")

    # LARGE SYNTHETIC TEST SECTION
    if run_large_test:
        st.header("🧪 Large Synthetic Dataset Test")
        
        with st.spinner("Generating large test datasets..."):
            text_data, runs_data, correlated_data = generate_large_test_data()
            
            datasets = [
                ("Text-like Data (Skewed Distribution)", text_data, "Huffman/Arithmetic"),
                ("Run-length Data (Long Repeated Sequences)", runs_data, "RLE"), 
                ("Correlated Data (Markovian Bytes)", correlated_data, "CABAC")
            ]
            
            for dataset_name, test_data, best_for in datasets:
                st.subheader(f"📊 {dataset_name}")
                st.write(f"**Size:** {len(test_data):,} bytes | **Best for:** {best_for}")
                
                # Run benchmarks on this dataset
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_algos = len(selected_algorithms)
                
                for i, algo in enumerate(selected_algorithms):
                    status_text.text(f"Testing {algo} on {dataset_name}... ({i+1}/{total_algos})")
                    
                    if algo == "RLE":
                        # RLE Benchmark
                        enc_rle, t_rle, m_rle = benchmark(rle_encode, test_data)
                        dec_rle = rle_decode(enc_rle)
                        rle_valid = dec_rle == test_data
                        results.append(("RLE", len(enc_rle), t_rle, m_rle, rle_valid))
                        
                    elif algo == "Huffman":
                        try:
                            # Use consistent benchmarking for encode + memory
                            enc_result, t_huff, m_huff = benchmark(huffman_encode, test_data)
                            enc_huff_data, codes = enc_result
                            
                            # Measure decode time separately but consistently
                            t_decode_start = time.perf_counter()
                            dec_huff = huffman_decode(enc_huff_data)
                            t_decode = time.perf_counter() - t_decode_start
                            
                            # Total time = encode + decode
                            total_time = t_huff + t_decode
                            
                            huff_valid = dec_huff == test_data
                            huff_size = len(enc_huff_data)
                            
                            results.append(("Huffman", huff_size, total_time, m_huff, huff_valid))
                            
                        except Exception as e:
                            st.error(f"Huffman failed: {e}")
                            results.append(("Huffman", 0, 0, 0, False))
                        
                    elif algo == "Arithmetic":
                        enc_result, t_arith, m_arith = benchmark(arithmetic_encode, test_data)
                        enc_arith, table, total, ac_size = enc_result
                        dec_arith = arithmetic_decode(enc_arith, table, total, len(test_data))
                        arith_valid = dec_arith == test_data
                        results.append(("Arithmetic", ac_size, t_arith, m_arith, arith_valid))
                        
                    elif algo == "CABAC":
                        enc_result, t_cabac, m_cabac = benchmark(cabac_encode, test_data)
                        enc_cabac, contexts, cabac_size = enc_result
                        dec_cabac = cabac_decode(enc_cabac, contexts, len(test_data))
                        cabac_valid = dec_cabac == test_data
                        results.append(("CABAC", cabac_size, t_cabac, m_cabac, cabac_valid))
                    
                    progress_bar.progress((i + 1) / total_algos)
                
                status_text.text(f"Complete for {dataset_name}!")
                
                # Display results for this dataset
                if results:
                    report_rows = []
                    for name, size, t, mem, valid in results:
                        ratio = len(test_data) / max(1, size)
                        mem_mb = max(mem / (1024 * 1024), 0.000001)
                        
                        report_rows.append({
                            "Algorithm": name,
                            "Compressed Size": f"{size:,} bytes",
                            "Compression Ratio": f"{ratio:.3f}:1", 
                            "Time": f"{t:.6f}s",
                            "Memory": f"{mem_mb:.2f} MB",
                            "Valid": "✅" if valid else "❌"
                        })
                    
                    st.dataframe(pd.DataFrame(report_rows), use_container_width=True)
                    
                    # Show best algorithm for this dataset
                    best_algo = max(results, key=lambda x: len(test_data) / max(1, x[1]))
                    st.success(f"🏆 **Best for {dataset_name}**: {best_algo[0]} with {len(test_data)/max(1, best_algo[1]):.2f}:1 ratio")
                    
                st.markdown("---")

    # Individual Algorithm Testing Section
    st.markdown("---")
    st.header("🔍 Individual Algorithm Testing")
    
    if series is not None and not run_bench and not run_large_test:
        raw_bytes = to_bytes(series if force_text else series.astype("uint8", errors="ignore"))
        
        test_col1, test_col2, test_col3, test_col4 = st.columns(4)
        
        with test_col1:
            if st.button("Test RLE Only", type="secondary"):
                with st.spinner("Testing RLE..."):
                    enc_rle, t_rle, m_rle = benchmark(rle_encode, raw_bytes)
                    dec_rle = rle_decode(enc_rle)
                    rle_valid = dec_rle == raw_bytes
                    
                    st.success(f"RLE Results:")
                    st.write(f"Compressed: {len(enc_rle):,} bytes")
                    st.write(f"Ratio: {len(raw_bytes)/max(1, len(enc_rle)):.2f}:1")
                    st.write(f"Time: {t_rle:.4f}s")
                    st.write(f"Valid: {'✅' if rle_valid else '❌'}")
        
        with test_col2:
            if st.button("Test Huffman Only", type="secondary"):
                with st.spinner("Testing Huffman..."):
                    enc_huff, codes = huffman_encode(raw_bytes)
                    t_huff = time.perf_counter()
                    dec_huff = huffman_decode(enc_huff)
                    t_huff = time.perf_counter() - t_huff
                    huff_valid = dec_huff == raw_bytes
                    
                    st.success(f"Huffman Results:")
                    st.write(f"Compressed: {len(enc_huff):,} bytes")
                    st.write(f"Ratio: {len(raw_bytes)/max(1, len(enc_huff)):.2f}:1")
                    st.write(f"Time: {t_huff:.4f}s")
                    st.write(f"Valid: {'✅' if huff_valid else '❌'}")
        
        with test_col3:
            if st.button("Test Arithmetic Only", type="secondary"):
                with st.spinner("Testing Arithmetic..."):
                    enc_arith, table, total, ac_size = arithmetic_encode(raw_bytes)
                    t_arith = time.perf_counter()
                    dec_arith = arithmetic_decode(enc_arith, table, total, len(raw_bytes))
                    t_arith = time.perf_counter() - t_arith
                    arith_valid = dec_arith == raw_bytes
                    
                    st.success(f"Arithmetic Results:")
                    st.write(f"Compressed: {ac_size:,} bytes")
                    st.write(f"Ratio: {len(raw_bytes)/max(1, ac_size):.2f}:1")
                    st.write(f"Time: {t_arith:.4f}s")
                    st.write(f"Valid: {'✅' if arith_valid else '❌'}")
        
        with test_col4:
            if st.button("Test CABAC Only", type="secondary"):
                with st.spinner("Testing CABAC..."):
                    enc_cabac, contexts, cabac_size = cabac_encode(raw_bytes)
                    t_cabac = time.perf_counter()
                    dec_cabac = cabac_decode(enc_cabac, contexts, len(raw_bytes))
                    t_cabac = time.perf_counter() - t_cabac
                    cabac_valid = dec_cabac == raw_bytes
                    
                    st.success(f"CABAC Results:")
                    st.write(f"Compressed: {cabac_size:,} bytes")
                    st.write(f"Ratio: {len(raw_bytes)/max(1, cabac_size):.2f}:1")
                    st.write(f"Time: {t_cabac:.4f}s")
                    st.write(f"Valid: {'✅' if cabac_valid else '❌'}")

    # Performance tips
    with st.expander("💡 Performance Tips"):
        st.markdown("""
        - **RLE**: Best for data with repeated values (like sparse matrices)
        - **Huffman**: Good general-purpose compression for mixed data
        - **Arithmetic**: Higher compression but slower, good for text
        - **CABAC**: Best for sequential/correlated data (time series, signals)
        *Note: Real compression ratios* """)
        
        
if __name__ == "__main__":
    main()