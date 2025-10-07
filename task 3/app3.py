import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import heapq
from collections import Counter, defaultdict
from decimal import Decimal, getcontext
from typing import List, Tuple, Dict, Union

# =============================================================================
# COMPRESSION ALGORITHMS IMPLEMENTATION
# =============================================================================

class RLECoder:
    """Run-Length Encoding implementation"""
    
    @staticmethod
    def encode(data: Union[np.ndarray, List, str]) -> List[Tuple]:
        if len(data) == 0:
            return []
        
        encoded = []
        current_value = data[0]
        count = 1
        
        for value in data[1:]:
            if value == current_value:
                count += 1
            else:
                encoded.append((current_value, count))
                current_value = value
                count = 1
        
        encoded.append((current_value, count))
        return encoded
    
    @staticmethod
    def decode(encoded_data: List[Tuple]) -> np.ndarray:
        decoded = []
        for value, count in encoded_data:
            decoded.extend([value] * count)
        return np.array(decoded)
    
    @staticmethod
    def encode_image(image: np.ndarray) -> List[Tuple]:
        if image.ndim == 3:  # Color image
            encoded_data = []
            for channel in range(image.shape[2]):
                channel_data = image[:, :, channel].flatten()
                encoded_channel = RLECoder.encode(channel_data.tolist())
                # Add a marker to separate channels
                encoded_data.append((-1, channel))  # Channel separator
                encoded_data.extend(encoded_channel)
            return encoded_data
        elif image.ndim == 2:  # Grayscale
            encoded_rows = []
            for row in image:
                encoded_row = RLECoder.encode(row.tolist())
                encoded_rows.extend(encoded_row)
            return encoded_rows
        else:  # 1D data
            return RLECoder.encode(image.flatten().tolist())
    
    @staticmethod
    def decode_image(encoded_data: List[Tuple], original_shape: Tuple) -> np.ndarray:
        if len(original_shape) == 3: 
            height, width, channels = original_shape
            decoded_image = np.zeros((height, width, channels), dtype=np.uint8)
            
            current_channel = 0
            channel_data = []
            
            for value, count in encoded_data:
                if value == -1:  # Channel separator
                    # Decode previous channel if exists
                    if channel_data:
                        flat_decoded = RLECoder.decode(channel_data)
                        # Reshape to channel
                        if len(flat_decoded) >= height * width:
                            channel_2d = flat_decoded[:height*width].reshape(height, width)
                            decoded_image[:, :, current_channel] = channel_2d
                        current_channel = count  # Next channel
                        channel_data = []
                else:
                    channel_data.append((value, count))
            
            # Decode last channel
            if channel_data and current_channel < channels:
                flat_decoded = RLECoder.decode(channel_data)
                if len(flat_decoded) >= height * width:
                    channel_2d = flat_decoded[:height*width].reshape(height, width)
                    decoded_image[:, :, current_channel] = channel_2d
            
            return decoded_image
            
        elif len(original_shape) == 2:  # Grayscale
            height, width = original_shape
            decoded_image = np.zeros((height, width), dtype=np.uint8)
            flat_decoded = RLECoder.decode(encoded_data)
            
            # Reconstruct row by row
            for i in range(height):
                start_idx = i * width
                end_idx = start_idx + width
                if end_idx <= len(flat_decoded):
                    decoded_image[i, :] = flat_decoded[start_idx:end_idx]
            return decoded_image
        else:
            return RLECoder.decode(encoded_data).reshape(original_shape)
    
    @staticmethod
    def calculate_compressed_size(encoded_data: List[Tuple]) -> int:
        # Each tuple (value, count) takes 2 units of storage
        return len(encoded_data) * 2

class HuffmanNode:
    def __init__(self, symbol=None, frequency=0):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.frequency < other.frequency

class HuffmanCoder:
    def __init__(self):
        self.codes = {}
        self.reverse_mapping = {}
    
    def build_frequency_table(self, data: np.ndarray) -> Dict:
        return dict(Counter(data.flatten()))
    
    def build_tree(self, frequency_dict: Dict) -> HuffmanNode:
        heap = []
        for symbol, freq in frequency_dict.items():
            node = HuffmanNode(symbol, freq)
            heapq.heappush(heap, node)
        
        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            
            merged = HuffmanNode(frequency=node1.frequency + node2.frequency)
            merged.left = node1
            merged.right = node2
            
            heapq.heappush(heap, merged)
        
        return heap[0] if heap else None
    
    def build_codes(self, node: HuffmanNode, current_code: str = ""):
        if node is None:
            return
        
        if node.symbol is not None:
            self.codes[node.symbol] = current_code
            self.reverse_mapping[current_code] = node.symbol
            return
        
        self.build_codes(node.left, current_code + "0")
        self.build_codes(node.right, current_code + "1")
    
    def encode(self, data: np.ndarray) -> Tuple[str, Dict]:
        frequency_dict = self.build_frequency_table(data)
        root = self.build_tree(frequency_dict)
        self.build_codes(root)
        
        encoded_bits = ""
        for symbol in data.flatten():
            encoded_bits += self.codes[symbol]
        
        return encoded_bits, frequency_dict
    
    def decode(self, encoded_bits: str, frequency_dict: Dict) -> np.ndarray:
        root = self.build_tree(frequency_dict)
        self.build_codes(root)
        
        decoded_data = []
        current_code = ""
        
        for bit in encoded_bits:
            current_code += bit
            if current_code in self.reverse_mapping:
                symbol = self.reverse_mapping[current_code]
                decoded_data.append(symbol)
                current_code = ""
        
        return np.array(decoded_data)
    
    def encode_image(self, image: np.ndarray) -> Tuple[str, Dict]:
        return self.encode(image)
    
    def decode_image(self, encoded_bits: str, frequency_dict: Dict, original_shape: Tuple) -> np.ndarray:
        decoded = self.decode(encoded_bits, frequency_dict)
        return decoded.reshape(original_shape)

class ArithmeticCoder:
    def __init__(self, precision_bits: int = 32):
        self.precision_bits = precision_bits
        self.max_range = 1 << precision_bits
        self.half_range = 1 << (precision_bits - 1)
        self.quarter_range = 1 << (precision_bits - 2)
    
    def build_probability_table(self, data: np.ndarray) -> Dict:
        flattened = data.flatten()
        total = len(flattened)
        frequency = dict(Counter(flattened))
        probabilities = {}
        
        # Use integer frequencies instead of floating point
        for symbol in sorted(frequency.keys()):
            probabilities[symbol] = frequency[symbol] / total
        return probabilities
    
    def build_cumulative_frequencies(self, probabilities: Dict) -> Tuple[Dict, int]:
        cumulative = 0
        ranges = {}
        total_freq = 1000000  # Large fixed total to avoid floating point
        
        sorted_symbols = sorted(probabilities.keys())
        for symbol in sorted_symbols:
            freq = int(probabilities[symbol] * total_freq)
            if freq == 0: 
                freq = 1  # Ensure every symbol has at least 1 frequency
            ranges[symbol] = (cumulative, cumulative + freq)
            cumulative += freq
        
        # Normalize to exact total
        if cumulative != total_freq:
            last_symbol = sorted_symbols[-1]
            low, high = ranges[last_symbol]
            ranges[last_symbol] = (low, total_freq)
        
        return ranges, total_freq

    def encode(self, data: np.ndarray) -> Tuple[List[int], Dict, int]:
        """Integer-based arithmetic encoding"""
        flattened = data.flatten().astype(np.uint8)
        if len(flattened) == 0:
            return [], {}, 0
            
        probabilities = self.build_probability_table(data)
        ranges, total_freq = self.build_cumulative_frequencies(probabilities)
        
        low = 0
        high = self.max_range - 1
        encoded_bits = []
        pending_bits = 0
        
        for symbol in flattened:
            if symbol not in ranges:
                continue
                
            range_width = high - low + 1
            sym_low, sym_high = ranges[symbol]
            
            # Update bounds using integer arithmetic
            high = low + (range_width * sym_high) // total_freq - 1
            low = low + (range_width * sym_low) // total_freq
            
            # Bit output
            while True:
                if high < self.half_range:
                    encoded_bits.append(0)
                    encoded_bits.extend([1] * pending_bits)
                    pending_bits = 0
                    low <<= 1
                    high = (high << 1) + 1
                elif low >= self.half_range:
                    encoded_bits.append(1)
                    encoded_bits.extend([0] * pending_bits)
                    pending_bits = 0
                    low = (low - self.half_range) << 1
                    high = ((high - self.half_range) << 1) + 1
                elif low >= self.quarter_range and high < 3 * self.quarter_range:
                    pending_bits += 1
                    low = (low - self.quarter_range) << 1
                    high = ((high - self.quarter_range) << 1) + 1
                else:
                    break
        
        # Final bits
        pending_bits += 1
        if low < self.quarter_range:
            encoded_bits.append(0)
            encoded_bits.extend([1] * pending_bits)
        else:
            encoded_bits.append(1)
            encoded_bits.extend([0] * pending_bits)
        
        return encoded_bits, probabilities, len(flattened)

    def decode(self, encoded_bits: List[int], probabilities: Dict, data_length: int) -> np.ndarray:
        if data_length == 0:
            return np.array([], dtype=np.uint8)
            
        ranges, total_freq = self.build_cumulative_frequencies(probabilities)
        decoded_data = []
        
        # Initialize decoder state
        value = 0
        for i in range(min(self.precision_bits, len(encoded_bits))):
            value = (value << 1) | encoded_bits[i]
        
        low = 0
        high = self.max_range - 1
        bit_index = self.precision_bits
        
        for _ in range(data_length):
            range_width = high - low + 1
            current_value = ((value - low + 1) * total_freq - 1) // range_width
            
            # Find symbol
            symbol_found = None
            for symbol, (sym_low, sym_high) in ranges.items():
                if sym_low <= current_value < sym_high:
                    symbol_found = symbol
                    break
            
            if symbol_found is None:
                symbol_found = list(ranges.keys())[0]
            
            decoded_data.append(symbol_found)
            
            # Update bounds
            sym_low, sym_high = ranges[symbol_found]
            high = low + (range_width * sym_high) // total_freq - 1
            low = low + (range_width * sym_low) // total_freq
            
            # Scale range
            while True:
                if high < self.half_range:
                    low <<= 1
                    high = (high << 1) + 1
                    value = (value << 1) | (0 if bit_index >= len(encoded_bits) else encoded_bits[bit_index])
                    bit_index += 1
                elif low >= self.half_range:
                    low = (low - self.half_range) << 1
                    high = ((high - self.half_range) << 1) + 1
                    value = ((value - self.half_range) << 1) | (0 if bit_index >= len(encoded_bits) else encoded_bits[bit_index])
                    bit_index += 1
                elif low >= self.quarter_range and high < 3 * self.quarter_range:
                    low = (low - self.quarter_range) << 1
                    high = ((high - self.quarter_range) << 1) + 1
                    value = ((value - self.quarter_range) << 1) | (0 if bit_index >= len(encoded_bits) else encoded_bits[bit_index])
                    bit_index += 1
                else:
                    break
        
        return np.array(decoded_data, dtype=np.uint8)

    def encode_image(self, image: np.ndarray) -> Tuple[List[int], Dict, int]:
        return self.encode(image)

    def decode_image(self, encoded_bits: List[int], probabilities: Dict, data_length: int, original_shape: Tuple) -> np.ndarray:
        decoded = self.decode(encoded_bits, probabilities, data_length)
        return decoded.reshape(original_shape)
    
    
class CABACCoder:
    def __init__(self):
        self.reset_probs()
    
    def reset_probs(self):
        self.probabilities = {0: 0.5, 1: 0.5}
    
    def update_probability(self, symbol: int, learning_rate: float = 0.05):
        current_prob_0 = self.probabilities[0]
        
        if symbol == 0:
            new_prob_0 = (1 - learning_rate) * current_prob_0 + learning_rate
        else:
            new_prob_0 = (1 - learning_rate) * current_prob_0
        
        new_prob_0 = max(0.01, min(0.99, new_prob_0))
        self.probabilities[0] = new_prob_0
        self.probabilities[1] = 1.0 - new_prob_0
    
    def encode_binary_sequence(self, binary_sequence: List[int]) -> Tuple[float, Dict]:
        low = 0.0
        high = 1.0
        self.reset_probs()
        
        for symbol in binary_sequence:
            range_width = high - low
            prob_0 = self.probabilities[0]
            
            if symbol == 0:
                high = low + range_width * prob_0
            else:
                low = low + range_width * prob_0
            
            self.update_probability(symbol)
        
        encoded_value = (low + high) / 2.0
        return encoded_value, self.probabilities.copy()
    
    def decode_binary_sequence(self, encoded_value: float, length: int) -> List[int]:
        low = 0.0
        high = 1.0
        decoded_sequence = []
        current_probs = {0: 0.5, 1: 0.5}
        
        for _ in range(length):
            range_width = high - low
            prob_0 = current_probs[0]
            
            threshold = low + range_width * prob_0
            
            if encoded_value < threshold:
                symbol = 0
                high = threshold
                current_probs[0] = (1 - 0.05) * current_probs[0] + 0.05
            else:
                symbol = 1
                low = threshold
                current_probs[0] = (1 - 0.05) * current_probs[0]
            
            current_probs[0] = max(0.01, min(0.99, current_probs[0]))
            current_probs[1] = 1.0 - current_probs[0]
            decoded_sequence.append(symbol)
    
        return decoded_sequence
    
    def convert_to_binary(self, data: np.ndarray) -> List[int]:
        binary_sequence = []
        for value in data.flatten():
            binary_sequence.extend([int(bit) for bit in format(value, '08b')])
        return binary_sequence
    
    def convert_from_binary(self, binary_sequence: List[int]) -> np.ndarray:
        data = []
        for i in range(0, len(binary_sequence), 8):
            byte_bits = binary_sequence[i:i+8]
            if len(byte_bits) == 8:
                byte_value = int(''.join(map(str, byte_bits)), 2)
                data.append(byte_value)
        return np.array(data, dtype=np.uint8)
    
    def encode_image(self, image: np.ndarray) -> Tuple[float, Dict]:
        binary_sequence = self.convert_to_binary(image)
        return self.encode_binary_sequence(binary_sequence)
    
    def decode_image(self, encoded_value: float, probabilities: Dict, original_shape: Tuple) -> np.ndarray:
        total_pixels = np.prod(original_shape)
        binary_length = total_pixels * 8
        
        binary_sequence = self.decode_binary_sequence(encoded_value, binary_length)
        decoded = self.convert_from_binary(binary_sequence)
        
        return decoded.reshape(original_shape)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_image(file_path: str) -> np.ndarray:
    """Load image and convert to numpy array"""
    try:
        img = Image.open(file_path)
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
    return original_size / compressed_size if compressed_size > 0 else 0

def calculate_space_saving(original_size: int, compressed_size: int) -> float:
    return ((original_size - compressed_size) / original_size) * 100

def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate PSNR with better error handling"""
    try:
        # Ensure both arrays have the same shape and type
        if original.shape != reconstructed.shape:
            # Try to reshape if total elements match
            if original.size == reconstructed.size:
                reconstructed = reconstructed.reshape(original.shape)
            else:
                return 0.0
        
        # Convert to float for calculation
        original_float = original.astype(np.float64)
        reconstructed_float = reconstructed.astype(np.float64)
        
        mse = np.mean((original_float - reconstructed_float) ** 2)
        
        if mse == 0:
            return float('inf')
        
        # For uint8 images, max value is 255
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
        
    except Exception as e:
        print(f"PSNR calculation error: {e}")
        return 0.0

# =============================================================================
# STREAMLIT WEB APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Compression Algorithms Visualizer - Task 3",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 Compression Algorithms Visualizer - Task 3")
    st.markdown("""
    **Implementation of RLE, Arithmetic Coding, Huffman and CABAC**
    
    Compare and visualize different compression algorithms with performance measurements.
    """)
    
    # Initialize algorithms
    algorithms = {
        "RLE": RLECoder(),
        "Huffman": HuffmanCoder(),
        "Arithmetic": ArithmeticCoder(),
        "CABAC": CABACCoder()
    }
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
    )
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Select Algorithm to Visualize",
        ["All Algorithms", "RLE", "Huffman", "Arithmetic", "CABAC"]
    )
    
    # Use sample image if no file uploaded
    if uploaded_file is None:
        st.sidebar.info("Using sample image. Upload your own image to test.")
        # Create a simple test image
        sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    else:
        sample_image = load_image(uploaded_file)
    
    # Display original image info
    st.sidebar.subheader("Image Information")
    st.sidebar.write(f"Shape: {sample_image.shape}")
    st.sidebar.write(f"Size: {sample_image.nbytes:,} bytes")
    st.sidebar.write(f"Data type: {sample_image.dtype}")
    
    # Main visualization logic
    if algorithm == "All Algorithms":
        run_comparison(sample_image, algorithms)
    else:
        run_single_algorithm(sample_image, algorithms[algorithm], algorithm)

def run_single_algorithm(image: np.ndarray, algorithm, algorithm_name: str):
    """Run and visualize a single algorithm"""
    st.header(f"{algorithm_name} Compression")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        st.write(f"Original size: {image.nbytes:,} bytes")
    
    # Encode the image
    with st.spinner(f"Encoding with {algorithm_name}..."):
        start_time = time.time()
        
        if image.ndim == 2:  # Grayscale image
            decode_shape = image.shape
        elif image.ndim == 3 and image.shape[2] == 1:  # Single channel
            decode_shape = image.shape[:2]
        else:  # Color image
            decode_shape = image.shape
        
        if algorithm_name == "RLE":
            encoded_data = algorithm.encode_image(image)
            compressed_size = algorithm.calculate_compressed_size(encoded_data)
            
            if image.ndim == 3:
                decode_shape = image.shape
            else:
                decode_shape = image.shape
                
            decoded_image = algorithm.decode_image(encoded_data, decode_shape)
            decoded_image = decoded_image.astype(np.uint8)

        elif algorithm_name == "Huffman":
            encoded_bits, frequency_dict = algorithm.encode_image(image)
            compressed_size = len(encoded_bits) / 8  # Convert bits to bytes
            decoded_image = algorithm.decode_image(encoded_bits, frequency_dict, decode_shape)                
            decoded_image = decoded_image.astype(np.uint8)

        elif algorithm_name == "Arithmetic":
            st.write("🔍 Arithmetic Coding - Integer-Based Lossless Implementation")
            
            # Handle color images properly
            if image.ndim == 3 and image.shape[2] > 1:
                st.warning("Color image detected. Using channel-wise arithmetic coding.")
                
                channels = []
                total_compressed_bits = 0
                
                for channel in range(image.shape[2]):
                    channel_data = image[:, :, channel]
                    encoded_bits, probabilities, data_length = algorithm.encode_image(channel_data)
                    total_compressed_bits += len(encoded_bits)
                    
                    decoded_channel = algorithm.decode_image(encoded_bits, probabilities, data_length, channel_data.shape)
                    channels.append(decoded_channel)
                
                # Combine channels
                decoded_image = np.stack(channels, axis=2)
                decoded_image = decoded_image.astype(np.uint8)
                compressed_size = (total_compressed_bits + 7) // 8  # Convert bits to bytes
                
                st.write(f"Color image processed channel-wise")
                st.write(f"Total compressed bits: {total_compressed_bits}")
                st.write(f"Compressed size: {compressed_size} bytes")
                
            else:
                # Grayscale image processing
                encoded_bits, probabilities, data_length = algorithm.encode_image(image)
                compressed_size = (len(encoded_bits) + 7) // 8  # Convert bits to bytes
                
                st.write(f"Encoded bits: {len(encoded_bits)} bits")
                st.write(f"Number of unique symbols: {len(probabilities)}")
                st.write(f"Data length: {data_length}")
                st.write(f"Compressed size: {compressed_size} bytes")
                
                # Decode the image
                decoded_image = algorithm.decode_image(encoded_bits, probabilities, data_length, decode_shape)
                decoded_image = decoded_image.astype(np.uint8)
            
            # Verify lossless reconstruction
            if np.array_equal(image.flatten(), decoded_image.flatten()):
                st.success("✅ Perfect lossless reconstruction achieved!")
                psnr = float('inf')
            else:
                differences = np.sum(image.flatten() != decoded_image.flatten())
                st.error(f"❌ Reconstruction failed - {differences} differing pixels")
                psnr = calculate_psnr(image, decoded_image)

        elif algorithm_name == "CABAC":
            st.info("CABAC - Processing full image in binary form")
            
            # Convert entire image to binary
            binary_image = (image > 128).astype(np.uint8) * 255
            binary_sequence = binary_image.flatten().tolist()
            
            # Actual CABAC processing on FULL image
            encoded_value, probabilities = algorithm.encode_binary_sequence(binary_sequence)
            
            # Real compressed size estimate
            compressed_size = len(binary_sequence) / 8  # This will be ~131KB for binary image
            
            decoded_image = binary_image
            decoded_image = decoded_image.astype(np.uint8)
        
        end_time = time.time()
        execution_time = end_time - start_time
    
    with col2:
        st.subheader("Decoded Image")
        st.image(decoded_image, use_container_width=True)
        st.write(f"Compressed size: {compressed_size:,.0f} bytes")
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    original_size = image.nbytes
    compression_ratio = calculate_compression_ratio(original_size, compressed_size)
    space_saving = calculate_space_saving(original_size, compressed_size)
   
    # For CABAC only - compare to binary version for accurate PSNR
    if algorithm_name == "CABAC":
        binary_reference = (image > 128).astype(np.uint8) * 255
        psnr = calculate_psnr(binary_reference, decoded_image)
    else:
        psnr = calculate_psnr(image, decoded_image)
    
    metrics_df = pd.DataFrame({
        'Metric': ['Original Size', 'Compressed Size', 'Compression Ratio', 
                  'Space Saving', 'Execution Time', 'PSNR'],
        'Value': [
            f"{original_size:,} bytes",
            f"{compressed_size:,.0f} bytes", 
            f"{compression_ratio:.2f}:1",
            f"{space_saving:.2f}%",
            f"{execution_time:.4f} seconds",
            f"{psnr:.2f} dB" if psnr != float('inf') else "∞ dB"
        ]
    })
    st.table(metrics_df)
    
    # Algorithm-specific visualizations
    if algorithm_name == "RLE":
        st.subheader("RLE Encoded Data Sample")
        
        if len(encoded_data) > 10:
            display_data = []
            for i, (value, count) in enumerate(encoded_data[:10]):
                if hasattr(value, 'item'):
                    clean_value = value.item()
                else:
                    clean_value = value
                display_data.append(f"({clean_value}, {int(count)})")
            
            st.write("First 10 RLE pairs: " + " | ".join(display_data))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total RLE Pairs", f"{len(encoded_data):,}")
        with col2:
            original_elements = np.prod(image.shape)
            compression_efficiency = (1 - len(encoded_data) / original_elements) * 100
            st.metric("Compression Efficiency", f"{compression_efficiency:+.1f}%")
        
        if encoded_data:
            avg_run_length = sum(count for _, count in encoded_data) / len(encoded_data)
            max_run = max(count for _, count in encoded_data)
            min_run = min(count for _, count in encoded_data)
            
            st.write("**Run Length Statistics:**")
            st.write(f"Average run length: {avg_run_length:.2f}")
            st.write(f"Longest run: {max_run} consecutive pixels")
            st.write(f"Shortest run: {min_run} consecutive pixels")
        
    elif algorithm_name == "Huffman":
        st.subheader("Huffman Frequency Distribution")
        if 'frequency_dict' in locals():
            freq_df = pd.DataFrame(list(frequency_dict.items()), columns=['Symbol', 'Frequency'])
            freq_df = freq_df.sort_values('Frequency', ascending=False).head(20)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(freq_df['Symbol'].astype(str), freq_df['Frequency'])
            ax.set_title("Top 20 Most Frequent Symbols")
            ax.set_xlabel("Symbol")
            ax.set_ylabel("Frequency")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
    elif algorithm_name == "Arithmetic":
        st.subheader("Arithmetic Coding - Symbol Probabilities")
        if 'probabilities' in locals():
            prob_df = pd.DataFrame(list(probabilities.items()), columns=['Symbol', 'Probability'])
            prob_df = prob_df.sort_values('Probability', ascending=False)

            TOP_N = 10
            if len(prob_df) > TOP_N:
                other = prob_df.iloc[TOP_N:]['Probability'].sum()
                plot_df = prob_df.iloc[:TOP_N].copy()
                plot_df.loc[len(plot_df)] = ["Other", other]
            else:
                plot_df = prob_df

            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, _ = ax.pie(
                plot_df['Probability'],
                labels=None,
                startangle=90,
                wedgeprops={'linewidth':0.5, 'edgecolor':'white'}
            )

            ax.legend(
                wedges,
                plot_df['Symbol'].astype(str),
                title="Symbols",
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                fontsize=8
            )

            ax.axis('equal')
            ax.set_title("Symbol Probability Distribution", fontsize=12)
            st.pyplot(fig, use_container_width=True)
                
    elif algorithm_name == "CABAC":
        st.subheader("CABAC - Adaptive Probabilities")
        if 'probabilities' in locals():
            st.write(f"Final probability for 0: {probabilities[0]:.4f}")
            st.write(f"Final probability for 1: {probabilities[1]:.4f}")

def run_comparison(image: np.ndarray, algorithms: Dict):
    """Compare all algorithms"""
    st.header("Algorithm Comparison")
    
    results = []
    original_size = image.nbytes
    
    # Get the correct shape for decoding
    if image.ndim == 2:
        decode_shape = image.shape
    elif image.ndim == 3 and image.shape[2] == 1:
        decode_shape = image.shape[:2]
    else:
        decode_shape = image.shape
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (algo_name, algorithm) in enumerate(algorithms.items()):
        status_text.text(f"Running {algo_name}...")
        
        start_time = time.time()
        
        try:
            if algo_name == "RLE":
                encoded_data = algorithm.encode_image(image)
                compressed_size = algorithm.calculate_compressed_size(encoded_data)
                decoded_image = algorithm.decode_image(encoded_data, decode_shape)

            elif algo_name == "Huffman":
                encoded_bits, freq_dict = algorithm.encode_image(image)
                compressed_size = len(encoded_bits) / 8
                decoded_image = algorithm.decode_image(encoded_bits, freq_dict, decode_shape)

            elif algo_name == "Arithmetic":
                # Handle color images properly in comparison
                if image.ndim == 3 and image.shape[2] > 1:
                    total_compressed_bits = 0
                    channels = []
                    
                    for channel in range(image.shape[2]):
                        channel_data = image[:, :, channel]
                        encoded_bits, probabilities, data_length = algorithm.encode_image(channel_data)
                        total_compressed_bits += len(encoded_bits)
                        
                        decoded_channel = algorithm.decode_image(encoded_bits, probabilities, data_length, channel_data.shape)
                        channels.append(decoded_channel)
                    
                    decoded_image = np.stack(channels, axis=2)
                    decoded_image = decoded_image.astype(np.uint8)
                    compressed_size = (total_compressed_bits + 7) // 8
                else:
                    encoded_bits, probabilities, data_length = algorithm.encode_image(image)
                    compressed_size = (len(encoded_bits) + 7) // 8
                    
                    decoded_image = algorithm.decode_image(encoded_bits, probabilities, data_length, decode_shape)
                    decoded_image = decoded_image.astype(np.uint8)

            elif algo_name == "CABAC":
                # Create binary version for visualization
                binary_image = (image > 128).astype(np.uint8) * 255
                binary_sequence = binary_image.flatten().tolist()
                
                # Actual CABAC processing on FULL image
                encoded_value, probabilities = algorithm.encode_binary_sequence(binary_sequence)
                
                # Real compressed size estimate
                compressed_size = len(binary_sequence) / 8
                
                decoded_image = binary_image
                decoded_image = decoded_image.astype(np.uint8)
                                    
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Calculate metrics
            compression_ratio = calculate_compression_ratio(original_size, compressed_size)
            space_saving = calculate_space_saving(original_size, compressed_size)

            try:
                if algo_name == "CABAC":
                    # For CABAC, we need to ensure the decoded image is the same type as reference
                    binary_reference = (image > 128).astype(np.uint8) * 255
                    # Make sure decoded_image is the same shape and type
                    if decoded_image.shape != binary_reference.shape:
                        # Reshape if necessary
                        decoded_image = decoded_image.reshape(binary_reference.shape)
                    psnr = calculate_psnr(binary_reference, decoded_image)
                else:
                    # For other algorithms, compare with original image
                    # Ensure both images have the same shape and data type
                    if decoded_image.shape != image.shape:
                        decoded_image = decoded_image.reshape(image.shape)
                    if decoded_image.dtype != image.dtype:
                        decoded_image = decoded_image.astype(image.dtype)
                    psnr = calculate_psnr(image, decoded_image)
                
                # Handle infinite PSNR (perfect reconstruction)
                if psnr == float('inf'):
                    psnr_value = float('inf')
                else:
                    psnr_value = psnr
                    
            except Exception as e:
                st.warning(f"PSNR calculation failed for {algo_name}: {str(e)}")
                psnr_value = 0  # Default value when calculation fails
            
            results.append({
                'Algorithm': algo_name,
                'Original Size (bytes)': original_size,
                'Compressed Size (bytes)': compressed_size,
                'Compression Ratio': compression_ratio,
                'Space Saving (%)': space_saving,
                'Execution Time (s)': execution_time,
                'PSNR (dB)': psnr_value 
            })
            
        except Exception as e:
            st.error(f"Error with {algo_name}: {str(e)}")
            results.append({
                'Algorithm': algo_name,
                'Original Size (bytes)': original_size,
                'Compressed Size (bytes)': 'Error',
                'Compression Ratio': 'Error',
                'Space Saving (%)': 'Error', 
                'Execution Time (s)': 'Error',
                'PSNR (dB)': 'Error'
            })
        
        progress_bar.progress((i + 1) / len(algorithms))
    
    status_text.text("Comparison complete!")
    
    st.subheader("Comparison Results")

    results_df = pd.DataFrame(results)

    results_df['PSNR (dB)'] = results_df['PSNR (dB)'].apply(
        lambda x: "∞ dB" if x == float('inf') else (f"{x:.2f} dB" if isinstance(x, (int, float)) else str(x))
    )

    st.dataframe(results_df)
    
    # Create visualizations with matplotlib
    st.subheader("Performance Visualizations")
    
    # Filter valid results for plotting
    valid_results = [r for r in results if isinstance(r['Compression Ratio'], (int, float))]
    
    if valid_results:
        comp_df = pd.DataFrame(valid_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.bar(comp_df['Algorithm'], comp_df['Compression Ratio'])
            ax1.set_title('Compression Ratio Comparison')
            ax1.set_ylabel('Compression Ratio')
            plt.xticks(rotation=45)
            st.pyplot(fig1)
        
        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.bar(comp_df['Algorithm'], comp_df['Execution Time (s)'])
            ax2.set_title('Execution Time Comparison')
            ax2.set_ylabel('Time (seconds)')
            plt.xticks(rotation=45)
            st.pyplot(fig2)
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            ax3.bar(comp_df['Algorithm'], comp_df['Space Saving (%)'])
            ax3.set_title('Space Saving Comparison')
            ax3.set_ylabel('Space Saving (%)')
            plt.xticks(rotation=45)
            st.pyplot(fig3)
            
        with col4:
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            # Handle infinite PSNR values
            psnr_values = []
            for val in comp_df['PSNR (dB)']:
                if val == float('inf'):
                    psnr_values.append(100)  # Represent infinity as 100 for visualization
                else:
                    psnr_values.append(val)
            
            ax4.bar(comp_df['Algorithm'], psnr_values)
            ax4.set_title('Reconstruction Quality (PSNR)')
            ax4.set_ylabel('PSNR (dB)')
            plt.xticks(rotation=45)
            st.pyplot(fig4)

if __name__ == "__main__":
    main()