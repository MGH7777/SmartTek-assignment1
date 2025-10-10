import heapq
import os
import time
from collections import Counter
from PIL import Image
import numpy as np

class HuffmanNode:
    def __init__(self, symbol=None, frequency=0):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.frequency < other.frequency

class HuffmanCoding:
    def __init__(self):
        self.codebook = {}
        self.reverse_codebook = {}
        self.tree = None
    
    def build_frequency_table(self, data):
        """Build frequency table from data bytes"""
        return Counter(data)
    
    def build_huffman_tree(self, frequency):
        """Build Huffman tree from frequency table"""
        if not frequency:
            return None
            
        priority_queue = []
        for symbol, freq in frequency.items():
            node = HuffmanNode(symbol, freq)
            heapq.heappush(priority_queue, node)
        
        while len(priority_queue) > 1:
            node1 = heapq.heappop(priority_queue)
            node2 = heapq.heappop(priority_queue)
            
            merged = HuffmanNode(frequency=node1.frequency + node2.frequency)
            merged.left = node1
            merged.right = node2
            
            heapq.heappush(priority_queue, merged)
        
        self.tree = priority_queue[0] if priority_queue else None
        return self.tree
    
    def generate_codebook(self, node=None, current_code=""):
        """Generate Huffman codebook from tree"""
        if node is None:
            node = self.tree
            self.codebook = {}
            self.reverse_codebook = {}
        
        if node.symbol is not None:
            self.codebook[node.symbol] = current_code
            self.reverse_codebook[current_code] = node.symbol
        else:
            if node.left:
                self.generate_codebook(node.left, current_code + "0")
            if node.right:
                self.generate_codebook(node.right, current_code + "1")
        
        return self.codebook
    
    def encode_data(self, data):
        """Encode data using Huffman codes"""
        if not self.codebook:
            self.generate_codebook()
        
        encoded_bits = []
        for byte in data:
            encoded_bits.append(self.codebook[byte])
        
        encoded_bitstring = ''.join(encoded_bits)
        return encoded_bitstring
    
    def decode_data(self, encoded_bitstring):
        """Decode Huffman-encoded bitstring back to original data"""
        decoded_bytes = []
        current_code = ""
        
        for bit in encoded_bitstring:
            current_code += bit
            if current_code in self.reverse_codebook:
                decoded_bytes.append(self.reverse_codebook[current_code])
                current_code = ""
        
        return bytes(decoded_bytes)

class WorkingCABAC:
    """Simplified but reliable CABAC implementation"""
    
    def __init__(self):
        self.prob_one = 0.5  
    
    def encode_binary_data(self, bitstring):
        """Simple reliable CABAC encoder"""
        if not bitstring:
            return b""
            
        bits = [int(bit) for bit in bitstring]
        encoded_bits = []
        
        low = 0
        high = 0xFFFFFF  # 24-bit range for simplicity
        pending_bits = 0
        
        for bit in bits:
            # Calculate range split
            range_width = high - low + 1
            split = low + int(range_width * self.prob_one)
            
            if bit == 1:
                low = split
            else:
                high = split - 1
            
            # Renormalize when range gets too small
            while (high - low) < 0x400000:
                if high < 0x800000:
                    # Output 0
                    encoded_bits.append('0')
                    for _ in range(pending_bits):
                        encoded_bits.append('1')
                    pending_bits = 0
                elif low >= 0x800000:
                    # Output 1  
                    encoded_bits.append('1')
                    for _ in range(pending_bits):
                        encoded_bits.append('0')
                    pending_bits = 0
                    low -= 0x800000
                    high -= 0x800000
                else:
                    # Underflow condition
                    pending_bits += 1
                    low -= 0x400000
                    high -= 0x400000
                
                low = (low << 1) & 0xFFFFFF
                high = ((high << 1) | 1) & 0xFFFFFF
        
        # Finalization
        pending_bits += 1
        if low < 0x400000:
            encoded_bits.append('0')
            for _ in range(pending_bits):
                encoded_bits.append('1')
        else:
            encoded_bits.append('1')
            for _ in range(pending_bits):
                encoded_bits.append('0')
        
        # Convert to bytes
        return self.bits_to_bytes(encoded_bits)
    
    def decode_binary_data(self, encoded_data, original_bit_length):
        """Simple reliable CABAC decoder"""
        if len(encoded_data) == 0 or original_bit_length == 0:
            return "0" * original_bit_length
            
        # Convert bytes to bits
        encoded_bits = self.bytes_to_bits(encoded_data)
        bit_stream = self.BitStream(encoded_bits)
        
        low = 0
        high = 0xFFFFFF
        code = 0
        
        # Read initial code (24 bits)
        for _ in range(24):
            code = (code << 1) | bit_stream.next_bit()
        
        decoded_bits = []
        pending_bits = 0
        
        for _ in range(original_bit_length):
            # Calculate range split
            range_width = high - low + 1
            split = low + int(range_width * self.prob_one)
            
            if code >= split:
                bit = 1
                low = split
            else:
                bit = 0
                high = split - 1
            
            decoded_bits.append(str(bit))
            
            # Renormalize (same as encoder)
            while (high - low) < 0x400000:
                if high < 0x800000:
                    # Do nothing for decoding
                    pass
                elif low >= 0x800000:
                    low -= 0x800000
                    high -= 0x800000
                    code -= 0x800000
                else:
                    # Underflow condition
                    pending_bits += 1
                    low -= 0x400000
                    high -= 0x400000
                    code -= 0x400000
                
                low = (low << 1) & 0xFFFFFF
                high = ((high << 1) | 1) & 0xFFFFFF
                code = (code << 1) | bit_stream.next_bit()
        
        return ''.join(decoded_bits)
    
    class BitStream:
        """Helper class for reading bits"""
        def __init__(self, bits):
            self.bits = bits
            self.index = 0
        
        def next_bit(self):
            if self.index < len(self.bits):
                bit = int(self.bits[self.index])
                self.index += 1
                return bit
            return 0
    
    def bits_to_bytes(self, bits):
        """Convert list of bits to bytes"""
        bytes_list = []
        current_byte = 0
        bit_count = 0
        
        for bit in bits:
            current_byte = (current_byte << 1) | int(bit)
            bit_count += 1
            
            if bit_count == 8:
                bytes_list.append(current_byte)
                current_byte = 0
                bit_count = 0
        
        # Add final byte if incomplete
        if bit_count > 0:
            current_byte <<= (8 - bit_count)
            bytes_list.append(current_byte)
        
        return bytes(bytes_list)
    
    def bytes_to_bits(self, data):
        """Convert bytes to list of bits"""
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append(str((byte >> i) & 1))
        return bits

class RLEEncoder:
    """Run-Length Encoding implementation"""
    
    @staticmethod
    def encode(data):
        """Encode data using Run-Length Encoding"""
        if not data:
            return b""
        
        encoded = []
        count = 1
        current = data[0]
        
        for byte in data[1:]:
            if byte == current and count < 255:
                count += 1
            else:
                encoded.extend([count, current])
                count = 1
                current = byte
        
        encoded.extend([count, current])
        return bytes(encoded)
    
    @staticmethod
    def decode(encoded_data):
        """Decode RLE-encoded data"""
        if len(encoded_data) % 2 != 0:
            raise ValueError("Invalid RLE data")
        
        decoded = []
        for i in range(0, len(encoded_data), 2):
            count = encoded_data[i]
            byte = encoded_data[i + 1]
            decoded.extend([byte] * count)
        
        return bytes(decoded)

class MultiLosslessCodingPipeline:
    """Complete pipeline integrating Huffman, CABAC, and RLE"""
    
    def __init__(self):
        self.huffman = HuffmanCoding()
        self.cabac = WorkingCABAC()
        self.rle = RLEEncoder()
    
    def load_image_data(self, image_path):
        """Load and prepare image data for compression"""
        try:
            print(f"Loading image: {image_path}")
            with Image.open(image_path) as img:
                print(f"Original image mode: {img.mode}, size: {img.size}")
                
                # Convert to grayscale for simpler analysis
                if img.mode != 'L':
                    img = img.convert('L')
                    print("Converted image to grayscale (L mode)")
                
                # Convert image to numpy array
                img_array = np.array(img)
                data = img_array.tobytes()
                
                print(f"✓ Successfully loaded image: {image_path}")
                print(f"✓ Image dimensions: {img_array.shape}")
                print(f"✓ Total pixels: {img_array.size}")
                print(f"✓ Total bytes: {len(data)}")
                
                return data, img_array
                
        except Exception as e:
            print(f"✗ Error loading image: {e}")
            return None, None
    
    def load_file_data(self, file_path):
        """Load any file data"""
        try:
            with open(file_path, 'rb') as file:
                data = file.read()
            print(f"✓ Loaded {len(data)} bytes from {file_path}")
            return data
        except Exception as e:
            print(f"✗ Error reading file: {e}")
            return None
    
    def huffman_compression(self, data):
        """Perform Huffman compression with timing"""
        start_time = time.time()
        
        # Build frequency table and Huffman tree
        frequency = self.huffman.build_frequency_table(data)
        self.huffman.build_huffman_tree(frequency)
        self.huffman.generate_codebook()
        
        # Encode data
        encoded_bits = self.huffman.encode_data(data)
        
        huffman_time = time.time() - start_time
        
        # Calculate compression stats
        original_bits = len(data) * 8
        encoded_bits_len = len(encoded_bits)
        compression_ratio = original_bits / encoded_bits_len if encoded_bits_len > 0 else 0
        space_saving = (1 - encoded_bits_len / original_bits) * 100 if original_bits > 0 else 0
        
        stats = {
            'original_size': len(data),
            'encoded_size': (encoded_bits_len + 7) // 8,  # Convert to bytes
            'compression_ratio': compression_ratio,
            'space_saving': space_saving,
            'time': huffman_time
        }
        
        return encoded_bits, stats
    
    def cabac_compression(self, bitstring):
        """Perform CABAC compression on bitstring with timing"""
        start_time = time.time()
        
        encoded_data = self.cabac.encode_binary_data(bitstring)
        
        cabac_time = time.time() - start_time
        
        # Calculate compression stats
        original_bits = len(bitstring)
        encoded_bits = len(encoded_data) * 8
        compression_ratio = original_bits / encoded_bits if encoded_bits > 0 else 0
        space_saving = (1 - encoded_bits / original_bits) * 100 if original_bits > 0 else 0
        
        stats = {
            'original_size': (original_bits + 7) // 8,
            'encoded_size': len(encoded_data),
            'compression_ratio': compression_ratio,
            'space_saving': space_saving,
            'time': cabac_time
        }
        
        return encoded_data, stats
    
    def rle_compression(self, data):
        """Perform RLE compression with timing"""
        start_time = time.time()
        
        encoded_data = self.rle.encode(data)
        
        rle_time = time.time() - start_time
        
        # Calculate compression stats
        compression_ratio = len(data) / len(encoded_data) if len(encoded_data) > 0 else 0
        space_saving = (1 - len(encoded_data) / len(data)) * 100 if len(data) > 0 else 0
        
        stats = {
            'original_size': len(data),
            'encoded_size': len(encoded_data),
            'compression_ratio': compression_ratio,
            'space_saving': space_saving,
            'time': rle_time
        }
        
        return encoded_data, stats
    
    def huffman_cabac_pipeline(self, data):
        """Complete Huffman → CABAC pipeline"""
        print("\n" + "="*60)
        print("HUFFMAN → CABAC PIPELINE")
        print("="*60)
        
        # Step 1: Huffman Compression
        print("\n1. HUFFMAN COMPRESSION")
        huffman_encoded, huffman_stats = self.huffman_compression(data)
        
        print(f"   Original: {huffman_stats['original_size']:>8} bytes")
        print(f"   Encoded:  {huffman_stats['encoded_size']:>8} bytes")
        print(f"   Ratio:    {huffman_stats['compression_ratio']:>8.2f}:1")
        print(f"   Saving:   {huffman_stats['space_saving']:>7.2f}%")
        print(f"   Time:     {huffman_stats['time']:>8.4f}s")
        
        # Step 2: CABAC Compression
        print("\n2. CABAC COMPRESSION")
        cabac_encoded, cabac_stats = self.cabac_compression(huffman_encoded)
        
        print(f"   Original: {cabac_stats['original_size']:>8} bytes")
        print(f"   Encoded:  {cabac_stats['encoded_size']:>8} bytes")
        print(f"   Ratio:    {cabac_stats['compression_ratio']:>8.2f}:1")
        print(f"   Saving:   {cabac_stats['space_saving']:>7.2f}%")
        print(f"   Time:     {cabac_stats['time']:>8.4f}s")
        
        # Overall pipeline stats
        overall_ratio = (len(data) * 8) / (len(cabac_encoded) * 8)
        overall_saving = (1 - len(cabac_encoded) / len(data)) * 100
        
        print(f"\n3. OVERALL PIPELINE RESULTS")
        print(f"   Original: {len(data):>8} bytes")
        print(f"   Final:    {len(cabac_encoded):>8} bytes")
        print(f"   Ratio:    {overall_ratio:>8.2f}:1")
        print(f"   Saving:   {overall_saving:>7.2f}%")
        print(f"   Total Time: {huffman_stats['time'] + cabac_stats['time']:.4f}s")
        
        return {
            'huffman': huffman_stats,
            'cabac': cabac_stats,
            'overall': {
                'ratio': overall_ratio,
                'saving': overall_saving,
                'time': huffman_stats['time'] + cabac_stats['time']
            }
        }
    
    def compare_all_methods(self, data):
        """Compare Huffman, CABAC, RLE, and combined pipeline"""
        print("\n" + "="*70)
        print("COMPREHENSIVE COMPRESSION METHOD COMPARISON")
        print("="*70)
        
        results = {}
        
        # RLE Compression
        print("\n1. RUN-LENGTH ENCODING (RLE)")
        rle_encoded, rle_stats = self.rle_compression(data)
        results['rle'] = rle_stats
        self._print_stats(rle_stats)
        
        # Huffman Compression
        print("\n2. HUFFMAN CODING")
        huffman_encoded, huffman_stats = self.huffman_compression(data)
        results['huffman'] = huffman_stats
        self._print_stats(huffman_stats)
        
        # CABAC on raw data (convert to bits first)
        print("\n3. CABAC ON RAW DATA")
        raw_bits = ''.join(format(byte, '08b') for byte in data)
        cabac_raw_encoded, cabac_raw_stats = self.cabac_compression(raw_bits)
        results['cabac_raw'] = cabac_raw_stats
        self._print_stats(cabac_raw_stats)
        
        # Huffman → CABAC Pipeline
        pipeline_stats = self.huffman_cabac_pipeline(data)
        results['pipeline'] = pipeline_stats['overall']
        
        # Print comparison summary
        self._print_comparison_summary(results)
        
        return results
    
    def _print_stats(self, stats):
        """Print compression statistics in consistent format"""
        print(f"   Original: {stats['original_size']:>8} bytes")
        print(f"   Encoded:  {stats['encoded_size']:>8} bytes")
        print(f"   Ratio:    {stats['compression_ratio']:>8.2f}:1")
        print(f"   Saving:   {stats['space_saving']:>7.2f}%")
        print(f"   Time:     {stats['time']:>8.4f}s")
    
    def _print_comparison_summary(self, results):
        """Print summary comparison of all methods"""
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Method':<20} {'Ratio':>8} {'Saving':>8} {'Time (s)':>10}")
        print("-" * 70)
        
        for method, stats in results.items():
            if method == 'pipeline':
                ratio = stats['ratio']
                saving = stats['saving']
                time_val = stats['time']
            else:
                ratio = stats['compression_ratio']
                saving = stats['space_saving']
                time_val = stats['time']
            
            print(f"{method.upper():<20} {ratio:>8.2f} {saving:>7.1f}% {time_val:>9.4f}")
        
        # Find best method
        best_ratio = max(results.values(), key=lambda x: x['compression_ratio'] if 'compression_ratio' in x else x['ratio'])
        best_method = [k for k, v in results.items() if v == best_ratio][0]
        
        print(f"\n🏆 BEST COMPRESSION: {best_method.upper()}")
        print(f"   Ratio: {best_ratio['compression_ratio'] if 'compression_ratio' in best_ratio else best_ratio['ratio']:.2f}:1")

def get_file_path_with_retry(prompt="Enter path to file: "):
    """Get file path with retry mechanism"""
    while True:
        file_path = input(prompt).strip()
        if file_path and os.path.exists(file_path):
            return file_path
        elif not file_path:
            print("✗ Please enter a file path.")
        else:
            print(f"✗ File '{file_path}' not found. Please try again.")

def show_current_directory_files():
    """Show files in current directory to help user"""
    print("\nFiles in current directory:")
    current_files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for i, file in enumerate(current_files[:10]):
        print(f"  {file}")
    if len(current_files) > 10:
        print(f"  ... and {len(current_files) - 10} more files")
    print()

def main():
    """Main application interface"""
    pipeline = MultiLosslessCodingPipeline()
    
    while True:
        print("\n" + "="*70)
        print("MULTI-LOSSLESS CODING PIPELINE - Complete Assignment")
        print("="*70)
        print("Choose operation:")
        print("1. Full Pipeline (Huffman → CABAC)")
        print("2. Compare All Methods (Huffman, CABAC, RLE, Pipeline)")
        print("3. Individual Algorithm Test")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4) [1]: ").strip() or "1"
        
        if choice == "4":
            print("Goodbye!")
            break
        
        print("\nChoose input type:")
        print("1. Image file (using Pillow)")
        print("2. Any file from hard disk")
        print("3. Use default sample data")
        
        input_choice = input("Enter choice (1-3) [1]: ").strip() or "1"
        
        if input_choice == "1":
            show_current_directory_files()
            file_path = get_file_path_with_retry("Enter path to image file: ")
            data, img_array = pipeline.load_image_data(file_path)
        elif input_choice == "2":
            show_current_directory_files()
            file_path = get_file_path_with_retry("Enter path to file: ")
            data = pipeline.load_file_data(file_path)
        else:
            # Create sample data
            sample_data = b"THIS IS A SAMPLE TEXT FOR COMPRESSION TESTING " * 50
            data = sample_data
            print("Using default sample data")
        
        if data is None:
            print("Failed to load data. Please try again.")
            continue
        
        if choice == "1":
            pipeline.huffman_cabac_pipeline(data)
        elif choice == "2":
            pipeline.compare_all_methods(data)
        elif choice == "3":
            # Individual algorithm testing
            print("\nChoose algorithm to test:")
            print("1. Huffman Coding")
            print("2. CABAC Coding") 
            print("3. RLE Encoding")
            
            algo_choice = input("Enter choice (1-3) [1]: ").strip() or "1"
            
            if algo_choice == "1":
                encoded, stats = pipeline.huffman_compression(data)
                print("\nHUFFMAN CODING RESULTS:")
                pipeline._print_stats(stats)
            elif algo_choice == "2":
                bits = ''.join(format(byte, '08b') for byte in data)
                encoded, stats = pipeline.cabac_compression(bits)
                print("\nCABAC CODING RESULTS:")
                pipeline._print_stats(stats)
            elif algo_choice == "3":
                encoded, stats = pipeline.rle_compression(data)
                print("\nRLE ENCODING RESULTS:")
                pipeline._print_stats(stats)
        
        # Ask to continue
        print("\n" + "="*70)
        continue_choice = input("Would you like to try another file? (y/n) [y]: ").strip().lower() or "y"
        if continue_choice not in ['y', 'yes']:
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()