import os, textwrap, json, sys, math, typing
import pyarrow  
import os
import struct
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, QLabel, QLineEdit,
    QComboBox, QSplitter, QTableWidget, QTableWidgetItem, QProgressBar, QCheckBox
)

# Matplotlib embedding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# -----------------------------
# Huffman Coding Implementation
# -----------------------------

@dataclass
class Node:
    freq: int
    byte: Optional[int] = None  # leaf: byte value 0..255
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    def __lt__(self, other):
        return self.freq < other.freq


class Huffman:
    MAGIC = b"HUF1"  # 4 bytes header
    # Header format: MAGIC + 256 * uint64 BE for byte frequencies + uint64 BE original_size

    @staticmethod
    def build_tree(freqs: Dict[int, int]) -> Optional[Node]:
        import heapq
        heap = []
        for b, f in freqs.items():
            if f > 0:
                heapq.heappush(heap, Node(freq=f, byte=b))
        if not heap:
            return None
        if len(heap) == 1:
            # special case: single symbol -> create a dummy parent
            only = heap[0]
            return Node(freq=only.freq, left=only, right=None)
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            parent = Node(freq=a.freq + b.freq, left=a, right=b)
            heapq.heappush(heap, parent)
        return heap[0]

    @staticmethod
    def build_codes(root: Optional[Node]) -> Dict[int, str]:
        codes: Dict[int, str] = {}
        if root is None:
            return codes
        def dfs(n: Node, path: str):
            if n.byte is not None:
                codes[n.byte] = path if path else "0"  # single-node case
                return
            if n.left is not None:
                dfs(n.left, path + "0")
            if n.right is not None:
                dfs(n.right, path + "1")
        dfs(root, "")
        return codes

    @staticmethod
    def encode_file(input_path: str, output_path: str):
        # First pass: frequencies
        freqs = {i: 0 for i in range(256)}
        total = 0
        with open(input_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                for b in chunk:
                    freqs[b] += 1

        # Build tree and codes
        root = Huffman.build_tree(freqs)
        codes = Huffman.build_codes(root)

        # Write header: MAGIC + 256 uint64 + original_size
        with open(output_path, "wb") as out:
            out.write(Huffman.MAGIC)
            for i in range(256):
                out.write(struct.pack(">Q", freqs[i]))
            out.write(struct.pack(">Q", total))

            # Second pass: write bitstream
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
                    # pad remaining bits with zeros
                    byte = (bit_buffer << (8 - bit_count)) & 0xFF
                    out.write(bytes([byte]))
                    bit_buffer = 0
                    bit_count = 0

            with open(input_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    for b in chunk:
                        code = codes[b]
                        for c in code:
                            bit_buffer = (bit_buffer << 1) | (1 if c == '1' else 0)
                            bit_count += 1
                            flush_bits(False)
                flush_bits(True)

    @staticmethod
    def decode_file(input_path: str, output_path: str):
        with open(input_path, "rb") as f:
            magic = f.read(4)
            if magic != Huffman.MAGIC:
                raise ValueError("Not a HUF1 file")
            freqs = {}
            for i in range(256):
                (val,) = struct.unpack(">Q", f.read(8))
                freqs[i] = val
            (original_size,) = struct.unpack(">Q", f.read(8))

            root = Huffman.build_tree(freqs)
            if root is None:
                # empty file
                open(output_path, "wb").close()
                return

            # Build decoding: traverse bits
            node = root
            written = 0
            out = open(output_path, "wb")
            try:
                data = f.read()
                for byte in data:
                    for i in range(7, -1, -1):
                        bit = (byte >> i) & 1
                        # walk tree
                        if node.byte is not None and node.right is None and node.left is not None:
                            # single-symbol special: always emit left leaf
                            out.write(bytes([node.left.byte]))
                            written += 1
                            if written >= original_size:
                                out.flush()
                                out.close()
                                return
                            node = root
                            continue
                        node = node.right if bit else node.left
                        if node is None:
                            # Shouldn't happen with correct padding
                            node = root
                            continue
                        if node.byte is not None:
                            out.write(bytes([node.byte]))
                            written += 1
                            if written >= original_size:
                                out.flush()
                                out.close()
                                return
                            node = root
            finally:
                out.close()

# -----------------------------
# Pandas Helpers
# -----------------------------

def read_csv_safely(path: str, sample_rows: int = 5000) -> pd.DataFrame:
    """
    Try to read a CSV with best-effort engine selection.
    Falls back between pyarrow, C, and Python engines safely.
    """
    try:
        # Try pyarrow first (fast, efficient)
        return pd.read_csv(path, engine="pyarrow")
    except Exception:
        try:
            # Then try C engine with low_memory disabled
            return pd.read_csv(path, engine="c", low_memory=False)
        except Exception:
            try:
                # Finally fall back to Python engine (no low_memory here)
                return pd.read_csv(path, engine="python")
            except Exception as e:
                raise RuntimeError(f"Failed to read CSV with all engines: {e}")



# -----------------------------
# Qt Table Helper
# -----------------------------

def populate_table_widget(table: QTableWidget, df: pd.DataFrame, max_rows: int = 500, max_cols: int = 50):
    """
    Populate QTableWidget from a DataFrame, limiting rows/cols for performance.
    """
    if df is None or df.empty:
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(0)
        return

    cols = list(df.columns)[:max_cols]
    sub = df.iloc[:max_rows][cols]

    table.setColumnCount(len(cols))
    table.setRowCount(len(sub))
    table.setHorizontalHeaderLabels([str(c) for c in cols])

    for r in range(len(sub)):
        row_vals = sub.iloc[r]
        for c, col in enumerate(cols):
            val = row_vals[col]
            item = QTableWidgetItem("" if pd.isna(val) else str(val))
            table.setItem(r, c, item)

    table.resizeColumnsToContents()


# -----------------------------
# Matplotlib Canvas
# -----------------------------

class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


# -----------------------------
# Main Window
# -----------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Huffman GUI — CSV Extract, Plot, Excel Export, Huffman Encode")
        self.resize(1200, 800)

        # Data
        self.df_full: Optional[pd.DataFrame] = None
        self.df_view: Optional[pd.DataFrame] = None
        self.current_csv_path: Optional[str] = None

        # Widgets
        self.btn_load = QPushButton("Load CSV")
        self.lbl_path = QLabel("No file loaded")
        self.lbl_path.setTextInteractionFlags(Qt.TextSelectableByMouse)

        top_left = QWidget()
        top_left_layout = QVBoxLayout(top_left)

        self.columns_list = QListWidget()
        self.columns_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        self.chk_select_all = QCheckBox("Select/Deselect All Columns")
        self.chk_select_all.stateChanged.connect(self.on_toggle_all_columns)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Row filter (pandas query), e.g. price > 100 and category == 'A'")

        self.btn_apply = QPushButton("Apply Extract")
        self.btn_export = QPushButton("Export to Excel")
        self.btn_encode = QPushButton("Huffman Encode Extracted CSV")
        self.btn_decode = QPushButton("Decode .huff → .csv")

        self.status = QProgressBar()
        self.status.setRange(0, 0)
        self.status.setVisible(False)

        # Plot controls
        self.cmb_x = QComboBox()
        self.cmb_y = QComboBox()
        self.cmb_plot_kind = QComboBox()
        self.cmb_plot_kind.addItems(["line", "scatter", "bar"])
        self.btn_plot = QPushButton("Plot")

        # Table + Plot
        self.table = QTableWidget()
        self.canvas = MplCanvas(width=6, height=4, dpi=100)

        # Layout splits
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(self.lbl_path)
        left_layout.addWidget(QLabel("Columns:"))
        left_layout.addWidget(self.chk_select_all)
        left_layout.addWidget(self.columns_list)
        left_layout.addWidget(QLabel("Row filter:"))
        left_layout.addWidget(self.filter_edit)
        left_layout.addWidget(self.btn_apply)
        left_layout.addWidget(self.btn_export)
        left_layout.addWidget(self.btn_encode)
        left_layout.addWidget(self.btn_decode)
        left_layout.addWidget(self.status)

        right_split = QSplitter(Qt.Vertical)
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.addWidget(QLabel("Preview (first 500 rows):"))
        table_layout.addWidget(self.table)

        plot_controls = QWidget()
        pc_layout = QHBoxLayout(plot_controls)
        pc_layout.addWidget(QLabel("X:"))
        pc_layout.addWidget(self.cmb_x)
        pc_layout.addWidget(QLabel("Y:"))
        pc_layout.addWidget(self.cmb_y)
        pc_layout.addWidget(QLabel("Kind:"))
        pc_layout.addWidget(self.cmb_plot_kind)
        pc_layout.addWidget(self.btn_plot)

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.addWidget(plot_controls)
        plot_layout.addWidget(self.canvas)

        right_split.addWidget(table_container)
        right_split.addWidget(plot_container)
        right_split.setSizes([400, 400])

        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(left_panel)
        main_split.addWidget(right_split)
        main_split.setSizes([350, 850])

        central = QWidget()
        central_layout = QVBoxLayout(central)
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.addWidget(QLabel("Huffman GUI — CSV Extract, Plot, Excel Export, Huffman Encode"))
        central_layout.addWidget(header)
        central_layout.addWidget(main_split)

        self.setCentralWidget(central)

        # Connections
        self.btn_load.clicked.connect(self.on_load_csv)
        self.btn_apply.clicked.connect(self.on_apply_extract)
        self.btn_export.clicked.connect(self.on_export_excel)
        self.btn_plot.clicked.connect(self.on_plot)
        self.btn_encode.clicked.connect(self.on_huffman_encode)
        self.btn_decode.clicked.connect(self.on_huffman_decode)

    # ---------- UI helpers ----------

    def set_busy(self, busy: bool):
        self.status.setVisible(busy)
        QApplication.setOverrideCursor(Qt.WaitCursor if busy else Qt.ArrowCursor)
        QApplication.processEvents()

    def message(self, text: str, title: str = "Info"):
        QMessageBox.information(self, title, text)

    def error(self, text: str, title: str = "Error"):
        QMessageBox.critical(self, title, text)

    def on_toggle_all_columns(self, state: int):
        for i in range(self.columns_list.count()):
            item = self.columns_list.item(i)
            item.setSelected(state == Qt.Checked)

    # ---------- Data operations ----------

    def on_load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        self.set_busy(True)
        try:
            df = read_csv_safely(path)
            self.df_full = df
            self.df_view = df.copy()
            self.current_csv_path = path

            # Fill columns list
            self.columns_list.clear()
            for col in df.columns:
                item = QListWidgetItem(str(col))
                item.setSelected(True)  # default select all
                self.columns_list.addItem(item)

            # Fill plot combos
            self.refresh_plot_combos()

            # Preview
            populate_table_widget(self.table, self.df_view)

            self.lbl_path.setText(path)
            self.message(f"Loaded CSV with shape {df.shape[0]} rows × {df.shape[1]} columns.", "Loaded")
        except Exception as e:
            self.error(f"Failed to load CSV:\n{e}")
        finally:
            self.set_busy(False)

    def refresh_plot_combos(self):
        self.cmb_x.clear()
        self.cmb_y.clear()
        if self.df_view is None or self.df_view.empty:
            return
        cols = list(self.df_view.columns)
        self.cmb_x.addItems([str(c) for c in cols])
        # Prefer numeric for Y
        num_cols = list(self.df_view.select_dtypes(include="number").columns)
        self.cmb_y.addItems([str(c) for c in (num_cols if num_cols else cols)])

    def on_apply_extract(self):
        if self.df_full is None:
            self.error("Load a CSV first.")
            return
        selected_cols = [item.text() for item in self.columns_list.selectedItems()]
        query_text = self.filter_edit.text().strip()

        self.set_busy(True)
        try:
            df = self.df_full
            if query_text:
                try:
                    df = df.query(query_text, engine="python")
                except Exception as e:
                    self.error(f"Invalid filter expression:\n{e}")
                    return
            if selected_cols:
                missing = [c for c in selected_cols if c not in df.columns]
                if missing:
                    self.error(f"Some selected columns are missing in the data: {missing}")
                    return
                df = df[selected_cols]
            self.df_view = df
            populate_table_widget(self.table, df)
            self.refresh_plot_combos()
            self.message(f"Extracted shape: {df.shape[0]} × {df.shape[1]}", "Extract applied")
        finally:
            self.set_busy(False)

    def on_export_excel(self):
        if self.df_view is None or self.df_view.empty:
            self.error("Nothing to export. Apply an extract first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Excel", "", "Excel Files (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        self.set_busy(True)
        try:
            self.df_view.to_excel(path, index=False)
            self.message(f"Saved: {path}", "Export complete")
        except Exception as e:
            self.error(f"Failed to export Excel:\n{e}")
        finally:
            self.set_busy(False)

    def on_plot(self):
        if self.df_view is None or self.df_view.empty:
            self.error("Nothing to plot.")
            return
        xcol = self.cmb_x.currentText()
        ycol = self.cmb_y.currentText()
        kind = self.cmb_plot_kind.currentText()

        if xcol not in self.df_view.columns or ycol not in self.df_view.columns:
            self.error("Invalid X or Y column.")
            return
        self.canvas.ax.clear()
        try:
            x = self.df_view[xcol]
            y = self.df_view[ycol]
            if kind == "line":
                self.canvas.ax.plot(x, y)
            elif kind == "scatter":
                self.canvas.ax.scatter(x, y)
            elif kind == "bar":
                # For bar, limit to first 100 for readability
                self.canvas.ax.bar(x.head(100), y.head(100))
            self.canvas.ax.set_xlabel(xcol)
            self.canvas.ax.set_ylabel(ycol)
            self.canvas.ax.set_title(f"{kind.title()} plot of {ycol} vs {xcol}")
            self.canvas.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            self.error(f"Plot failed:\n{e}")

    def on_huffman_encode(self):
        if self.df_view is None or self.df_view.empty:
            self.error("Nothing to encode. Apply an extract first.")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Huffman (.huff)", "", "Huffman Files (*.huff)")
        if not out_path:
            return
        if not out_path.lower().endswith(".huff"):
            out_path += ".huff"

        # Persist current extract as a temporary CSV, then encode that CSV.
        self.set_busy(True)
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp_csv = os.path.join(td, "extract.csv")
                self.df_view.to_csv(tmp_csv, index=False)
                Huffman.encode_file(tmp_csv, out_path)
            self.message(f"Encoded to: {out_path}", "Huffman encode")
        except Exception as e:
            self.error(f"Encoding failed:\n{e}")
        finally:
            self.set_busy(False)

    def on_huffman_decode(self):
        in_path, _ = QFileDialog.getOpenFileName(self, "Open Huffman File", "", "Huffman Files (*.huff)")
        if not in_path:
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Decoded CSV", "", "CSV Files (*.csv)")
        if not out_path:
            return
        if not out_path.lower().endswith(".csv"):
            out_path += ".csv"
        self.set_busy(True)
        try:
            Huffman.decode_file(in_path, out_path)
            self.message(f"Decoded to: {out_path}", "Huffman decode")
        except Exception as e:
            self.error(f"Decoding failed:\n{e}")
        finally:
            self.set_busy(False)


def main():
    import sys
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

