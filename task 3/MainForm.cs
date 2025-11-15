using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using System.Diagnostics;
using System.Linq;

namespace CompressionVisualizer
{
    public partial class MainForm : Form
    {
        private Bitmap? originalImage;
        private string? currentImagePath;
        private readonly CompressionAlgorithms algorithms;

        // UI Controls
        private ComboBox algorithmComboBox = null!;
        private Button loadImageButton = null!;
        private Button compressButton = null!;
        private Button compareButton = null!;
        private PictureBox originalPictureBox = null!;
        private PictureBox compressedPictureBox = null!;
        private ListView resultsListView = null!;
        private ListView comparisonListView = null!; 
        private Label statusLabel = null!;
        private ProgressBar progressBar = null!;
        private TabControl tabControl = null!;

        public MainForm()
        {
            algorithms = new CompressionAlgorithms();
            InitializeComponent();
            this.Text = "Compression Algorithms Visualizer - Task 3";
            this.Size = new Size(1200, 800);
        }

        private void InitializeComponent()
        {
            // Initialize all controls
            algorithmComboBox = new ComboBox();
            loadImageButton = new Button();
            compressButton = new Button();
            compareButton = new Button();
            originalPictureBox = new PictureBox();
            compressedPictureBox = new PictureBox();
            resultsListView = new ListView();
            comparisonListView = new ListView();
            statusLabel = new Label();
            progressBar = new ProgressBar();
            tabControl = new TabControl();

            // Layout
            var mainTable = new TableLayoutPanel();
            mainTable.Dock = DockStyle.Fill;
            mainTable.RowCount = 4;
            mainTable.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
            mainTable.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
            mainTable.RowStyles.Add(new RowStyle(SizeType.Percent, 70));
            mainTable.RowStyles.Add(new RowStyle(SizeType.Absolute, 30));

            // Controls panel
            var controlsPanel = new Panel { Dock = DockStyle.Fill, Height = 40 };

            algorithmComboBox.Location = new Point(10, 10);
            algorithmComboBox.Size = new Size(150, 25);
            algorithmComboBox.DropDownStyle = ComboBoxStyle.DropDownList;
            algorithmComboBox.Items.AddRange(new[] { "All Algorithms", "RLE", "Huffman", "Arithmetic", "CABAC" });
            algorithmComboBox.SelectedIndex = 0;

            loadImageButton.Location = new Point(170, 10);
            loadImageButton.Size = new Size(100, 25);
            loadImageButton.Text = "Load Image";
            loadImageButton.Click += LoadImageButton_Click;

            compressButton.Location = new Point(280, 10);
            compressButton.Size = new Size(100, 25);
            compressButton.Text = "Compress";
            compressButton.Click += CompressButton_Click;

            compareButton.Location = new Point(390, 10);
            compareButton.Size = new Size(100, 25);
            compareButton.Text = "Compare All";
            compareButton.Click += CompareButton_Click;

            controlsPanel.Controls.AddRange(new Control[] { algorithmComboBox, loadImageButton, compressButton, compareButton });

            // Progress panel
            var progressPanel = new Panel { Dock = DockStyle.Fill, Height = 40 };
            progressBar.Location = new Point(10, 10);
            progressBar.Size = new Size(500, 20);
            progressBar.Visible = false;
            progressPanel.Controls.Add(progressBar);

            // Tab control
            tabControl.Dock = DockStyle.Fill;
            var singleTab = new TabPage("Single Algorithm");
            var compareTab = new TabPage("Comparison");

            // --- Single Algorithm tab ---
            var singleTable = new TableLayoutPanel();
            singleTable.Dock = DockStyle.Fill;
            singleTable.RowCount = 1;
            singleTable.ColumnCount = 2;
            singleTable.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
            singleTable.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));

            originalPictureBox.Dock = DockStyle.Fill;
            originalPictureBox.SizeMode = PictureBoxSizeMode.Zoom;
            originalPictureBox.BorderStyle = BorderStyle.FixedSingle;

            compressedPictureBox.Dock = DockStyle.Fill;
            compressedPictureBox.SizeMode = PictureBoxSizeMode.Zoom;
            compressedPictureBox.BorderStyle = BorderStyle.FixedSingle;

            singleTable.Controls.Add(originalPictureBox, 0, 0);
            singleTable.Controls.Add(compressedPictureBox, 1, 0);

            resultsListView.Dock = DockStyle.Fill;
            resultsListView.View = View.Details;
            resultsListView.FullRowSelect = true;
            resultsListView.Columns.Add("Metric", 200);
            resultsListView.Columns.Add("Value", 200);

            var splitContainer = new SplitContainer();
            splitContainer.Dock = DockStyle.Fill;
            splitContainer.Orientation = Orientation.Horizontal;
            splitContainer.Panel1.Controls.Add(singleTable);
            splitContainer.Panel2.Controls.Add(resultsListView);
            splitContainer.SplitterDistance = 400;

            singleTab.Controls.Add(splitContainer);

            // --- Comparison tab  ---
            comparisonListView.Dock = DockStyle.Fill;
            comparisonListView.View = View.Details;
            comparisonListView.FullRowSelect = true;
            comparisonListView.GridLines = true;
            comparisonListView.FullRowSelect = true;
            comparisonListView.Columns.Add("Algorithm", 150);
            comparisonListView.Columns.Add("Compressed Size", 150);
            comparisonListView.Columns.Add("Compression Ratio", 150);
            comparisonListView.Columns.Add("Space Saving", 150);
            comparisonListView.Columns.Add("Time (s)", 100);
            comparisonListView.Columns.Add("PSNR", 100);

            compareTab.Controls.Add(comparisonListView);

            tabControl.TabPages.Add(singleTab);
            tabControl.TabPages.Add(compareTab);

            // Status label
            statusLabel.Dock = DockStyle.Fill;
            statusLabel.Text = "Ready to load and compress images";
            statusLabel.TextAlign = ContentAlignment.MiddleLeft;

            mainTable.Controls.Add(controlsPanel, 0, 0);
            mainTable.Controls.Add(progressPanel, 0, 1);
            mainTable.Controls.Add(tabControl, 0, 2);
            mainTable.Controls.Add(statusLabel, 0, 3);

            this.Controls.Add(mainTable);
        }

        private void LoadImageButton_Click(object? sender, EventArgs e)
        {
            using var openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Image Files|*.png;*.jpg;*.jpeg;*.bmp;*.tiff";
            openFileDialog.Title = "Select an Image File";

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    currentImagePath = openFileDialog.FileName;
                    originalImage = new Bitmap(currentImagePath);
                    originalPictureBox.Image = originalImage;

                    statusLabel.Text = $"Loaded: {Path.GetFileName(currentImagePath)} - Size: {originalImage.Width}x{originalImage.Height}";

                    UpdateResults(new List<string[]>
                    {
                        new[] { "File Name", Path.GetFileName(currentImagePath) },
                        new[] { "Dimensions", $"{originalImage.Width} x {originalImage.Height}" },
                        new[] { "Pixel Format", originalImage.PixelFormat.ToString() },
                        new[] { "Estimated Size", $"{GetImageSizeInBytes(originalImage):N0} bytes" }
                    });
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Error loading image: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void CompressButton_Click(object? sender, EventArgs e)
        {
            if (originalImage == null)
            {
                MessageBox.Show("Please load an image first.", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            var selectedAlgorithm = algorithmComboBox.SelectedItem?.ToString();
            if (selectedAlgorithm == "All Algorithms")
            {
                RunComparison();
                return;
            }

            if (selectedAlgorithm != null)
                RunSingleAlgorithm(selectedAlgorithm);
        }

        private void CompareButton_Click(object? sender, EventArgs e)
        {
            if (originalImage == null)
            {
                MessageBox.Show("Please load an image first.", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            RunComparison();
        }

        private void RunSingleAlgorithm(string algorithmName)
        {
            try
            {
                statusLabel.Text = $"Running {algorithmName} compression...";
                progressBar.Visible = true;
                progressBar.Value = 0;
                Application.DoEvents();

                var stopwatch = Stopwatch.StartNew();

                CompressionResult result = algorithmName switch
                {
                    "RLE" => algorithms.RunRLE(originalImage!),
                    "Huffman" => algorithms.RunHuffman(originalImage!),
                    "Arithmetic" => algorithms.RunArithmetic(originalImage!),
                    "CABAC" => algorithms.RunCABAC(originalImage!),
                    _ => throw new ArgumentException("Unknown algorithm")
                };

                stopwatch.Stop();
                result.ExecutionTime = stopwatch.Elapsed.TotalSeconds;
                compressedPictureBox.Image = result.DecompressedImage;

                var originalSize = GetImageSizeInBytes(originalImage!);
                var compressionRatio = originalSize / (double)result.CompressedSize;
                var spaceSaving = ((originalSize - result.CompressedSize) / (double)originalSize) * 100;

                var results = new List<string[]>
                {
                    new[] { "Algorithm", algorithmName },
                    new[] { "Original Size", $"{originalSize:N0} bytes" },
                    new[] { "Compressed Size", $"{result.CompressedSize:N0} bytes" },
                    new[] { "Compression Ratio", $"{compressionRatio:F2}:1" },
                    new[] { "Space Saving", $"{spaceSaving:F2}%" },
                    new[] { "Execution Time", $"{result.ExecutionTime:F4} seconds" },
                    new[] { "PSNR", result.PSNR == double.PositiveInfinity ? "∞ dB" : $"{result.PSNR:F2} dB" }
                };

                results.AddRange(result.AdditionalInfo);

                UpdateResults(results);
                statusLabel.Text = $"{algorithmName} compression completed successfully";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error during compression: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                statusLabel.Text = "Compression failed";
            }
            finally
            {
                progressBar.Visible = false;
            }
        }

        private void RunComparison()
        {
            try
            {
                statusLabel.Text = "Running comparison of all algorithms...";
                progressBar.Visible = true;
                progressBar.Value = 0;
                Application.DoEvents();

                comparisonListView.Items.Clear();

                var algorithmsToRun = new[] { "RLE", "Huffman", "Arithmetic", "CABAC" };
                var originalSize = GetImageSizeInBytes(originalImage!);

                for (int i = 0; i < algorithmsToRun.Length; i++)
                {
                    progressBar.Value = (i * 100) / algorithmsToRun.Length;
                    Application.DoEvents();

                    var stopwatch = Stopwatch.StartNew();
                    CompressionResult result = algorithmsToRun[i] switch
                    {
                        "RLE" => algorithms.RunRLE(originalImage!),
                        "Huffman" => algorithms.RunHuffman(originalImage!),
                        "Arithmetic" => algorithms.RunArithmetic(originalImage!),
                        "CABAC" => algorithms.RunCABAC(originalImage!),
                        _ => throw new ArgumentException("Unknown algorithm")
                    };

                    stopwatch.Stop();
                    result.ExecutionTime = stopwatch.Elapsed.TotalSeconds;

                    var compressionRatio = originalSize / (double)result.CompressedSize;
                    var spaceSaving = ((originalSize - result.CompressedSize) / (double)originalSize) * 100;

                    var item = new ListViewItem(result.AlgorithmName ?? algorithmsToRun[i]);
                    item.SubItems.Add($"{result.CompressedSize:N0} bytes");
                    item.SubItems.Add($"{compressionRatio:F2}:1");
                    item.SubItems.Add($"{spaceSaving:F2}%");
                    item.SubItems.Add($"{result.ExecutionTime:F4}");
                    item.SubItems.Add(result.PSNR == double.PositiveInfinity ? "∞ dB" : $"{result.PSNR:F2} dB");
                    comparisonListView.Items.Add(item);
                }

                statusLabel.Text = "Comparison completed successfully";
                tabControl.SelectedIndex = 1;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error during comparison: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                statusLabel.Text = "Comparison failed";
            }
            finally
            {
                progressBar.Visible = false;
            }
        }

        private void UpdateResults(List<string[]> results)
        {
            resultsListView.Items.Clear();
            foreach (var row in results)
            {
                var item = new ListViewItem(row[0]);
                for (int i = 1; i < row.Length; i++)
                    item.SubItems.Add(row[i]);
                resultsListView.Items.Add(item);
            }
        }

        private long GetImageSizeInBytes(Bitmap image)
        {
            return image.Width * image.Height * Image.GetPixelFormatSize(image.PixelFormat) / 8;
        }
    }

    public class CompressionResult
    {
        public string? AlgorithmName { get; set; }
        public int CompressedSize { get; set; }
        public double ExecutionTime { get; set; }
        public double PSNR { get; set; }
        public Bitmap? DecompressedImage { get; set; }
        public List<string[]> AdditionalInfo { get; set; } = new List<string[]>();
    }
}
