import sys
import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import pyworld as pw
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTextEdit,
    QFileDialog,
    QSlider,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent


def _analyze_and_convert(x, fs):
    _f0, t = pw.dio(x, fs)
    f0 = pw.stonemask(x, _f0, t, fs)
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)

    valid_f0 = f0[f0 > 0]
    if len(valid_f0) == 0:
        raise RuntimeError("未检测到有效的人声基频。")

    median_f0 = np.median(valid_f0)
    threshold = 165.0
    if median_f0 < threshold:
        detected = "男声"
        pitch_ratio = 220.0 / median_f0
        formant_ratio = 1.18
    else:
        detected = "女声"
        pitch_ratio = 120.0 / median_f0
        formant_ratio = 0.85

    modified_f0 = f0 * pitch_ratio
    modified_sp = np.zeros_like(sp)
    num_bins = sp.shape[1]
    old_freq_axis = np.arange(num_bins)
    for i in range(sp.shape[0]):
        new_freq_axis = old_freq_axis / formant_ratio
        modified_sp[i, :] = np.interp(new_freq_axis, old_freq_axis, sp[i, :])

    y = pw.synthesize(modified_f0, modified_sp, ap, fs)
    return y, f0, modified_f0, median_f0, detected


def plot_f0_histogram(input_path, save_path, bins=50):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, fs = librosa.load(input_path, sr=None, dtype=np.float64)
    _, f0_before, f0_after, _, _ = _analyze_and_convert(x, fs)

    valid_f0_before = f0_before[f0_before > 0]
    valid_f0_after = f0_after[f0_after > 0]
    if len(valid_f0_before) == 0 or len(valid_f0_after) == 0:
        raise RuntimeError("未检测到有效基频，无法绘制直方图。")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(valid_f0_before, bins=bins, color="#FF9800", edgecolor="#222", alpha=0.7)
    ax1.set_xlabel("F0 (Hz)")
    ax1.set_ylabel("Count")
    ax1.set_title("Before Conversion", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    ax2.hist(valid_f0_after, bins=bins, color="#4CAF50", edgecolor="#222", alpha=0.7)
    ax2.set_xlabel("F0 (Hz)")
    ax2.set_ylabel("Count")
    ax2.set_title("After Conversion", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"F0 Comparison - {os.path.basename(input_path)}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_mel_spectrogram_comparison(input_path, converted_path, save_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, fs = librosa.load(input_path, sr=None, dtype=np.float32)
    y, fs2 = librosa.load(converted_path, sr=fs, dtype=np.float32)
    if fs2 != fs:
        raise RuntimeError("采样率不一致，无法绘制 Mel 对比图。")

    n_fft = 1024
    hop_length = 256
    n_mels = 128

    mel_before = librosa.feature.melspectrogram(
        y=x, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    mel_after = librosa.feature.melspectrogram(
        y=y, sr=fs, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )

    mel_before_db = librosa.power_to_db(mel_before, ref=np.max)
    mel_after_db = librosa.power_to_db(mel_after, ref=np.max)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    img1 = librosa.display.specshow(
        mel_before_db,
        sr=fs,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=axes[0],
    )
    axes[0].set_title("Mel Spectrogram - Before", fontweight="bold")

    librosa.display.specshow(
        mel_after_db,
        sr=fs,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=axes[1],
    )
    axes[1].set_title("Mel Spectrogram - After", fontweight="bold")

    cbar = fig.colorbar(img1, ax=axes.ravel().tolist(), format="%+2.0f dB", shrink=0.85)
    cbar.set_label("dB")
    fig.suptitle(f"Mel Spectrogram Comparison - {os.path.basename(input_path)}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


class VoiceConverterThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, input_path):
        super().__init__()
        self.input_path = input_path

    def run(self):
        try:
            self.log_signal.emit(f"正在加载音频: {os.path.basename(self.input_path)}")
            x, fs = librosa.load(self.input_path, sr=None, dtype=np.float64)
            self.log_signal.emit("正在使用 WORLD 提取特征...")

            y, _, _, median_f0, detected = _analyze_and_convert(x, fs)
            self.log_signal.emit(f"基频中位数: {median_f0:.1f} Hz")
            self.log_signal.emit(f"识别结果: {detected}")

            output_dir = os.path.dirname(self.input_path)
            output_name = "converted_" + os.path.basename(self.input_path)
            output_path = os.path.join(output_dir, output_name)
            sf.write(output_path, y, fs)

            self.log_signal.emit("转换成功")
            self.finished_signal.emit(output_path)
        except Exception as e:
            self.log_signal.emit(f"发生异常: {e}")
            self.finished_signal.emit("")


class PlotF0Thread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, input_path, save_path):
        super().__init__()
        self.input_path = input_path
        self.save_path = save_path

    def run(self):
        try:
            self.log_signal.emit(f"绘制 F0 对比图: {os.path.basename(self.input_path)}")
            plot_f0_histogram(self.input_path, self.save_path)
            self.log_signal.emit("F0 对比图已保存")
            self.finished_signal.emit(self.save_path)
        except Exception as e:
            self.log_signal.emit(f"F0 绘图失败: {e}")
            self.finished_signal.emit("")


class PlotMelThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, input_path, converted_path, save_path):
        super().__init__()
        self.input_path = input_path
        self.converted_path = converted_path
        self.save_path = save_path

    def run(self):
        try:
            self.log_signal.emit("绘制 Mel 语谱图对比...")
            plot_mel_spectrogram_comparison(self.input_path, self.converted_path, self.save_path)
            self.log_signal.emit("Mel 对比图已保存")
            self.finished_signal.emit(self.save_path)
        except Exception as e:
            self.log_signal.emit(f"Mel 绘图失败: {e}")
            self.finished_signal.emit("")


class DragDropLabel(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setText("\n\n拖动 WAV 音频文件到这里\n\n")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f9f9f9;
                font-size: 16px;
                color: #555;
            }
            """
        )
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                "border: 2px dashed #4CAF50; background-color: #e8f5e9; font-size: 16px;"
            )

    def dragLeaveEvent(self, event):
        self.setStyleSheet("border: 2px dashed #aaa; background-color: #f9f9f9; font-size: 16px;")

    def dropEvent(self, event):
        self.setStyleSheet("border: 2px dashed #aaa; background-color: #f9f9f9; font-size: 16px;")
        urls = event.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()
            if filepath.lower().endswith(".wav"):
                self.file_dropped.emit(filepath)
            else:
                self.setText("\n\n请拖入 WAV 格式的音频\n\n")


class VoiceChangerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_filepath = None
        self.converted_filepath = None

        self.player = QMediaPlayer(self)
        self.player.positionChanged.connect(self.on_player_position_changed)
        self.player.durationChanged.connect(self.on_player_duration_changed)

        self.initUI()

    def initUI(self):
        self.setWindowTitle("智能语音隐私保护与声线转换系统 - 融合增强版")
        self.resize(760, 560)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.drop_label = DragDropLabel()
        self.drop_label.file_dropped.connect(self.load_file)
        layout.addWidget(self.drop_label)

        self.btn_convert = QPushButton("开始转换")
        self.btn_convert.setMinimumHeight(38)
        self.btn_convert.setStyleSheet(
            "font-size: 15px; font-weight: bold; background-color: #4CAF50; color: white;"
        )
        self.btn_convert.clicked.connect(self.start_conversion)
        self.btn_convert.setEnabled(False)
        layout.addWidget(self.btn_convert)

        plot_row = QHBoxLayout()
        self.btn_plot_f0 = QPushButton("保存 F0 对比图")
        self.btn_plot_f0.setStyleSheet("font-size: 14px; background-color: #2196F3; color: white;")
        self.btn_plot_f0.clicked.connect(self.on_plot_f0_clicked)
        self.btn_plot_f0.setEnabled(False)
        plot_row.addWidget(self.btn_plot_f0)

        self.btn_plot_mel = QPushButton("保存 Mel 对比图")
        self.btn_plot_mel.setStyleSheet("font-size: 14px; background-color: #FF7043; color: white;")
        self.btn_plot_mel.clicked.connect(self.on_plot_mel_clicked)
        self.btn_plot_mel.setEnabled(False)
        plot_row.addWidget(self.btn_plot_mel)
        layout.addLayout(plot_row)

        player_row = QHBoxLayout()
        self.btn_play_original = QPushButton("播放原声")
        self.btn_play_original.clicked.connect(self.play_original)
        self.btn_play_original.setEnabled(False)
        player_row.addWidget(self.btn_play_original)

        self.btn_play_converted = QPushButton("播放转换后")
        self.btn_play_converted.clicked.connect(self.play_converted)
        self.btn_play_converted.setEnabled(False)
        player_row.addWidget(self.btn_play_converted)

        self.btn_stop = QPushButton("停止播放")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_stop.setEnabled(False)
        player_row.addWidget(self.btn_stop)
        layout.addLayout(player_row)

        self.slider_position = QSlider(Qt.Horizontal)
        self.slider_position.setRange(0, 0)
        self.slider_position.sliderMoved.connect(self.seek_position)
        layout.addWidget(self.slider_position)

        self.label_time = QLabel("00:00 / 00:00")
        self.label_time.setAlignment(Qt.AlignRight)
        layout.addWidget(self.label_time)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #2b2b2b; color: #a9b7c6; font-family: Consolas;")
        layout.addWidget(self.log_text)

    def load_file(self, filepath):
        self.current_filepath = filepath
        self.converted_filepath = None

        self.drop_label.setText(f"\n\n已加载:\n{os.path.basename(filepath)}\n\n")
        self.log_message(f"已准备好文件: {filepath}")

        self.btn_convert.setEnabled(True)
        self.btn_plot_f0.setEnabled(True)
        self.btn_plot_mel.setEnabled(False)
        self.btn_play_original.setEnabled(True)
        self.btn_play_converted.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def log_message(self, msg):
        self.log_text.append(msg)

    def start_conversion(self):
        if not self.current_filepath:
            return

        self.btn_convert.setEnabled(False)
        self.btn_convert.setText("正在转换中...")

        self.thread = VoiceConverterThread(self.current_filepath)
        self.thread.log_signal.connect(self.log_message)
        self.thread.finished_signal.connect(self.conversion_finished)
        self.thread.start()

    def conversion_finished(self, output_path):
        self.btn_convert.setEnabled(True)
        self.btn_convert.setText("开始转换")
        if output_path:
            self.converted_filepath = output_path
            self.log_message(f"文件已保存至: {output_path}")
            self.log_message("-" * 32)
            self.btn_play_converted.setEnabled(True)
            self.btn_plot_mel.setEnabled(True)
        else:
            self.log_message("转换失败。")

    def on_plot_f0_clicked(self):
        if not self.current_filepath:
            return
        suggested = os.path.splitext(os.path.basename(self.current_filepath))[0] + "_f0_hist.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存 F0 对比图", suggested, "PNG Files (*.png);;All Files (*)"
        )
        if not save_path:
            return

        self.btn_plot_f0.setEnabled(False)
        self.log_message(f"正在保存 F0 对比图: {save_path}")
        self.plot_f0_thread = PlotF0Thread(self.current_filepath, save_path)
        self.plot_f0_thread.log_signal.connect(self.log_message)
        self.plot_f0_thread.finished_signal.connect(self.plot_f0_finished)
        self.plot_f0_thread.start()

    def plot_f0_finished(self, path):
        self.btn_plot_f0.setEnabled(True)
        if path:
            self.log_message(f"F0 图像已保存: {path}")
        else:
            self.log_message("F0 绘图未完成或失败。")

    def on_plot_mel_clicked(self):
        if not self.current_filepath:
            return
        if not self.converted_filepath or not os.path.exists(self.converted_filepath):
            self.log_message("请先完成转换，再生成 Mel 对比图。")
            return

        suggested = os.path.splitext(os.path.basename(self.current_filepath))[0] + "_mel_compare.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存 Mel 对比图", suggested, "PNG Files (*.png);;All Files (*)"
        )
        if not save_path:
            return

        self.btn_plot_mel.setEnabled(False)
        self.log_message(f"正在保存 Mel 对比图: {save_path}")
        self.plot_mel_thread = PlotMelThread(self.current_filepath, self.converted_filepath, save_path)
        self.plot_mel_thread.log_signal.connect(self.log_message)
        self.plot_mel_thread.finished_signal.connect(self.plot_mel_finished)
        self.plot_mel_thread.start()

    def plot_mel_finished(self, path):
        self.btn_plot_mel.setEnabled(True)
        if path:
            self.log_message(f"Mel 图像已保存: {path}")
        else:
            self.log_message("Mel 绘图未完成或失败。")

    def _play_file(self, file_path, tag):
        if not file_path or not os.path.exists(file_path):
            self.log_message(f"{tag}文件不存在，无法播放。")
            return
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        self.player.play()
        self.log_message(f"正在播放 {tag}: {os.path.basename(file_path)}")

    def play_original(self):
        self._play_file(self.current_filepath, "原声")

    def play_converted(self):
        self._play_file(self.converted_filepath, "转换后")

    def stop_playback(self):
        self.player.stop()
        self.log_message("播放已停止。")

    def on_player_position_changed(self, position_ms):
        self.slider_position.blockSignals(True)
        self.slider_position.setValue(position_ms)
        self.slider_position.blockSignals(False)

        duration = max(self.player.duration(), 0)
        self.label_time.setText(f"{self._fmt_ms(position_ms)} / {self._fmt_ms(duration)}")

    def on_player_duration_changed(self, duration_ms):
        self.slider_position.setRange(0, max(duration_ms, 0))
        self.label_time.setText(f"00:00 / {self._fmt_ms(duration_ms)}")

    def seek_position(self, value_ms):
        self.player.setPosition(value_ms)

    @staticmethod
    def _fmt_ms(ms):
        seconds = int(ms / 1000)
        m = seconds // 60
        s = seconds % 60
        return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = VoiceChangerApp()
    ex.show()
    sys.exit(app.exec_())
