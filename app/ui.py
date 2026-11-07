"""Main window (PySide6)."""
import re
import threading
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QSyntaxHighlighter,
                           QTextCharFormat, QTextCursor)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QTabWidget,
                               QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMenu, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSlider,
                               QSplitter, QToolButton, QVBoxLayout, QWidget)

from . import audio_io, config, gemini, tags
from . import llm, net, providers
from .engines import SR
from .gemini_tts import DELIVERY
from .synth import Cancelled, RenderOptions, render

DEFAULT_MODELS = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-3.5-flash", "gemini-2.5-flash-lite"]
DEFAULT_TTS_MODELS = ["gemini-3.8-flash-tts", "gemini-3.1-flash-tts-preview", "gemini-3.8-flash-lite-tts", "gemini-2.5-flash-preview-tts"]
PREVIEW_TEXT = {
    "en": "Hello! This is how my voice sounds. I hope you like it.",
    "bn": "হ্যালো! আমার কণ্ঠ এমন শোনায়। আশা করি আপনার ভালো লাগবে।",
}


# ------------------------------------------------------------------ helpers
class TagHighlighter(QSyntaxHighlighter):
    def __init__(self, doc):
        super().__init__(doc)
        self.ok = QTextCharFormat()
        self.ok.setForeground(QColor("#2f7de1"))
        self.ok.setFontWeight(QFont.Bold)
        self.bad = QTextCharFormat()
        self.bad.setForeground(QColor("#d64545"))
        self.bad.setFontStrikeOut(True)

    def highlightBlock(self, text):
        for m in tags.TAG_RE.finditer(text):
            fmt = self.ok if tags.is_known(m.group(1)) else self.bad
            self.setFormat(m.start(), m.end() - m.start(), fmt)


class Worker(QObject):
    progress = Signal(int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @Slot()
    def run(self):
        try:
            self.done.emit(self.fn(self.progress.emit))
        except Cancelled:
            self.failed.emit("__cancelled__")
        except Exception as e:  # shown to the user
            self.failed.emit(f"{type(e).__name__}: {e}")


def fmt_time(ms: int) -> str:
    s = max(0, ms // 1000)
    return f"{s // 60}:{s % 60:02d}"


def slug(text: str) -> str:
    import unicodedata
    clean = "".join(ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in tags.strip_tags(text))
    return ("_".join(clean.split()[:5]) or "audio")[:40]


# ------------------------------------------------------------------ settings
def _key_row(settings, setting, name, url, free, store):
    """Key field + 'Show' + 'Get a free key' link. Returns a QGroupBox-ready form layout."""
    edit = QLineEdit(settings.get(setting, ""))
    edit.setEchoMode(QLineEdit.Password)
    edit.setPlaceholderText(f"Paste your free {name} API key")
    show = QCheckBox("Show")
    show.toggled.connect(lambda on: edit.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
    row = QHBoxLayout()
    row.addWidget(edit, 1)
    row.addWidget(show)
    store[setting] = edit
    link = QLabel(f'<a href="{url}">Get a free key →</a>   <span style="color:gray">Free: {free}</span>')
    link.setOpenExternalLinks(True)
    link.setWordWrap(True)
    return row, link


class SettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(640)
        self.settings = settings
        self.eleven_voices = None
        self.edits: dict[str, QLineEdit] = {}      # settings key -> field

        lay = QVBoxLayout(self)
        intro = QLabel("Everything here is free. Click a link, sign in, create a key and paste it. "
                       "You only need one key to start (Gemini is best). Keys stay on this PC (settings.json).")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        tabs = QTabWidget()
        lay.addWidget(tabs, 1)

        # Voices tab
        vt = QWidget()
        vl = QVBoxLayout(vt)
        for prov in providers.PROVIDERS:
            box = QGroupBox(f"{prov.name}: {prov.blurb}")
            form = QFormLayout(box)
            if prov.key_setting:
                row, link = _key_row(settings, prov.key_setting, prov.name, prov.key_url, prov.free, self.edits)
                form.addRow("API key:", row)
                form.addRow("", link)
            else:
                form.addRow(QLabel(f'<span style="color:gray">{prov.free}</span>'))
            if prov.id == "gemini":
                self.tts_model = QComboBox()
                self.tts_model.setEditable(True)
                self.tts_model.addItems(DEFAULT_TTS_MODELS)
                self.tts_model.setCurrentText(settings.get("gemini_tts_model", DEFAULT_TTS_MODELS[0]))
                form.addRow("Voice model:", self.tts_model)
            vl.addWidget(box)
        self.fallback = QCheckBox("If a free limit runs out, switch to the next voice service automatically")
        self.fallback.setToolTip("Order: Gemini → ElevenLabs → Sarvam → Edge (only services with a key)")
        self.fallback.setChecked(bool(settings.get("auto_fallback", True)))
        vl.addWidget(self.fallback)
        vl.addStretch(1)
        tabs.addTab(vt, "🎙 Voices")

        # Auto-tag tab
        at = QWidget()
        al = QVBoxLayout(at)
        note = QLabel("Auto-tag tries the AI chosen next to the ✨ Auto-tag button first, then the others "
                      "below. If one fails or changes your words, the next one is used.")
        note.setWordWrap(True)
        al.addWidget(note)
        for svc in llm.SERVICES:
            box = QGroupBox(f"{svc.name}: {svc.free}")
            form = QFormLayout(box)
            if svc.id == "gemini":
                form.addRow(QLabel("Uses the Gemini key from the Voices tab."))
                self.model = QComboBox()
                self.model.setEditable(True)
                self.model.addItems(DEFAULT_MODELS)
                self.model.setCurrentText(settings.get("gemini_model", DEFAULT_MODELS[0]))
                form.addRow("Model:", self.model)
            else:
                row, link = _key_row(settings, svc.key_setting, svc.name, svc.key_url, svc.free, self.edits)
                form.addRow("API key:", row)
                form.addRow("", link)
            al.addWidget(box)
        al.addStretch(1)
        tabs.addTab(at, "✨ Auto-tag")

        test = QPushButton("Test keys")
        test.clicked.connect(self.test_keys)
        self.result = QLabel("")
        self.result.setWordWrap(True)
        trow = QHBoxLayout()
        trow.addWidget(test)
        trow.addWidget(self.result, 1)
        lay.addLayout(trow)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _key(self, setting):
        return config.clean_key(self.edits[setting].text())

    def test_keys(self):
        ok = lambda t: f"<span style='color:#2e9e5b'>✓ {t}</span>"
        bad = lambda t: f"<span style='color:#d64545'>✗ {t}</span>"
        lines = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for svc in llm.SERVICES:
                key = self._key(svc.key_setting)
                if not key:
                    continue
                try:
                    lines.append(ok(f"{svc.name}: {llm.test_key(svc.id, key)}"))
                except Exception as e:
                    lines.append(bad(f"{svc.name}: {e}"))
            key = self._key("elevenlabs_api_key")
            if key:
                try:
                    vs = providers.list_eleven_voices(key)
                    self.eleven_voices = [[v.id, v.name, v.gender, v.lang] for v in vs]
                    lines.append(ok(f"ElevenLabs: works, {len(vs)} voices"))
                except Exception as e:
                    lines.append(bad(str(e)))
            if self._key("sarvam_api_key"):
                lines.append("Sarvam: checked on first use")
        finally:
            QApplication.restoreOverrideCursor()
        self.result.setText("<br>".join(lines) or "Paste a key first.")

    def values(self) -> dict:
        out = {k: config.clean_key(e.text()) for k, e in self.edits.items()}
        out.update({"gemini_model": self.model.currentText().strip(),
                    "gemini_tts_model": self.tts_model.currentText().strip(),
                    "auto_fallback": self.fallback.isChecked()})
        if self.eleven_voices:
            out["eleven_voices"] = self.eleven_voices
        return out


# ------------------------------------------------------------------ main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text to Voice Maker")
        self.resize(1180, 760)
        self.settings = config.load_settings()
        providers.set_eleven_voices(self.settings.get("eleven_voices"))
        self._thread = None
        self._worker = None
        self._on_result = None
        self.cancel = threading.Event()
        self._preview_n = 0
        for old in config.CACHE.glob("preview_*.wav"):
            try:
                old.unlink(missing_ok=True)
            except OSError:   # still open in another window
                pass
        self._preview_n = int(datetime.now().timestamp())

        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)

        self._build_ui()
        self._load_prefs()
        self.refresh_history()

    # ---------------- UI
    def _build_ui(self):
        # toolbar row
        self.tag_ai = QComboBox()
        self.tag_ai.setToolTip("Which AI to try first. The others are used automatically if it fails.")
        self._fill_tag_ai()
        self.autotag_btn = QPushButton("✨ Auto-tag")
        self.autotag_btn.setToolTip("Add emotion tags to the text automatically")
        self.autotag_btn.clicked.connect(self.autotag)

        insert_btn = QToolButton()
        insert_btn.setText("Insert tag ▾")
        insert_btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(insert_btn)
        menu.addSection("Mood (lasts until the next mood)")
        for t in tags.STYLE_TAGS:
            menu.addAction(f"[{t}]", lambda t=t: self.insert_tag(t))
        menu.addSection("Events")
        for t in tags.EVENT_TAGS:
            menu.addAction(f"[{t}]", lambda t=t: self.insert_tag(t))
        insert_btn.setMenu(menu)

        remove_btn = QPushButton("Remove tags")
        remove_btn.clicked.connect(self.remove_tags)
        settings_btn = QPushButton("⚙ Settings")
        settings_btn.clicked.connect(self.open_settings)

        top = QHBoxLayout()
        top.addWidget(self.autotag_btn)
        top.addWidget(self.tag_ai)
        top.addWidget(insert_btn)
        top.addWidget(remove_btn)
        top.addStretch(1)
        top.addWidget(settings_btn)

        # editor
        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont("Nirmala UI", 13))
        self.editor.setPlaceholderText(
            "Type or paste your script here (English, বাংলা, or both)…\n\n"
            "Example:\n[excited] Welcome to my channel! Today we have something special.\n"
            "[calm] আজ আমরা একটি নতুন গল্প শুনব। [pause] চলুন শুরু করি।")
        self.highlighter = TagHighlighter(self.editor.document())
        self.char_lbl = QLabel("0 characters")
        self.editor.textChanged.connect(
            lambda: self.char_lbl.setText(f"{len(self.editor.toPlainText())} characters"))
        legend = QLabel("<span style='color:#2f7de1'><b>[blue]</b></span> = tag that will be applied   "
                        "<span style='color:#d64545'><s>[red]</s></span> = unknown tag, skipped")
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addLayout(top)
        ll.addWidget(self.editor, 1)
        foot = QHBoxLayout()
        foot.addWidget(legend)
        foot.addStretch(1)
        foot.addWidget(self.char_lbl)
        ll.addLayout(foot)

        # voice panel
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("Auto (English + বাংলা)", "auto")
        self.lang_combo.addItem("English only", "en")
        self.lang_combo.addItem("বাংলা only", "bn")

        self.voice = QComboBox()
        self.voice.setMaxVisibleItems(30)
        self._fill_voices()
        self.voice.setToolTip("Gemini and ElevenLabs act out the moods. Add free keys in ⚙ Settings.")
        self.preview_btn = QPushButton("▶ Preview")
        self.preview_btn.clicked.connect(self.preview_voice)
        vrow = QHBoxLayout()
        vrow.addWidget(self.voice, 1)
        vrow.addWidget(self.preview_btn)

        self.delivery = QComboBox()
        for key, (label, _) in DELIVERY.items():
            self.delivery.addItem(label, key)
        self.delivery.setToolTip("Overall narration character")

        self.speed = QSlider(Qt.Horizontal)
        self.speed.setRange(50, 200)
        self.speed_lbl = QLabel("1.00×")
        self.speed.valueChanged.connect(lambda v: self.speed_lbl.setText(f"{v / 100:.2f}×"))
        srow = QHBoxLayout()
        srow.addWidget(self.speed, 1)
        srow.addWidget(self.speed_lbl)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["MP3", "WAV"])

        form = QFormLayout()
        form.addRow("Language:", self.lang_combo)
        form.addRow("Voice:", vrow)
        form.addRow("Delivery:", self.delivery)
        form.addRow("Speed:", srow)
        form.addRow("Format:", self.format_combo)
        voice_box = QGroupBox("Voice")
        voice_box.setLayout(form)

        self.gen_btn = QPushButton("🎙 Generate voice")
        self.gen_btn.setMinimumHeight(44)
        self.gen_btn.setStyleSheet("QPushButton{font-size:15px;font-weight:bold;}")
        self.gen_btn.clicked.connect(self.generate)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_clicked)
        grow = QHBoxLayout()
        grow.addWidget(self.gen_btn, 1)
        grow.addWidget(self.cancel_btn)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)

        # player
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self.toggle_play)
        stop_btn = QPushButton("■")
        stop_btn.setFixedWidth(40)
        stop_btn.clicked.connect(self.player.stop)
        self.seek = QSlider(Qt.Horizontal)
        self.seek.sliderMoved.connect(self.player.setPosition)
        self.time_lbl = QLabel("0:00 / 0:00")
        self.player.positionChanged.connect(self._on_pos)
        self.player.durationChanged.connect(lambda d: self.seek.setRange(0, d))
        self.player.playbackStateChanged.connect(
            lambda s: self.play_btn.setText("❚❚" if s == QMediaPlayer.PlayingState else "▶"))
        self.now_lbl = QLabel("Nothing loaded")
        self.now_lbl.setStyleSheet("color:gray")
        prow = QHBoxLayout()
        prow.addWidget(self.play_btn)
        prow.addWidget(stop_btn)
        prow.addWidget(self.seek, 1)
        prow.addWidget(self.time_lbl)
        save_btn = QPushButton("💾 Save as…")
        save_btn.clicked.connect(self.save_as)
        player_box = QGroupBox("Player")
        pl = QVBoxLayout(player_box)
        pl.addWidget(self.now_lbl)
        pl.addLayout(prow)
        pl.addWidget(save_btn)

        # history
        self.history = QListWidget()
        self.history.itemDoubleClicked.connect(lambda it: self.play_file(Path(it.data(Qt.UserRole))))
        h_play = QPushButton("Play")
        h_play.clicked.connect(self._play_selected)
        h_open = QPushButton("Open folder")
        h_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(config.OUTPUT))))
        h_del = QPushButton("Delete")
        h_del.clicked.connect(self._delete_selected)
        hrow = QHBoxLayout()
        hrow.addWidget(h_play)
        hrow.addWidget(h_open)
        hrow.addWidget(h_del)
        hist_box = QGroupBox("History")
        hl = QVBoxLayout(hist_box)
        hl.addWidget(self.history, 1)
        hl.addLayout(hrow)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(voice_box)
        rl.addLayout(grow)
        rl.addWidget(self.progress)
        rl.addWidget(self.status)
        rl.addWidget(player_box)
        rl.addWidget(hist_box, 1)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([700, 440])

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addWidget(split)
        self.setCentralWidget(central)

    def _load_prefs(self):
        s = self.settings
        self._select(self.lang_combo, s.get("language"))
        self._fill_voices()
        self._select(self.voice, s.get("voice"))
        self.lang_combo.currentIndexChanged.connect(self._fill_voices)
        self._select(self.delivery, s.get("delivery", "documentary"))
        self.speed.setValue(int(float(s.get("speed", 1.0)) * 100))
        self.format_combo.setCurrentText(s.get("format", "MP3"))

    @staticmethod
    def _select(combo, data):
        i = combo.findData(data)
        if i >= 0:
            combo.setCurrentIndex(i)

    def _save_prefs(self):
        self.settings.update({
            "language": self.lang_combo.currentData(),
            "voice": self.voice.currentData(),
            "autotag_ai": self.tag_ai.currentData(),
            "delivery": self.delivery.currentData(),
            "speed": self.speed.value() / 100,
            "format": self.format_combo.currentText(),
        })
        config.save_settings(self.settings)

    # ---------------- background tasks
    def _run(self, fn, on_result, busy_text):
        if self._thread is not None:
            return
        self._on_result = on_result
        th = QThread(self)
        w = Worker(fn)
        w.moveToThread(th)
        th.started.connect(w.run)
        w.progress.connect(self._on_progress)
        w.done.connect(self._on_done)
        w.failed.connect(self._on_failed)
        w.done.connect(th.quit)
        w.failed.connect(th.quit)
        th.finished.connect(self._on_thread_finished)
        th.finished.connect(w.deleteLater)
        th.finished.connect(th.deleteLater)
        self._thread, self._worker = th, w
        net.set_cancel(self.cancel)
        self._set_busy(True, busy_text)
        th.start()

    def _set_busy(self, busy, text=""):
        for wdg in (self.gen_btn, self.autotag_btn, self.preview_btn):
            wdg.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.editor.setReadOnly(busy)
        if not busy:
            self.progress.setRange(0, 100)
        if busy:
            self.progress.setValue(0)
            self.status.setText(text)

    @Slot(int, str)
    def _on_progress(self, pct, msg):
        if pct < 0:                      # unknown length: moving "busy" bar
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(pct)
        if not self.cancel.is_set():
            self.status.setText(msg)

    def _cancel_clicked(self):
        self.cancel.set()
        self.cancel_btn.setEnabled(False)
        self.status.setText("Cancelling…")

    @Slot(object)
    def _on_done(self, result):
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        if self._on_result:
            self._on_result(result)

    @Slot(str)
    def _on_failed(self, msg):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        if msg == "__cancelled__":
            self.status.setText("Cancelled.")
            return
        self.status.setText("Error.")
        QMessageBox.critical(self, "Error", msg)

    @Slot()
    def _on_thread_finished(self):
        self._thread = self._worker = None
        net.set_cancel(None)
        self._set_busy(False)

    # ---------------- actions
    def insert_tag(self, t):
        self.editor.insertPlainText(f"[{t}] ")
        self.editor.setFocus()

    def _replace_text(self, new_text):
        cur = self.editor.textCursor()
        cur.beginEditBlock()
        cur.select(QTextCursor.Document)
        cur.insertText(new_text)
        cur.endEditBlock()   # one Ctrl+Z restores the old text

    def remove_tags(self):
        self._replace_text(tags.strip_tags(self.editor.toPlainText()))

    def open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec():
            self.settings.update(dlg.values())
            config.save_settings(self.settings)
            providers.set_eleven_voices(self.settings.get("eleven_voices"))
            self._fill_voices()
            self._fill_tag_ai()
            self.status.setText("Settings saved.")

    def _fill_tag_ai(self):
        current = self.tag_ai.currentData() or self.settings.get("autotag_ai", "gemini")
        self.tag_ai.clear()
        for svc in llm.SERVICES:
            has = bool(self.settings.get(svc.key_setting))
            self.tag_ai.addItem(svc.name + ("" if has else " (no key)"), svc.id)
        i = self.tag_ai.findData(current)
        self.tag_ai.setCurrentIndex(max(i, 0))

    def autotag(self):
        text = self.editor.toPlainText()
        if not text.strip():
            return
        if not any(self.settings.get(svc.key_setting) for svc in llm.SERVICES):
            if QMessageBox.question(self, "Free key needed",
                                    "Auto-tag needs a free AI key (Gemini, Groq, ...).\n"
                                    "Open Settings to add one?") == QMessageBox.Yes:
                self.open_settings()
            return
        settings = dict(self.settings)
        first = self.tag_ai.currentData()
        self.cancel = threading.Event()

        def job(progress):
            tagged, used, problems = llm.autotag(text, settings, first, notify=lambda m: progress(-1, m))
            note = f"Tagged by {used}. Press Ctrl+Z to undo."
            if problems:
                note += "\n⚠ " + "; ".join(problems)
            return tagged, note, False

        def on_result(res):
            tagged, note, warn = res
            self._replace_text(tagged)
            self.status.setText(note)

        self._run(job, on_result, "Auto-tagging…")

    def _fill_voices(self):
        current = self.voice.currentData() or self.settings.get("voice")
        self.voice.clear()
        keys = self._keys()
        lang = self.lang_combo.currentData() if hasattr(self, "lang_combo") else "auto"
        for prov in providers.PROVIDERS:
            voices = [v for v in prov.voices if lang == "auto" or v.lang in ("multi", lang)]
            if not voices:
                if prov.voices and lang == "bn":
                    self.voice.addItem(f"── {prov.name} · no Bangla voices on the free plan "
                                       "(its built-in voices have an English accent)")
                    self.voice.model().item(self.voice.count() - 1).setEnabled(False)
                continue
            ready = providers.has_key(prov, keys)
            self.voice.addItem(f"── {prov.name} · {prov.blurb}" + ("" if ready else "  (add free key in ⚙ Settings)"))
            self.voice.model().item(self.voice.count() - 1).setEnabled(False)
            for v in voices:
                extra = v.note or v.gender.title()
                self.voice.addItem(f"    {v.name}  ({extra})", f"{prov.id}:{v.id}")
        i = self.voice.findData(current)
        self.voice.setCurrentIndex(i if i >= 0 else 1)

    def _keys(self) -> dict:
        return {p.id: self.settings.get(p.key_setting, "") for p in providers.PROVIDERS if p.key_setting}

    def _options(self):
        return RenderOptions(language=self.lang_combo.currentData(),
                             voice=self.voice.currentData(),
                             speed=self.speed.value() / 100,
                             delivery=self.delivery.currentData(),
                             keys=self._keys(),
                             auto_fallback=bool(self.settings.get("auto_fallback", True)),
                             gemini_tts_model=self.settings.get("gemini_tts_model") or DEFAULT_TTS_MODELS[0])

    def generate(self):
        text = self.editor.toPlainText()
        if not tags.strip_tags(text).strip():
            QMessageBox.information(self, "Empty", "Type some text first.")
            return
        self._save_prefs()
        opts = self._options()
        ext = self.format_combo.currentText().lower()
        self.cancel = threading.Event()
        cancel = self.cancel
        path = config.OUTPUT / f"{datetime.now():%Y%m%d_%H%M%S}_{slug(text)}.{ext}"

        def job(progress):
            notes = []
            audio = render(text, opts, progress, cancel, notes)
            audio_io.save(path, audio, SR)
            return path, len(audio) / SR, notes

        def on_result(res):
            p, secs, notes = res
            self.status.setText(f"Done: {secs:.1f} s of audio → {p.name}" +
                                "".join(f"\n⚠ {n}" for n in notes))
            self.refresh_history(select=p)
            self.play_file(p)

        self._run(job, on_result, "Starting…")

    def preview_voice(self):
        opts = self._options()
        opts.auto_fallback = False          # let the user hear exactly this voice
        _, v = providers.lookup(opts.voice)
        lang = "en" if opts.language == "en" or v.lang == "en" else "bn"
        opts.language = lang
        name = self.voice.currentText()
        self._preview_n += 1
        path = config.CACHE / f"preview_{self._preview_n}.wav"
        self.cancel = threading.Event()
        cancel = self.cancel

        def job(progress):
            notes = []
            audio = render(PREVIEW_TEXT[lang], opts, progress, cancel, notes)
            audio_io.save(path, audio, SR)
            return path, notes

        def on_result(res):
            p, notes = res
            self.status.setText(f"Preview: {name}" + "".join(f"\n⚠ {n}" for n in notes))
            self.play_file(p, label=f"Preview: {name}")

        self._run(job, on_result, "Preparing preview…")

    # ---------------- player & history
    def play_file(self, path: Path, label=None):
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.current_file = path
        self.now_lbl.setText(label or path.name)
        self.player.play()

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        elif not self.player.source().isEmpty():
            self.player.play()

    def _on_pos(self, pos):
        if not self.seek.isSliderDown():
            self.seek.setValue(pos)
        self.time_lbl.setText(f"{fmt_time(pos)} / {fmt_time(self.player.duration())}")

    def save_as(self):
        src = getattr(self, "current_file", None)
        if not src or not Path(src).exists():
            QMessageBox.information(self, "Nothing to save", "Generate audio first.")
            return
        dst, _ = QFileDialog.getSaveFileName(self, "Save audio", str(Path.home() / src.name),
                                             "Audio (*.mp3 *.wav)")
        if dst:
            Path(dst).write_bytes(Path(src).read_bytes())
            self.status.setText(f"Saved to {dst}")

    def refresh_history(self, select: Path | None = None):
        self.history.clear()
        files = sorted([p for p in config.OUTPUT.iterdir() if p.suffix.lower() in (".mp3", ".wav")],
                       key=lambda p: p.stat().st_mtime, reverse=True)[:300]
        for p in files:
            it = QListWidgetItem(p.name)
            it.setData(Qt.UserRole, str(p))
            self.history.addItem(it)
            if select and p == select:
                self.history.setCurrentItem(it)

    def _play_selected(self):
        it = self.history.currentItem()
        if it:
            self.play_file(Path(it.data(Qt.UserRole)))

    def _delete_selected(self):
        it = self.history.currentItem()
        if not it:
            return
        p = Path(it.data(Qt.UserRole))
        if QMessageBox.question(self, "Delete", f"Delete {p.name}?") != QMessageBox.Yes:
            return
        if getattr(self, "current_file", None) == p:
            self.player.stop()
            self.player.setSource(QUrl())
        p.unlink(missing_ok=True)
        self.refresh_history()

    def closeEvent(self, e):
        self._save_prefs()
        self.cancel.set()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
        super().closeEvent(e)
