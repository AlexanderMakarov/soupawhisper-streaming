"""Menu bar / system tray status icon for SoupaWhisper."""

from __future__ import annotations

import io
import logging
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "tray"
POLL_INTERVAL_S = 0.2  # Icon / tooltip / menu label refresh only; actions are immediate
STATE_NAMES = ("idle", "loading", "recording", "transcribing", "error", "meeting")
# Widen every chip to this label so EN/RU center like AUTO.
BADGE_WIDTH_GUIDE = "AUTO"
BADGE_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/SFNS.ttf",
)

IS_MACOS = platform.system() == "Darwin"

# macOS menu bar geometry. pystray scales whatever image it is given to
# NSStatusBar.thickness() square and hands it to AppKit at 1x, which both fills the bar
# edge to edge and blurs on Retina. We compose the pixmap ourselves instead: the glyph is
# inset so it matches Apple's own icons, and the whole thing is drawn at the display scale.
#
# Everything below is a fraction of the measured bar thickness rather than a point size,
# because that thickness is not a constant: it has changed across macOS releases and can
# differ per display. Ratios keep the same proportions on whatever bar we are handed.
MACOS_GLYPH_RATIO = 13 / 22  # mic height when the language chip is stacked under it
MACOS_GLYPH_ONLY_RATIO = 16 / 22  # mic height when the bar shows the glyph alone
MACOS_EDGE_PADDING_RATIO = 1 / 22
MACOS_MIN_BADGE_FONT_PX = 6
# Used only when AppKit cannot be queried; 22pt @2x is the common case.
MACOS_FALLBACK_THICKNESS = 22.0
MACOS_FALLBACK_SCALE = 2
# Template images are drawn from alpha alone, so macOS tints them for the current menu bar.
# Only the monochrome states qualify: recording/transcribing/error carry meaning in colour.
MACOS_TEMPLATE_STATES = frozenset({"idle", "loading"})
# Non-template states are composited over whatever the wallpaper puts behind the menu bar,
# so the chip borrows the glyph's colour: readable on a light and a dark bar alike.
MACOS_BADGE_RED = (232, 40, 44, 255)
MACOS_BADGE_COLORS = {
    "recording": MACOS_BADGE_RED,
    "error": MACOS_BADGE_RED,
    "transcribing": (196, 132, 0, 255),
}


class TrayStartError(Exception):
    """Tray requested but could not be started."""


def derive_state(dictation: Any) -> str:
    """Map dictation flags to a tray state name.

    Priority: error > loading > meeting > recording > transcribing > idle

    Meeting outranks recording: while a meeting is being captured the dictation
    hotkey is disabled, so "recording" would be misleading, and the meeting icon
    must stay visible for the whole call as the live-capture indicator.
    """
    if getattr(dictation, "model_error", None) or getattr(dictation, "_tray_error", None):
        return "error"
    if not getattr(dictation, "model_loaded", None) or not dictation.model_loaded.is_set():
        return "loading"
    if getattr(dictation, "meeting_active", False):
        return "meeting"
    if getattr(dictation, "recording", False):
        return "recording"
    if getattr(dictation, "stopping", False) or getattr(dictation, "transcribing", False):
        return "transcribing"
    return "idle"


def config_flag(dictation: Any, key: str, default: bool = True) -> bool:
    config = getattr(dictation, "config", None) or {}
    if isinstance(config, dict):
        return bool(config.get(key, default))
    return default


def language_label(dictation: Any) -> tuple[str, bool]:
    """Return (label, from_layout) for the language badge / menu line."""
    import meeting

    if meeting.is_english_only_model(getattr(dictation, "model", None)):
        return "EN", False
    session = getattr(dictation, "_session_enforced_language", None)
    if session:
        return session.upper(), True
    configured = None
    config = getattr(dictation, "config", None) or {}
    if isinstance(config, dict):
        configured = config.get("language")
    if configured:
        return str(configured).upper(), False
    return "AUTO", False


def tooltip_for(dictation: Any, state: str) -> str:
    hotkey = "F12"
    get_name = getattr(dictation, "get_hotkey_name", None)
    if callable(get_name):
        hotkey = (get_name() or "F12").upper()
    config = getattr(dictation, "config", None) or {}
    model = config.get("model", "model") if isinstance(config, dict) else "model"
    model = getattr(dictation, "dictation_label", None) or model
    err = getattr(dictation, "model_error", None) or getattr(dictation, "_tray_error", None)

    if state == "loading":
        return f"SoupaWhisper - loading {model}..."
    if state == "recording":
        return f"Recording - press {hotkey} to stop"
    if state == "transcribing":
        return "Transcribing..."
    if state == "error":
        return str(err) if err else "SoupaWhisper - error"
    return f"SoupaWhisper - idle ({hotkey} to record)"


def latin1_safe(text: str) -> str:
    """AppIndicator/DBus titles are latin-1; drop non-encodable chars."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def badge_font_size(icon_height: int) -> int:
    """Point size tuned for downscaled tray panels (regular weight, not bold)."""
    return max(10, int(round((icon_height / 4) * 1.7)) - 3)


def load_badge_font(size: int) -> ImageFont.ImageFont:
    for path in BADGE_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose_language_badge(base: Image.Image, badge: str) -> Image.Image:
    """Draw a language chip bottom-centered on the icon (XFCE often ignores AppIndicator labels)."""
    if not badge:
        return base
    img = base.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = load_badge_font(badge_font_size(h))
    guide_bbox = draw.textbbox((0, 0), BADGE_WIDTH_GUIDE, font=font)
    text_bbox = draw.textbbox((0, 0), badge, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    chip_tw = max(tw, guide_bbox[2] - guide_bbox[0])
    chip_th = max(th, guide_bbox[3] - guide_bbox[1])
    pad_x = max(2, h // 20)
    pad_y = max(1, h // 28)
    chip_w = min(w, chip_tw + 2 * pad_x)
    chip_h = min(h, chip_th + 2 * pad_y)
    x0 = (w - chip_w) // 2
    y0 = h - chip_h
    draw.rectangle((x0, y0, x0 + chip_w - 1, y0 + chip_h - 1), fill=(20, 20, 20, 230))
    tx = x0 + (chip_w - tw) // 2 - text_bbox[0]
    ty = y0 + (chip_h - th) // 2 - text_bbox[1]
    draw.text((tx, ty), badge, fill=(255, 255, 255, 255), font=font)
    return img


def fit_badge_font(text: str, max_height: int) -> ImageFont.ImageFont:
    """Largest candidate font whose text fits the chip band."""
    size = max(MACOS_MIN_BADGE_FONT_PX, max_height)
    while size > MACOS_MIN_BADGE_FONT_PX:
        font = load_badge_font(size)
        bbox = font.getbbox(text)
        if bbox[3] - bbox[1] <= max_height:
            return font
        size -= 1
    return load_badge_font(MACOS_MIN_BADGE_FONT_PX)


def macos_menu_bar_image(
    base: Image.Image,
    badge: str,
    state: str,
    thickness: float = MACOS_FALLBACK_THICKNESS,
    scale: int = MACOS_FALLBACK_SCALE,
) -> Image.Image:
    """Compose the menu bar pixmap: inset glyph, language chip stacked under it.

    Rendered at ``scale`` times the point size so AppKit can draw it crisply on Retina.
    Every label is laid out on the ``BADGE_WIDTH_GUIDE`` box, so EN/RU/AUTO share one
    width and the icon never shifts as the session language changes.
    """
    height = int(round(thickness * scale))
    pad = max(1, int(round(height * MACOS_EDGE_PADDING_RATIO)))
    if badge:
        glyph_px = int(round(height * MACOS_GLYPH_RATIO))
        badge_px = max(1, height - 2 * pad - glyph_px)
    else:
        glyph_px = min(int(round(height * MACOS_GLYPH_ONLY_RATIO)), height - 2 * pad)
        badge_px = 0

    glyph = base.convert("RGBA").resize((glyph_px, glyph_px), Image.LANCZOS)

    font = None
    guide_bbox = (0, 0, 0, 0)
    if badge_px:
        font = fit_badge_font(BADGE_WIDTH_GUIDE, badge_px)
        guide_bbox = font.getbbox(BADGE_WIDTH_GUIDE)
    width = max(glyph_px, guide_bbox[2] - guide_bbox[0] + 2 * pad)

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glyph_y = pad if badge_px else (height - glyph_px) // 2
    img.alpha_composite(glyph, ((width - glyph_px) // 2, glyph_y))
    if not badge_px or font is None:
        return img

    draw = ImageDraw.Draw(img)
    text_bbox = draw.textbbox((0, 0), badge, font=font)
    tx = (width - (text_bbox[2] - text_bbox[0])) // 2 - text_bbox[0]
    band_top = glyph_y + glyph_px
    ty = band_top + (badge_px - (text_bbox[3] - text_bbox[1])) // 2 - text_bbox[1]
    # Template states keep plain black ink; macOS repaints it in the menu bar's own colour.
    fill = (0, 0, 0, 255)
    if state not in MACOS_TEMPLATE_STATES:
        fill = MACOS_BADGE_COLORS.get(state, (232, 40, 44, 255))
    draw.text((tx, ty), badge, fill=fill, font=font)
    return img


def _macos_bar_metrics() -> tuple[float, int]:
    """(menu bar thickness in points, display scale). Falls back to the common 22pt @2x."""
    try:
        import AppKit

        thickness = float(AppKit.NSStatusBar.systemStatusBar().thickness())
        screen = AppKit.NSScreen.mainScreen()
        scale = int(round(float(screen.backingScaleFactor()))) if screen else MACOS_FALLBACK_SCALE
        return thickness, max(1, scale)
    except Exception as e:
        logger.debug("Could not read menu bar metrics: %s", e)
        return MACOS_FALLBACK_THICKNESS, MACOS_FALLBACK_SCALE


def _ns_image_from(img: Image.Image, *, point_size: tuple[float, float], template: bool):
    """Wrap a PIL image in an NSImage sized in points, so the extra pixels read as Retina."""
    import io

    import AppKit
    import Foundation

    buf = io.BytesIO()
    img.save(buf, "png")
    ns_image = AppKit.NSImage.alloc().initWithData_(Foundation.NSData(buf.getvalue()))
    ns_image.setSize_(AppKit.NSMakeSize(*point_size))
    ns_image.setTemplate_(template)
    return ns_image


def load_state_images(assets_dir: Path = ASSETS_DIR) -> dict[str, Image.Image]:
    """Load 64×64 state PNGs (required). Optionally keep 16px variants aside."""
    images: dict[str, Image.Image] = {}
    missing = []
    for name in STATE_NAMES:
        path = assets_dir / f"{name}.png"
        if not path.is_file():
            missing.append(str(path))
            continue
        images[name] = Image.open(path).convert("RGBA")
    if missing:
        raise TrayStartError(f"Missing tray assets: {', '.join(missing)}")
    return images


def _open_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if IS_MACOS:
        subprocess.Popen(["open", str(path)], start_new_session=True)
    else:
        subprocess.Popen(["xdg-open", str(path)], start_new_session=True)


def _open_linux_journal() -> bool:
    terminal_cmds = [
        ["xdg-terminal-exec", "journalctl", "--user", "-u", "soupawhisper", "-e"],
        ["x-terminal-emulator", "-e", "journalctl --user -u soupawhisper -e"],
    ]
    for cmd in terminal_cmds:
        try:
            subprocess.Popen(cmd, start_new_session=True)
            return True
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.debug("journalctl terminal launch failed (%s): %s", cmd[0], e)
    return False


class TrayStatus:
    """Owns the pystray main-thread loop.

    Menu actions (Start/Stop, Quit, …) run immediately via callbacks.
    A short poll only refreshes icon, tooltip, and dynamic menu labels.
    """

    def __init__(self, dictation: Any, assets_dir: Path = ASSETS_DIR):
        self.dictation = dictation
        self.assets_dir = assets_dir
        self._images = load_state_images(assets_dir)
        self._images_16: dict[str, Image.Image] = {}
        for name in STATE_NAMES:
            p16 = assets_dir / f"{name}-16.png"
            if p16.is_file():
                self._images_16[name] = Image.open(p16).convert("RGBA")
        self._icon = None
        self._last_state: Optional[str] = None
        self._last_title: Optional[str] = None
        self._last_menu_signature: Optional[tuple] = None
        self._last_badge: Optional[str] = None
        self._stop_poll = threading.Event()

    def prepare(self) -> None:
        """Import pystray / create Icon (does not block). Raises TrayStartError."""
        try:
            import pystray
            from pystray import Menu, MenuItem
        except Exception as e:
            raise TrayStartError(f"Failed to import pystray: {e}") from e

        state = derive_state(self.dictation)
        badge = self._language_badge_text()
        image = self._image_for(state, badge)
        title = self._title_for(state)

        menu = self._build_menu(Menu, MenuItem)

        try:
            self._icon = pystray.Icon(
                "soupawhisper",
                icon=image,
                title=title,
                menu=menu,
            )
        except Exception as e:
            raise TrayStartError(f"Failed to create system tray icon: {e}") from e

        self._last_state = state
        self._last_title = title
        self._last_badge = badge

    def run_blocking(self) -> None:
        """Block on the tray event loop (main thread). Call prepare() first."""
        if self._icon is None:
            self.prepare()
        assert self._icon is not None
        try:
            self._icon.run(setup=self._setup)
        except TrayStartError:
            raise
        except Exception as e:
            raise TrayStartError(f"Tray icon failed: {e}") from e

    def stop(self) -> None:
        self._stop_poll.set()
        icon = self._icon
        if icon is not None:
            try:
                icon.stop()
            except Exception as e:
                logger.debug("tray icon.stop() failed: %s", e)

    def _setup(self, icon) -> None:
        icon.visible = True
        # pystray has just installed its own squashed 1x pixmap; replace it before the
        # first frame the user sees, then keep it in step from _refresh().
        if IS_MACOS:
            self._install_macos_image(self._last_state, self._last_badge)
        try:
            while not self._stop_poll.is_set() and getattr(self.dictation, "running", True):
                self._refresh(icon)
                time.sleep(POLL_INTERVAL_S)
        finally:
            try:
                icon.stop()
            except Exception:
                pass

    def _refresh(self, icon) -> None:
        state = derive_state(self.dictation)
        title = self._title_for(state)
        badge = self._language_badge_text()
        if state != self._last_state or badge != self._last_badge:
            if IS_MACOS:
                self._install_macos_image(state, badge)
            else:
                icon.icon = self._image_for(state, badge)
                self._apply_language_badge(badge)
            self._last_state = state
            self._last_badge = badge
        if title != self._last_title:
            icon.title = title
            self._last_title = title
        # Only rebuild the menu when labels/enablement change. Calling
        # update_menu() every poll closes an open menu on AppIndicator/XFCE.
        sig = self._menu_signature()
        if sig != self._last_menu_signature:
            try:
                icon.update_menu()
            except Exception:
                pass
            self._last_menu_signature = sig

    def _image_for(self, state: str, badge: str = "") -> Image.Image:
        base = self._images[state]
        if badge:
            return compose_language_badge(base, badge)
        return base

    def _language_badge_text(self) -> str:
        if not config_flag(self.dictation, "tray_show_language", True):
            return ""
        label, _ = language_label(self.dictation)
        return latin1_safe(label)

    def _install_macos_image(self, state: str, badge: str) -> None:
        """Draw the menu bar pixmap ourselves — see macos_menu_bar_image() for why."""
        icon = self._icon
        status_item = getattr(icon, "_status_item", None) if icon is not None else None
        if status_item is None:
            return
        thickness, scale = _macos_bar_metrics()
        img = macos_menu_bar_image(self._images[state], badge, state, thickness, scale)
        try:
            ns_image = _ns_image_from(
                img,
                point_size=(img.width / scale, img.height / scale),
                template=state in MACOS_TEMPLATE_STATES,
            )
            button = status_item.button()
            button.setImage_(ns_image)
            # The language is baked into the pixmap; a button title would repeat it.
            button.setTitle_("")
        except Exception as e:
            logger.debug("Could not install macOS menu bar image: %s", e)

    def _apply_language_badge(self, badge: str) -> None:
        """Also set the AppIndicator label when available (icon pixmap is primary)."""
        icon = self._icon
        if icon is None:
            return
        appind = getattr(icon, "_appindicator", None)
        if appind is not None:
            guide = "WWW" if badge else ""

            def apply():
                try:
                    appind.set_label(badge, guide)
                except Exception as e:
                    logger.debug("set_label failed: %s", e)
                return False

            try:
                from gi.repository import GLib

                GLib.idle_add(apply)
            except Exception as e:
                logger.debug("Could not schedule set_label: %s", e)

    def _build_menu(self, Menu, MenuItem):
        return Menu(
            MenuItem(lambda item: self._state_menu_text(), None, enabled=False),
            *self._model_rows(Menu, MenuItem),
            MenuItem(lambda item: self._language_menu_text(), None, enabled=False),
            Menu.SEPARATOR,
            MenuItem(
                self._toggle_menu_text,
                self._on_toggle_dictation,
                enabled=self._toggle_enabled,
            ),
            MenuItem(
                self._meeting_menu_text,
                self._on_toggle_meeting,
                enabled=self._meeting_enabled,
            ),
            Menu.SEPARATOR,
            MenuItem("Restart hotkey listener", self._on_restart_listener),
            MenuItem(
                lambda item: self._meeting_disk_text(),
                Menu(
                    MenuItem("Delete recorded audio files", self._on_purge_meeting_audio),
                    MenuItem("Open recorded audio folder", self._on_open_meeting_audio),
                ),
            ),
            MenuItem("Open log", self._on_open_log),
            MenuItem("Open config", self._on_open_config),
            MenuItem("Quit", self._on_quit),
        )

    def _model_rows(self, Menu, MenuItem) -> list:
        """Model submenus when [behavior] models is set, else the plain status row."""
        if not self._model_choices():
            return [MenuItem(lambda item: self._model_menu_text(), None, enabled=False)]
        return [
            MenuItem(
                lambda item: self._dictation_model_text(),
                self._model_submenu(Menu, MenuItem, "dictation_label", "select_dictation_model"),
                enabled=lambda item: self._model_switch_enabled(),
            ),
            MenuItem(
                lambda item: self._meeting_model_text(),
                self._model_submenu(Menu, MenuItem, "meeting_label", "select_meeting_model"),
                enabled=lambda item: self._model_switch_enabled(),
            ),
        ]

    def _model_submenu(self, Menu, MenuItem, active_attr: str, select_method: str):
        return Menu(*(
            self._model_item(MenuItem, choice.label, active_attr, select_method)
            for choice in self._model_choices()
        ))

    def _model_item(self, MenuItem, label: str, active_attr: str, select_method: str):
        # A named closure, because pystray rejects actions with more than two
        # parameters, which rules out binding `label` as a lambda default.
        def on_select(icon, item) -> None:
            getattr(self.dictation, select_method)(label)

        def is_active(item) -> bool:
            return getattr(self.dictation, active_attr) == label

        return MenuItem(label, on_select, checked=is_active, radio=True)

    def _model_choices(self) -> list:
        return getattr(self.dictation, "model_choices", None) or []

    def _dictation_model_text(self) -> str:
        label = self.dictation.dictation_label
        if not self.dictation.model_loaded.is_set():
            return f"Dictation model: loading {label}…"
        return f"Dictation model: {label}"

    def _meeting_model_text(self) -> str:
        downloading = self.dictation.meeting_download_label
        if downloading:
            return f"Meeting model: downloading {downloading}…"
        return f"Meeting model: {self.dictation.meeting_label}"

    def _model_switch_enabled(self) -> bool:
        return self.dictation.can_switch_model()

    def _model_rows_signature(self) -> tuple:
        if not self._model_choices():
            return ()
        return (
            self._dictation_model_text(),
            self._meeting_model_text(),
            self._model_switch_enabled(),
        )

    def _meeting_rows_signature(self) -> tuple:
        """The meeting phase only: a rebuild per percent would keep closing an open menu."""
        import meeting

        progress = getattr(self.dictation, "meeting_progress", None)
        return (
            self._meeting_enabled(),
            getattr(self.dictation, "meeting_active", False),
            progress == meeting.PROGRESS_LOADING_MODEL,
        )

    def _menu_signature(self) -> tuple:
        """Stable snapshot of dynamic menu fields (avoid needless rebuilds)."""
        return (
            *self._model_rows_signature(),
            *self._meeting_rows_signature(),
            derive_state(self.dictation),
            self._state_menu_text(),
            self._model_menu_text(),
            self._language_menu_text(),
            self._toggle_menu_text(),
            self._toggle_enabled(),
        )

    def _title_for(self, state: str) -> str:
        tip = tooltip_for(self.dictation, state)
        if not config_flag(self.dictation, "tray_show_language", True):
            return latin1_safe(tip)
        label, _ = language_label(self.dictation)
        return latin1_safe(f"{tip} [{label}]")

    def _state_menu_text(self) -> str:
        state = derive_state(self.dictation)
        return f"● {state.capitalize()}"

    def _model_menu_text(self) -> str:
        config = getattr(self.dictation, "config", None) or {}
        model = config.get("model", "?") if isinstance(config, dict) else "?"
        return f"Model: {model}"

    def _language_menu_text(self) -> str:
        label, from_layout = language_label(self.dictation)
        if from_layout:
            return f"Language: {label} (from layout)"
        return f"Language: {label}"

    def _toggle_menu_text(self, item=None) -> str:
        d = self.dictation
        if (
            getattr(d, "recording", False)
            or getattr(d, "stopping", False)
            or getattr(d, "transcribing", False)
        ):
            return "Stop dictation"
        return "Start dictation"

    def _toggle_enabled(self, item=None) -> bool:
        state = derive_state(self.dictation)
        if state == "loading":
            return False
        # Allow Stop while busy; block Start when stuck in error and not recording
        if state == "error" and not getattr(self.dictation, "recording", False):
            return False
        if getattr(self.dictation, "stopping", False) or getattr(self.dictation, "transcribing", False):
            # Stop already in progress — leave item visible but inactive
            return False
        return True

    def _on_toggle_dictation(self, icon, item) -> None:
        """Same immediate path as the hotkey (not gated on the icon poll)."""
        d = self.dictation
        schedule = getattr(d, "_schedule_hotkey_action", None)
        if getattr(d, "recording", False):
            if callable(schedule):
                schedule(d.stop_recording, "stop_recording")
            else:
                d.stop_recording()
        else:
            if callable(schedule):
                schedule(d.start_recording, "start_recording")
            else:
                d.start_recording()
        # Best-effort snappy visual update; poll will catch up if this races
        try:
            self._refresh(icon)
        except Exception:
            pass

    def _meeting_tmp_root(self) -> Path:
        import meeting

        return Path(getattr(self.dictation, "_meeting_tmp_root", meeting.TMP_ROOT))

    def _meeting_menu_text(self, item=None) -> str:
        if getattr(self.dictation, "meeting_transcribing", False):
            progress = getattr(self.dictation, "meeting_progress", None)
            return f"Transcribing meeting\u2026 {progress}" if progress else "Transcribing meeting\u2026"
        active = getattr(self.dictation, "meeting_active", False)
        return "Stop meeting recording" if active else "Start meeting recording"

    def _meeting_enabled(self, item=None) -> bool:
        """Disabled only while transcribing -- a second meeting cannot start until
        the previous one has been written out."""
        return not getattr(self.dictation, "meeting_transcribing", False)

    def _meeting_disk_text(self) -> str:
        """Scratch-audio footprint, so a forgotten recording is visible in the menu."""
        import meeting

        usage = meeting.wav_usage(self._meeting_tmp_root())
        if usage.count == 0:
            return "Meeting audio: none"
        plural = "" if usage.count == 1 else "s"
        return f"Meeting audio: {usage.human} ({usage.count} file{plural})"

    def _on_toggle_meeting(self, icon, item) -> None:
        try:
            self.dictation.toggle_meeting()
        except Exception as e:
            logger.error("Meeting toggle failed: %s", e, exc_info=True)

    def _on_purge_meeting_audio(self, icon, item) -> None:
        """Delete scratch WAVs, sparing a meeting that is still recording."""
        import meeting

        recorder = getattr(self.dictation, "meeting", None)
        keep = recorder.session_dir if getattr(self.dictation, "meeting_active", False) and recorder else None
        try:
            freed = meeting.purge_wavs(self._meeting_tmp_root(), keep=keep)
            logger.info("Deleted meeting audio, freed %.0f MB", freed / 1e6)
        except Exception as e:
            logger.error("Could not delete meeting audio: %s", e, exc_info=True)
        # The size is too costly to poll, so ask for a rebuild to show the new one.
        self._last_menu_signature = None

    def _on_open_meeting_audio(self, icon, item) -> None:
        root = self._meeting_tmp_root()
        try:
            root.mkdir(parents=True, exist_ok=True)
            _open_path(root)
        except Exception as e:
            logger.error("Could not open %s: %s", root, e, exc_info=True)

    def _on_restart_listener(self, icon, item) -> None:
        fn: Optional[Callable[[], None]] = getattr(self.dictation, "restart_hotkey_listener", None)
        if not callable(fn):
            logger.warning("restart_hotkey_listener is not available")
            return
        try:
            fn()
        except Exception as e:
            logger.error("Failed to restart hotkey listener: %s", e, exc_info=True)
            self.dictation._tray_error = str(e)

    def _on_open_log(self, icon, item) -> None:
        if IS_MACOS:
            log_path = Path.home() / "Library" / "Logs" / "soupawhisper.log"
            try:
                _open_path(log_path)
                return
            except FileNotFoundError:
                logger.warning("Log file not found at %s (foreground runs log to stderr)", log_path)
                return
        if _open_linux_journal():
            return
        logger.warning(
            "Could not open journalctl for soupawhisper; foreground runs log to stderr. "
            "Try: journalctl --user -u soupawhisper -f"
        )

    def _on_open_config(self, icon, item) -> None:
        # Prefer path from dictate module when available
        try:
            import dictate as dictate_mod

            path = Path(dictate_mod.CONFIG_PATH)
        except Exception:
            path = Path.home() / ".config" / "soupawhisper" / "config.ini"
        try:
            _open_path(path)
        except FileNotFoundError:
            logger.warning(
                "Config not found at %s — copy config.example.ini there first",
                path,
            )

    def _on_quit(self, icon, item) -> None:
        try:
            self.dictation.stop()
        finally:
            self.stop()
