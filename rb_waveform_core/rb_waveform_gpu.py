# Copyright (c) mrmilbe

"""GPU-accelerated waveform display in a separate window (pygame / SDL2).

Why this exists:
    The tkinter display path re-resized and re-uploaded the full waveform image to
    a software-rendered Tk PhotoImage every frame (per deck), which dominated CPU.
    Here the full-track waveform is uploaded to the GPU once as a Texture; per-frame
    scrolling/zooming is just a GPU draw of a sub-rectangle into the deck viewport -
    the "move a surface" model, like the OS compositor moving a window.

Position in data flow:
    deck_controller (analysis) -> render.prerender_full_waveform (PIL image, once per
    track/config) -> set_track() uploads it to a GPU Texture -> render_frame() draws
    the visible time window each frame (cheap).

Threading note:
    The Window/Renderer/Textures are created and used from a single thread (the same
    thread that drives render_frame, i.e. the tkinter main thread via root.after). SDL
    rendering must stay on the thread that created the renderer - do not call into this
    from a worker thread.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# Linear texture filtering (read by SDL when textures are created / scaled). This
# is what makes the scaled waveform smooth instead of nearest-neighbour "pixely".
os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "1")

import pygame
from pygame._sdl2.video import Renderer, Texture, Window
from PIL import Image


@dataclass
class _DeckTexture:
    texture: Texture
    pixels_per_second: float
    total_duration: float
    downbeats: tuple = ()   # downbeat times in seconds (drawn as overlay lines)
    cues: tuple = ()        # (time_sec, (r,g,b), is_memory) cue markers


@dataclass
class DeckFrame:
    """Per-deck state needed to draw one frame."""
    current_time: Optional[float]   # playhead time in seconds, or None if no position yet
    visible_seconds: float          # width of the visible window (zoom x tempo)
    label: str = ""


class WaveformGpuWindow:
    """A standalone SDL2 window that draws one scrolling waveform per deck on the GPU."""

    def __init__(
        self,
        deck_count: int,
        size: Tuple[int, int] = (1280, 400),
        *,
        background: Tuple[int, int, int] = (5, 5, 5),
        playhead_color: Tuple[int, int, int] = (255, 0, 0),
        beat_color: Tuple[int, int, int] = (235, 235, 235),
        beat_end_color: Tuple[int, int, int] = (255, 60, 60),
        beat_width: float = 2.0,
        beat_triangles: bool = False,
        vsync: bool = False,
        borderless: bool = True,
        always_on_top: bool = True,
        on_close=None,
    ) -> None:
        self.deck_count = max(1, int(deck_count))
        self.background = pygame.Color(*background)
        self.playhead_color = pygame.Color(*playhead_color)
        self.beat_color = pygame.Color(*beat_color)
        self.beat_end_color = pygame.Color(*beat_end_color)
        self._beat_width = float(beat_width)
        self._beat_triangles = bool(beat_triangles)
        self.on_close = on_close
        self._closed = False
        self._title = "rkbx_wave — Waveforms"
        self._hwnd: Optional[int] = None
        self._on_top = bool(always_on_top)

        pygame.init()

        self.window = self._create_window(size, borderless)
        # accelerated=1 forces a hardware renderer; fall back to auto if unavailable.
        try:
            self.renderer = Renderer(self.window, accelerated=1, vsync=vsync)
            self.hardware = True
        except Exception as exc:  # pragma: no cover - depends on host GPU/drivers
            print(f"[GPU] Hardware renderer unavailable ({exc}); using auto renderer")
            self.renderer = Renderer(self.window, accelerated=-1, vsync=vsync)
            self.hardware = False

        # Conservative max texture width; long tracks reduce pixels/second to fit.
        self.max_texture_width = 8192

        try:
            pygame.font.init()
            self._font = pygame.font.SysFont("Arial", 14)
            self._cue_font = pygame.font.SysFont("Arial", 11, bold=True)
        except Exception:
            self._font = None
            self._cue_font = None
        self._label_cache: dict[str, Texture] = {}
        self._cue_label_cache: dict = {}  # (letter, fg) -> Texture

        self._beat_tex = self._make_beat_texture()
        self._tri_top, self._tri_bottom = self._make_triangle_textures()
        self._decks: dict[int, Optional[_DeckTexture]] = {i: None for i in range(self.deck_count)}

        # Apply always-on-top via Win32 (the SDL flag is unreliable on Windows).
        self.set_always_on_top(self._on_top)

    def _create_window(self, size, borderless: bool) -> Window:
        """Create a resizable SDL window (borderless optional). Border/size/position
        are all runtime-toggleable so the user can switch between a clean overlay
        and a normal movable/resizable window without recreating anything."""
        title = "rkbx_wave — Waveforms"
        try:
            return Window(title=title, size=size, resizable=True, borderless=borderless)
        except TypeError:
            return Window(title=title, size=size, resizable=True)

    # ------------------------------------------------------------------
    # Window controls (border / always-on-top) — driven from the tk GUI
    # ------------------------------------------------------------------
    def _user32(self):
        u = ctypes.windll.user32
        from ctypes import wintypes
        u.FindWindowW.restype = wintypes.HWND
        u.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        u.SetWindowPos.restype = wintypes.BOOL
        u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        return u

    def _get_hwnd(self) -> Optional[int]:
        if self._hwnd:
            return self._hwnd
        try:
            self._hwnd = self._user32().FindWindowW(None, self._title) or None
        except Exception:
            self._hwnd = None
        return self._hwnd

    def set_always_on_top(self, enabled: bool) -> None:
        """Make the waveform window topmost (or not) via Win32 SetWindowPos."""
        self._on_top = bool(enabled)
        try:
            hwnd = self._get_hwnd()
            if not hwnd:
                return
            HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
            SWP_NOMOVE, SWP_NOSIZE, SWP_NOACTIVATE = 0x0002, 0x0001, 0x0010
            self._user32().SetWindowPos(
                hwnd, HWND_TOPMOST if enabled else HWND_NOTOPMOST,
                0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            )
        except Exception:
            pass

    def set_borderless(self, borderless: bool) -> None:
        """Toggle the window border live. Bordered = native move/resize; borderless
        = clean overlay. Re-assert topmost afterwards (toggling can drop it)."""
        try:
            self.window.borderless = bool(borderless)
        except Exception:
            pass
        if self._on_top:
            self.set_always_on_top(True)

    def reassert_topmost(self) -> None:
        """Periodically re-apply topmost so it can't be stolen by other windows."""
        if self._on_top:
            self.set_always_on_top(True)

    def _make_beat_texture(self) -> Optional[Texture]:
        """A 3px-wide soft vertical line (alpha falloff at the edges). Drawn at a
        float x with linear filtering it renders as a thin, anti-aliased beat
        marker that slides sub-pixel-smoothly instead of snapping to whole pixels."""
        try:
            c = self.beat_color
            surf = pygame.Surface((3, 1), pygame.SRCALPHA)
            surf.set_at((0, 0), (c.r, c.g, c.b, 70))
            surf.set_at((1, 0), (c.r, c.g, c.b, 255))
            surf.set_at((2, 0), (c.r, c.g, c.b, 70))
            tex = Texture.from_surface(self.renderer, surf)
            tex.blend_mode = pygame.BLENDMODE_BLEND
            return tex
        except Exception:
            return None

    def _make_triangle_textures(self):
        """Two small triangles in the end-marker colour: one pointing down (drawn at
        the top of the beat line) and one pointing up (drawn at the bottom), like the
        red markers Rekordbox draws at the ends of each beat line."""
        try:
            c = self.beat_end_color
            w, h = 9, 6
            top = pygame.Surface((w, h), pygame.SRCALPHA)      # apex pointing down
            pygame.draw.polygon(top, (c.r, c.g, c.b, 255), [(0, 0), (w - 1, 0), ((w - 1) / 2, h - 1)])
            bottom = pygame.Surface((w, h), pygame.SRCALPHA)   # apex pointing up
            pygame.draw.polygon(bottom, (c.r, c.g, c.b, 255), [(0, h - 1), (w - 1, h - 1), ((w - 1) / 2, 0)])
            tt = Texture.from_surface(self.renderer, top)
            bt = Texture.from_surface(self.renderer, bottom)
            tt.blend_mode = pygame.BLENDMODE_BLEND
            bt.blend_mode = pygame.BLENDMODE_BLEND
            return tt, bt
        except Exception:
            return None, None

    def set_beat_style(
        self,
        color: Optional[Tuple[int, int, int]] = None,
        end_color: Optional[Tuple[int, int, int]] = None,
        width: Optional[float] = None,
        triangles: Optional[bool] = None,
    ) -> None:
        """Update beat line colour / end-marker colour / width / markers live (cheap)."""
        if color is not None:
            self.beat_color = pygame.Color(*color)
            self._beat_tex = self._make_beat_texture()
        if end_color is not None:
            self.beat_end_color = pygame.Color(*end_color)
            self._tri_top, self._tri_bottom = self._make_triangle_textures()
        if width is not None:
            self._beat_width = max(0.5, float(width))
        if triangles is not None:
            self._beat_triangles = bool(triangles)

    # ------------------------------------------------------------------
    # Texture management (called on track load / config change)
    # ------------------------------------------------------------------
    def choose_pixels_per_second(self, total_duration: float, base: float = 50.0) -> float:
        """Pick a texture density that keeps the whole-track texture within GPU limits."""
        total_duration = max(float(total_duration), 1e-3)
        max_pps = self.max_texture_width / total_duration
        return max(1.0, min(base, max_pps))

    def clamp_pixels_per_second(self, desired: float, total_duration: float) -> float:
        """Clamp a desired pixels-per-second (from zoom) so the texture fits the GPU.

        Sizing the texture near the on-screen resolution per zoom is what kills the
        minification aliasing: the GPU then scales ~1:1 instead of skipping texels.
        """
        total_duration = max(float(total_duration), 1e-3)
        cap = self.max_texture_width / total_duration
        return max(1.0, min(float(desired), cap))

    def set_track(
        self,
        deck_idx: int,
        image: Image.Image,
        pixels_per_second: float,
        total_duration: float,
        downbeats_sec: Sequence[float] = (),
        cues: Sequence = (),
    ) -> None:
        """Upload a full-track waveform image to a GPU texture for the given deck."""
        if deck_idx not in self._decks:
            return
        surface = pygame.image.frombuffer(image.tobytes(), image.size, "RGB")
        texture = Texture.from_surface(self.renderer, surface)
        self._decks[deck_idx] = _DeckTexture(
            texture=texture,
            pixels_per_second=float(pixels_per_second),
            total_duration=max(float(total_duration), 1e-6),
            downbeats=tuple(float(t) for t in downbeats_sec),
            cues=tuple(cues),
        )

    def clear_track(self, deck_idx: int) -> None:
        self._decks[deck_idx] = None

    # ------------------------------------------------------------------
    # Per-frame rendering
    # ------------------------------------------------------------------
    def render_frame(self, frames: Sequence[DeckFrame]) -> None:
        if self._closed:
            return
        w, h = self.window.size
        deck_h = max(1, h // self.deck_count)

        self.renderer.draw_color = self.background
        self.renderer.clear()

        for idx in range(self.deck_count):
            viewport = pygame.Rect(0, idx * deck_h, w, deck_h)
            frame = frames[idx] if idx < len(frames) else None
            deck = self._decks.get(idx)
            if frame is not None and deck is not None and frame.current_time is not None:
                self._draw_deck(deck, frame, viewport)
            if frame is not None and frame.label:
                self._draw_label(frame.label, viewport)

        self.renderer.present()

    def _draw_deck(self, deck: _DeckTexture, frame: DeckFrame, viewport: pygame.Rect) -> None:
        total = deck.total_duration
        visible = max(frame.visible_seconds, 1e-3)
        start_time = frame.current_time - visible / 2.0
        px_per_sec = viewport.width / visible
        vh = float(viewport.height)

        # Clip + offset to the deck viewport so we can draw in deck-local coords and
        # let off-screen parts of the (wide) quad be clipped away cheaply by the GPU.
        prev_vp = self.renderer.get_viewport()
        self.renderer.set_viewport(viewport)
        try:
            # Whole-track texture as ONE float-positioned quad. Float dst = sub-pixel
            # scrolling (no 1px snapping/jitter); linear filtering keeps it smooth.
            full_w = total * px_per_sec
            full_x = (0.0 - start_time) * px_per_sec
            deck.texture.draw(dstrect=(full_x, 0.0, full_w, vh))

            # Beat markers: soft anti-aliased vertical lines at exact time->x, drawn
            # at float x so they glide sub-pixel-smoothly with the waveform.
            if deck.downbeats and self._beat_tex is not None:
                lo, hi = start_time, start_time + visible
                bw = self._beat_width
                draw_tris = self._beat_triangles and self._tri_top is not None
                for bt in deck.downbeats:
                    if bt < lo or bt > hi:
                        continue
                    bx = (bt - start_time) * px_per_sec
                    self._beat_tex.draw(dstrect=(bx - bw / 2.0, 0.0, bw, vh))
                    if draw_tris:
                        tw, th = self._tri_top.width, self._tri_top.height
                        self._tri_top.draw(dstrect=(bx - tw / 2.0, 0.0, tw, th))
                        self._tri_bottom.draw(dstrect=(bx - tw / 2.0, vh - th, tw, th))

            # Cue markers: a vertical line in the cue colour, with a small flag at the
            # top. Hot cues get a thicker line + larger flag; memory cues a thinner one.
            if deck.cues:
                lo, hi = start_time, start_time + visible
                for ct, color, is_memory, number in deck.cues:
                    if ct < lo or ct > hi:
                        continue
                    cx = (ct - start_time) * px_per_sec
                    self.renderer.draw_color = pygame.Color(*color)
                    lw = 1.0 if is_memory else 2.0
                    self.renderer.fill_rect((cx - lw / 2.0, 0.0, lw, vh))
                    # Square marker at the top, centred on the cue line (double height).
                    sq = 10.0 if is_memory else 14.0
                    self.renderer.fill_rect((cx - sq / 2.0, 0.0, sq, sq))
                    # Hot cues (A-H): draw the slot letter inside the square, in a
                    # contrasting colour (black on light fills, white on dark).
                    if not is_memory and 1 <= number <= 8:
                        fg = (0, 0, 0) if (color[0] + color[1] + color[2]) > 384 else (255, 255, 255)
                        lt = self._cue_letter_texture("ABCDEFGH"[number - 1], fg)
                        if lt is not None:
                            lx = cx - lt.width / 2.0
                            ly = (sq - lt.height) / 2.0
                            lt.draw(dstrect=(lx, ly, lt.width, lt.height))

            # Playhead: fixed at viewport center (link-follow keeps current_time there).
            cx = viewport.width / 2.0
            self.renderer.draw_color = self.playhead_color
            self.renderer.fill_rect((cx - 0.5, 0.0, 1.0, vh))
        finally:
            self.renderer.set_viewport(prev_vp)

    def _cue_letter_texture(self, letter: str, fg: Tuple[int, int, int]):
        """Cached small bold letter texture for a hot-cue square (A-H)."""
        if self._cue_font is None:
            return None
        key = (letter, fg)
        tex = self._cue_label_cache.get(key)
        if tex is None:
            surf = self._cue_font.render(letter, True, fg)
            tex = Texture.from_surface(self.renderer, surf)
            tex.blend_mode = pygame.BLENDMODE_BLEND
            self._cue_label_cache[key] = tex
        return tex

    def _draw_label(self, text: str, viewport: pygame.Rect) -> None:
        if self._font is None:
            return
        tex = self._label_cache.get(text)
        if tex is None:
            surf = self._font.render(text, True, (220, 220, 220))
            tex = Texture.from_surface(self.renderer, surf)
            self._label_cache[text] = tex
        dst = pygame.Rect(viewport.x + 6, viewport.y + 4, tex.width, tex.height)
        tex.draw(dstrect=dst)

    # ------------------------------------------------------------------
    # Event pump / lifecycle
    # ------------------------------------------------------------------
    def pump(self) -> None:
        """Process window events; invoke on_close when the user closes the window."""
        if self._closed:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._request_close()
            elif event.type == getattr(pygame, "WINDOWEVENT", -1):
                # Window close button on some platforms arrives as a WINDOWEVENT.
                if getattr(event, "event", None) == getattr(pygame, "WINDOWEVENT_CLOSE", -2):
                    self._request_close()

    def _request_close(self) -> None:
        if self._closed:
            return
        if self.on_close is not None:
            self.on_close()
        else:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            pygame.quit()
        except Exception:
            pass
