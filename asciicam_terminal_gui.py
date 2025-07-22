import sys
import os
import time
import numpy as np
import cv2
from colorama import init, Fore, Style, Cursor
import atexit
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import logging
import shutil
from datetime import datetime
import colorsys

# =====================
# Logging Configuration
# =====================

LOG_FILE = 'asciicam_gui.log'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def log_event(msg):
    print(msg)
    logging.info(msg)


def log_error(msg):
    print(msg)
    logging.error(msg)


# =====================
# ASCII Art Configuration
# =====================

init(autoreset=True)
ASCII_PALETTES = [
    '@%#*+=-:. ',
    ' .:-=+*#%@',
    '█▓▒░ .',
    '⣿⣿⣶⣤⣀⡀ ',
    '01',
]
PALETTE_NAMES = [
    'Dense', 'Inverse', 'Blocks', 'Braille', 'Binary'
]
COLOR_MAP = [
    Fore.BLACK, Fore.RED, Fore.GREEN, Fore.YELLOW,
    Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE
]
CLEAR_SCREEN = '\033[2J\033[H'
FLUSH_DELAY = 1 / 30


# =====================
# Shared State for GUI/Terminal
# =====================

class SharedState:
    """Thread-safe shared state for GUI and terminal camera."""

    def __init__(self):
        # Camera/ASCII settings
        self.width = 80
        self.contrast = 1.0
        self.brightness = 0
        self.color_mode = False
        self.rainbow_mode = False
        self.palette_idx = 0
        self.inverted = False
        self.face_overlay = False
        self.show_fps = True
        # Control flags
        self.quit = False
        self.save_ascii = False
        self.save_image = False
        self.auto_width = False
        # Stats
        self.fps = 0.0
        self.frame_count = 0
        # Internal state
        self.lock = threading.Lock()
        self.last_stats = ''
        self.crop_roi = None
        self.reset_requested = False
        self.cropping = False


# =====================
# Terminal ASCII Camera
# =====================

class ASCIICameraTerminal:
    """Main ASCII camera loop for terminal display, controlled by SharedState."""

    def __init__(self, shared_state):
        self.shared = shared_state
        self.height_factor = 0.55
        self.fps_window = 30
        self.frame_times = []
        self.start_time = time.time()
        # Load face detection if available
        try:
            import cv2.data
            FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        except Exception:
            self.face_cascade = None
        self.ascii_map = np.array([c for c in ASCII_PALETTES[self.shared.palette_idx]])
        # Crop selection at start
        self.select_crop_region()

    def select_crop_region(self):
        """Open a window for the user to select a crop region. Updates shared.crop_roi."""
        self.shared.cropping = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log_error("Error: Could not open webcam for crop selection.")
            self.shared.crop_roi = None
            self.shared.cropping = False
            return
        log_event("[PREVIEW] Select region to crop. Drag a rectangle, then press ENTER or SPACE. Press ESC to cancel.")
        ret, frame = cap.read()
        if not ret:
            log_error("Failed to grab frame for crop selection.")
            cap.release()
            self.shared.crop_roi = None
            self.shared.cropping = False
            return
        roi = cv2.selectROI("Preview - Select ROI and press ENTER", frame, showCrosshair=True, fromCenter=False)
        cap.release()
        cv2.destroyAllWindows()
        if roi == (0, 0, 0, 0):
            log_event("No region selected. Using full frame.")
            self.shared.crop_roi = None
        else:
            log_event(f"Selected region: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
            self.shared.crop_roi = roi
        self.shared.cropping = False

    def frame_to_ascii(self, gray_frame):
        """Convert a grayscale frame to an ASCII character array."""
        indices = np.clip((gray_frame.astype(np.float32) / 255.0) * (len(self.ascii_map) - 1), 0, len(self.ascii_map) - 1)
        indices = indices.astype(int)
        return self.ascii_map[indices]

    def apply_color(self, char, intensity, j=None, i=None, ascii_w=None, ascii_h=None):
        """Map intensity to a terminal color, or rainbow if enabled."""
        if self.shared.rainbow_mode and ascii_w and ascii_h and j is not None and i is not None:
            hue = j / ascii_w
            rgb = colorsys.hsv_to_rgb(hue, 1, 1)
            r, g, b = [int(x * 255) for x in rgb]
            return f'\033[38;2;{r};{g};{b}m{char}{Style.RESET_ALL}'
        color_idx = min(int(intensity / 255.0 * len(COLOR_MAP)), len(COLOR_MAP) - 1)
        return COLOR_MAP[color_idx] + char

    def run(self):
        """Main camera loop: reads frames, processes, and prints ASCII art to terminal."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log_error("Error: Could not open webcam.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(CLEAR_SCREEN, end='', flush=True)
        try:
            while not self.shared.quit:
                # Handle reset/crop request
                if self.shared.reset_requested:
                    log_event("Reset requested from GUI. Pausing camera and opening crop selection.")
                    # Only reset controls, not crop
                    with self.shared.lock:
                        self.shared.width = 80
                        self.shared.contrast = 1.0
                        self.shared.brightness = 0
                        self.shared.color_mode = False
                        self.shared.rainbow_mode = False
                        self.shared.palette_idx = 0
                        self.shared.inverted = False
                        self.shared.face_overlay = False
                        self.shared.show_fps = True
                    self.shared.reset_requested = False
                    print(CLEAR_SCREEN, end='', flush=True)
                if self.shared.cropping:
                    time.sleep(0.1)
                    continue
                ret, frame = cap.read()
                if not ret:
                    log_error("Failed to grab frame.")
                    break
                frame_start = time.time()
                frame = cv2.flip(frame, 1)
                # Apply crop if set
                crop = self.shared.crop_roi
                if crop is not None:
                    x, y, w, h = crop
                    frame = frame[y:y+h, x:x+w]
                # Face detection
                faces = []
                if self.shared.face_overlay and self.face_cascade is not None:
                    try:
                        gray_for_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray_for_face, 1.1, 4)
                    except Exception as e:
                        log_error(f"Face detection error: {e}")
                        faces = []
                # Resize and grayscale
                width = self.shared.width
                contrast = self.shared.contrast
                brightness = self.shared.brightness
                palette_idx = self.shared.palette_idx
                # Clamp palette_idx and rebuild ascii_map
                try:
                    palette_idx = int(palette_idx)
                    if palette_idx < 0 or palette_idx >= len(ASCII_PALETTES):
                        palette_idx = 0
                        log_error(f"Palette index out of range, reset to 0.")
                except Exception as e:
                    palette_idx = 0
                    log_error(f"Palette index error: {e}, reset to 0.")
                self.shared.palette_idx = palette_idx
                try:
                    if self.shared.inverted:
                        ascii_map = np.array([c for c in ASCII_PALETTES[palette_idx][::-1]])
                    else:
                        ascii_map = np.array([c for c in ASCII_PALETTES[palette_idx]])
                    self.ascii_map = ascii_map
                except Exception as e:
                    log_error(f"Error rebuilding ascii_map: {e}")
                    self.ascii_map = np.array([c for c in ASCII_PALETTES[0]])
                resized = self.resize_frame(frame, width)
                resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                adjusted = np.clip(resized_gray * contrast + brightness, 0, 255).astype(np.uint8)
                ascii_frame = self.frame_to_ascii(adjusted)
                output_lines = []
                ascii_h, ascii_w = ascii_frame.shape
                adj_h, adj_w = adjusted.shape
                y_idx = np.clip((np.linspace(0, adj_h - 1, ascii_h)).astype(int), 0, adj_h - 1)
                x_idx = np.clip((np.linspace(0, adj_w - 1, ascii_w)).astype(int), 0, adj_w - 1)
                intensities = adjusted[np.ix_(y_idx, x_idx)]
                # Build ASCII output (color or B&W)
                if self.shared.color_mode:
                    color_indices = np.minimum((intensities / 255.0 * len(COLOR_MAP)).astype(int), len(COLOR_MAP) - 1)
                    for i in range(ascii_h):
                        chars = ascii_frame[i]
                        colors = color_indices[i]
                        line = ''.join([self.apply_color(chars[j], intensities[i, j], j, i, ascii_w, ascii_h) for j in range(ascii_w)])
                        output_lines.append(line)
                else:
                    for i, row in enumerate(ascii_frame):
                        output_lines.append(''.join(row))
                # Face overlay in ASCII
                if self.shared.face_overlay and faces is not None and len(faces) > 0:
                    for (fx, fy, fw, fh) in faces:
                        ascii_fx = int(fx / frame.shape[1] * ascii_w)
                        ascii_fy = int(fy / frame.shape[0] * ascii_h)
                        ascii_fw = max(1, int(fw / frame.shape[1] * ascii_w))
                        ascii_fh = max(1, int(fh / frame.shape[0] * ascii_h))
                        for y in range(ascii_fy, min(ascii_fy + ascii_fh, ascii_h)):
                            if 0 <= y < ascii_h:
                                row = list(output_lines[y])
                                for x in range(ascii_fx, min(ascii_fx + ascii_fw, ascii_w)):
                                    if 0 <= x < ascii_w:
                                        row[x] = '\033[41m' + row[x] + Style.RESET_ALL
                                output_lines[y] = ''.join(row)
                # FPS calculation
                now = time.time()
                self.frame_times.append(now)
                if len(self.frame_times) > self.fps_window:
                    self.frame_times.pop(0)
                if len(self.frame_times) > 1:
                    self.shared.fps = (len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])
                else:
                    self.shared.fps = 0
                self.shared.frame_count += 1
                # Save ASCII or image if requested
                if self.shared.save_ascii:
                    try:
                        fname = f"ascii_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        with open(fname, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(output_lines))
                        log_event(f"Saved ASCII art to {fname}")
                    except Exception as e:
                        log_error(f"Failed to save ASCII art: {e}")
                    self.shared.save_ascii = False
                if self.shared.save_image:
                    try:
                        fname = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        cv2.imwrite(fname, frame)
                        log_event(f"Saved image to {fname}")
                    except Exception as e:
                        log_error(f"Failed to save image: {e}")
                    self.shared.save_image = False
                # Print to terminal
                stats_line = f"FPS: {self.shared.fps:.1f} | Size: {self.shared.width} | Palette: {PALETTE_NAMES[self.shared.palette_idx]} | Color: {self.shared.color_mode} | Rainbow: {self.shared.rainbow_mode} | Invert: {self.shared.inverted} | Face: {self.shared.face_overlay}"
                if self.shared.show_fps:
                    output_lines.append('')
                    output_lines.append(stats_line)
                print(CLEAR_SCREEN + '\n'.join(output_lines), end='', flush=True)
                # Sleep for frame rate
                frame_time = time.time() - frame_start
                sleep_time = max(FLUSH_DELAY - frame_time, 0)
                time.sleep(sleep_time)
        finally:
            cap.release()
            log_event("Camera released. Goodbye!")
            print(Cursor.POS() + Style.RESET_ALL + "Camera released. Goodbye!")
            if os.name == 'nt':
                os.system('cls')
            else:
                os.system('clear')

    def resize_frame(self, frame, width):
        """Resize frame to desired width while preserving aspect ratio."""
        aspect = frame.shape[1] / frame.shape[0]
        new_height = int(width / aspect * self.height_factor)
        if new_height < 10:
            new_height = 10
        return cv2.resize(frame, (width, new_height))


# =====================
# GUI Remote (Tkinter)
# =====================

class ControlGUI(threading.Thread):
    """Tkinter GUI for controlling the ASCII camera in real time."""

    def __init__(self, shared_state):
        super().__init__()
        self.shared = shared_state
        self.daemon = True

    def run(self):
        root = tk.Tk()
        root.title("ASCII Camera Remote")
        root.configure(bg="#222244")
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 12, 'bold'), padding=6)
        style.configure('TLabel', font=('Arial', 11), background="#222244", foreground="#fff")
        style.configure('TCheckbutton', background="#222244", foreground="#fff")

        # Controls Frame
        controls = tk.Frame(root, bg="#222244")
        controls.pack(padx=10, pady=10)

        # Width
        tk.Label(controls, text="ASCII Width", bg="#222244", fg="#fff").grid(row=0, column=0, sticky='w')
        width_slider = tk.Scale(controls, from_=20, to=160, orient=tk.HORIZONTAL, length=180, fg="#222244", bg="#444466", troughcolor="#8888aa")
        width_slider.set(self.shared.width)
        width_slider.grid(row=0, column=1, padx=5)

        # Contrast
        tk.Label(controls, text="Contrast", bg="#222244", fg="#fff").grid(row=1, column=0, sticky='w')
        contrast_slider = tk.Scale(controls, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, length=180, fg="#222244", bg="#444466", troughcolor="#8888aa")
        contrast_slider.set(self.shared.contrast)
        contrast_slider.grid(row=1, column=1, padx=5)

        # Brightness
        tk.Label(controls, text="Brightness", bg="#222244", fg="#fff").grid(row=2, column=0, sticky='w')
        brightness_slider = tk.Scale(controls, from_=-255, to=255, orient=tk.HORIZONTAL, length=180, fg="#222244", bg="#444466", troughcolor="#8888aa")
        brightness_slider.set(self.shared.brightness)
        brightness_slider.grid(row=2, column=1, padx=5)

        # Palette
        tk.Label(controls, text="Palette", bg="#222244", fg="#fff").grid(row=3, column=0, sticky='w')
        palette_menu = ttk.Combobox(controls, values=PALETTE_NAMES, state="readonly")
        palette_menu.current(self.shared.palette_idx)
        palette_menu.grid(row=3, column=1, padx=5)
        def on_palette_change(event):
            try:
                idx = palette_menu.current()
                with self.shared.lock:
                    self.shared.palette_idx = idx
                log_event(f"Palette changed to {PALETTE_NAMES[idx]}")
            except Exception as e:
                log_error(f"Palette change error: {e}")
        palette_menu.bind("<<ComboboxSelected>>", on_palette_change)

        # Color mode
        color_var = tk.BooleanVar(value=self.shared.color_mode)
        color_check = tk.Checkbutton(controls, text="Color Mode", variable=color_var, bg="#222244", fg="#fff", selectcolor="#00ffcc")
        color_check.grid(row=4, column=0, sticky='w')

        # Rainbow mode
        rainbow_var = tk.BooleanVar(value=self.shared.rainbow_mode)
        rainbow_check = tk.Checkbutton(controls, text="Rainbow", variable=rainbow_var, bg="#222244", fg="#fff", selectcolor="#ff00cc")
        rainbow_check.grid(row=4, column=1, sticky='w')

        # Invert palette
        inverted_var = tk.BooleanVar(value=self.shared.inverted)
        invert_check = tk.Checkbutton(controls, text="Invert Palette", variable=inverted_var, bg="#222244", fg="#fff", selectcolor="#ffff00")
        invert_check.grid(row=5, column=0, sticky='w')

        # Face overlay
        face_var = tk.BooleanVar(value=self.shared.face_overlay)
        face_check = tk.Checkbutton(controls, text="Face Overlay", variable=face_var, bg="#222244", fg="#fff", selectcolor="#ff4444")
        face_check.grid(row=5, column=1, sticky='w')

        # Save buttons
        tk.Button(controls, text="Save ASCII", command=lambda: setattr(self.shared, 'save_ascii', True), bg="#44cc44", fg="#fff").grid(row=6, column=0, pady=5)
        tk.Button(controls, text="Save Image", command=lambda: setattr(self.shared, 'save_image', True), bg="#44aaff", fg="#fff").grid(row=6, column=1, pady=5)

        # FPS display
        fps_var = tk.StringVar(value="0.0")
        tk.Label(controls, text="FPS:", bg="#222244", fg="#fff").grid(row=7, column=0, sticky='w')
        fps_label = tk.Label(controls, textvariable=fps_var, bg="#222244", fg="#fff")
        fps_label.grid(row=7, column=1, sticky='w')

        # Show FPS toggle
        show_fps_var = tk.BooleanVar(value=self.shared.show_fps)
        show_fps_check = tk.Checkbutton(controls, text="Show FPS/Stats", variable=show_fps_var, bg="#222244", fg="#fff", selectcolor="#00ff00")
        show_fps_check.grid(row=8, column=0, sticky='w')

        # Help overlay
        def show_help():
            messagebox.showinfo("Help", """
ASCII Camera Controls:
- Adjust width, contrast, brightness, palette, color, rainbow, invert, face overlay from the GUI.
- Save ASCII art or webcam image with the buttons.
- Toggle FPS/stats display.
- Reset to restore defaults (does not affect crop).
- Quit from the GUI or Ctrl+C in terminal.
- All changes are live!
""")
        tk.Button(controls, text="Help", command=show_help, bg="#ffaa00", fg="#222244").grid(row=8, column=1, sticky='w')

        # Reset button (controls only)
        def reset_all():
            with self.shared.lock:
                self.shared.width = 80
                self.shared.contrast = 1.0
                self.shared.brightness = 0
                self.shared.color_mode = False
                self.shared.rainbow_mode = False
                self.shared.palette_idx = 0
                self.shared.inverted = False
                self.shared.face_overlay = False
                self.shared.show_fps = True
            # Also update GUI widgets to match defaults
            width_slider.set(80)
            contrast_slider.set(1.0)
            brightness_slider.set(0)
            palette_menu.current(0)
            color_var.set(False)
            rainbow_var.set(False)
            inverted_var.set(False)
            face_var.set(False)
            show_fps_var.set(True)
            log_event("Reset all controls to defaults (crop unchanged).")
        tk.Button(controls, text="Reset Controls", command=reset_all, bg="#ff4444", fg="#fff").grid(row=9, column=0, pady=10)

        # Quit
        tk.Button(controls, text="Quit", command=lambda: setattr(self.shared, 'quit', True), bg="#222244", fg="#fff").grid(row=9, column=1, pady=10)

        # Update loop
        def update_shared():
            with self.shared.lock:
                self.shared.width = width_slider.get()
                self.shared.contrast = contrast_slider.get()
                self.shared.brightness = brightness_slider.get()
                # Palette index is set by Combobox callback
                self.shared.color_mode = color_var.get()
                self.shared.rainbow_mode = rainbow_var.get()
                self.shared.face_overlay = face_var.get()
                self.shared.inverted = inverted_var.get()
                self.shared.show_fps = show_fps_var.get()
                fps_var.set(f"{self.shared.fps:.1f}")
            root.after(100, update_shared)
        update_shared()
        root.mainloop()


# =====================
# Main Entrypoint
# =====================

if __name__ == "__main__":
    shared = SharedState()
    gui = ControlGUI(shared)
    gui.start()
    atexit.register(lambda: print(Style.RESET_ALL))
    cam = ASCIICameraTerminal(shared)
    try:
        cam.run()
    except KeyboardInterrupt:
        shared.quit = True
        log_event("Interrupted by user.")
        print("\nInterrupted by user.") 