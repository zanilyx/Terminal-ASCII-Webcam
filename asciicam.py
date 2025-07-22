# ascii_camera_gpu.py

import cv2
import numpy as np
import time
import os
import sys
from colorama import init, Fore, Style, Cursor
import atexit
import logging
import shutil
from datetime import datetime
import concurrent.futures
import multiprocessing
import colorsys

try:
    import cv2.data
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
except Exception:
    face_cascade = None

# Cross-platform getch for non-blocking keypress
import threading
import queue

if os.name == 'nt':
    import msvcrt
    def getch():
        if msvcrt.kbhit():
            return msvcrt.getch().decode('utf-8', errors='ignore')
        return None
else:
    import sys, select, tty, termios
    def getch():
        dr, dw, de = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None

# Initialize colorama
init(autoreset=True)

# ASCII character set (dense to sparse)
ASCII_CHARS = '@%#*+=-:. '

# Reduce terminal flicker by precomputing clear behavior
CLEAR_SCREEN = '\033[2J\033[H'  # ANSI clear and home
FLUSH_DELAY = 1 / 30  # Max 30 FPS

# Setup logging
LOG_FILE = 'asciicam.log'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def log_event(msg):
    print(msg)
    logging.info(msg)

def log_error(msg):
    print(msg)
    logging.error(msg)

# Multiprocessing worker for color mode

def color_line_worker(args):
    chars, intensities_row, color_map, rainbow_mode, ascii_w, ascii_h, i = args
    line = ''
    for j, char in enumerate(chars):
        intensity = intensities_row[j]
        if rainbow_mode:
            hue = j / ascii_w
            rgb = colorsys.hsv_to_rgb(hue, 1, 1)
            r, g, b = [int(x * 255) for x in rgb]
            line += f'\033[38;2;{r};{g};{b}m{char}\033[0m'
        else:
            color_idx = min(int(intensity / 255.0 * len(color_map)), len(color_map) - 1)
            line += color_map[color_idx] + char
    return line

class ASCIICamera:
    def __init__(self, width=80, contrast=1.0, brightness=0, use_cuda=False, color_mode=False):
        self.width = width
        self.contrast = contrast
        self.brightness = brightness
        self.use_cuda = use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.color_mode = color_mode
        self.running = False
        self.height_factor = 0.55
        self.show_fps = True
        self.show_help = False
        self.rainbow_mode = False
        self.face_overlay = False
        self.palettes = [
            '@%#*+=-:. ',
            ' .:-=+*#%@',
            '█▓▒░ .',
            '⣿⣿⣶⣤⣀⡀ ',
            '01',
        ]
        self.palette_names = [
            'Dense', 'Inverse', 'Blocks', 'Braille', 'Binary'
        ]
        self.palette_idx = 0
        self.ascii_map = np.array([c for c in self.palettes[self.palette_idx]])
        self.color_map = [
            Fore.BLACK, Fore.RED, Fore.GREEN, Fore.YELLOW,
            Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE
        ]
        self.fps = 0
        self.frame_times = []  # For moving average FPS
        self.fps_window = 30  # Number of frames for moving average
        self.frame_count = 0
        self.start_time = time.time()
        self.crop_roi = None  # (x, y, w, h) or None
        self.inverted = False
        log_event(f"ASCIICamera initialized: width={width}, contrast={contrast}, brightness={brightness}, use_cuda={self.use_cuda}, color_mode={color_mode}")

    def select_crop_region(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log_error("Error: Could not open webcam for preview.")
            return None
        log_event("[PREVIEW] Select region to crop. Drag a rectangle, then press ENTER or SPACE. Press ESC to cancel.")
        while True:
            ret, frame = cap.read()
            if not ret:
                log_error("Failed to grab frame for preview.")
                cap.release()
                return None
            cv2.imshow("Preview - Select ROI and press ENTER", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                log_event("Crop selection cancelled. Using full frame.")
                return None
            # Wait for user to select ROI
            if cv2.getWindowProperty("Preview - Select ROI and press ENTER", cv2.WND_PROP_VISIBLE) < 1:
                break
            # Let user select ROI
            roi = cv2.selectROI("Preview - Select ROI and press ENTER", frame, showCrosshair=True, fromCenter=False)
            cap.release()
            cv2.destroyAllWindows()
            if roi == (0, 0, 0, 0):
                log_event("No region selected. Using full frame.")
                return None
            log_event(f"Selected region: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
            return roi

    def resize_frame(self, frame, width):
        """Resize frame to desired width while preserving aspect ratio."""
        aspect = frame.shape[1] / frame.shape[0]
        new_height = int(width / aspect * self.height_factor)
        return cv2.resize(frame, (width, new_height))

    def frame_to_ascii(self, gray_frame):
        """Convert grayscale frame to ASCII art."""
        # Normalize pixel values to 0–69 (70 levels), map to 9 ASCII chars
        indices = np.clip((gray_frame.astype(np.float32) / 255.0) * (len(self.ascii_map) - 1), 0, len(self.ascii_map) - 1)
        indices = indices.astype(int)
        return self.ascii_map[indices]

    def apply_color(self, char, intensity, j=None, i=None, ascii_w=None, ascii_h=None):
        """Map intensity to a terminal color."""
        if self.rainbow_mode and ascii_w and ascii_h and j is not None and i is not None:
            # Rainbow: color by position
            import colorsys
            hue = j / ascii_w
            rgb = colorsys.hsv_to_rgb(hue, 1, 1)
            r, g, b = [int(x * 255) for x in rgb]
            return f'\033[38;2;{r};{g};{b}m{char}{Style.RESET_ALL}'
        color_idx = min(int(intensity / 255.0 * len(self.color_map)), len(self.color_map) - 1)
        return self.color_map[color_idx] + char

    def save_ascii(self, output_lines):
        try:
            fname = f"ascii_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(fname, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            log_event(f"Saved ASCII art to {fname}")
        except Exception as e:
            log_error(f"Failed to save ASCII art: {e}")

    def save_image(self, frame):
        try:
            fname = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fname, frame)
            log_event(f"Saved image to {fname}")
        except Exception as e:
            log_error(f"Failed to save image: {e}")

    def invert_palette(self):
        self.inverted = not self.inverted
        self.ascii_map = np.array([c for c in self.ascii_map[::-1]])
        log_event(f"Palette inverted: {self.inverted}")

    def next_palette(self):
        self.palette_idx = (self.palette_idx + 1) % len(self.palettes)
        self.ascii_map = np.array([c for c in self.palettes[self.palette_idx]])
        log_event(f"Switched to palette: {self.palette_names[self.palette_idx]}")

    def toggle_rainbow(self):
        self.rainbow_mode = not self.rainbow_mode
        log_event(f"Rainbow mode: {self.rainbow_mode}")

    def toggle_fps(self):
        self.show_fps = not self.show_fps
        log_event(f"Show FPS: {self.show_fps}")

    def toggle_help(self):
        self.show_help = not self.show_help

    def toggle_face_overlay(self):
        self.face_overlay = not self.face_overlay
        log_event(f"Face overlay: {self.face_overlay}")

    def auto_width(self):
        try:
            cols, _ = shutil.get_terminal_size()
            self.width = max(20, min(cols, 160))
            log_event(f"Auto width set to terminal size: {self.width}")
        except Exception as e:
            log_error(f"Failed to auto-set width: {e}")

    def adjust_contrast(self, delta):
        self.contrast = max(0.1, min(self.contrast + delta, 5.0))
        log_event(f"Contrast set to: {self.contrast}")

    def adjust_brightness(self, delta):
        self.brightness = max(-255, min(self.brightness + delta, 255))
        log_event(f"Brightness set to: {self.brightness}")

    def show_help_overlay(self):
        help_lines = [
            "[ASCII Camera Help]",
            "q: Quit",
            "c: Toggle color mode",
            "f: Toggle FPS display",
            "r: Toggle rainbow/heatmap color mode",
            "w: Auto width to terminal size",
            "s: Save ASCII art to text file",
            "p: Save cropped webcam image as PNG/JPG",
            "i: Invert ASCII palette",
            "a: Next ASCII palette",
            "t: Toggle face detection overlay",
            "[: Decrease contrast",
            "]: Increase contrast",
            "{: Decrease brightness",
            "}: Increase brightness",
            "+/-: Increase/decrease ASCII width",
            "h: Toggle this help overlay",
            "",
            f"Current palette: {self.palette_names[self.palette_idx]}"
        ]
        return help_lines

    def run(self):
        log_event("Starting ASCII camera run loop.")
        # Preview and crop selection
        self.crop_roi = self.select_crop_region()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log_error("Error: Could not open webcam.")
            return
        # Set resolution for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(CLEAR_SCREEN, end='', flush=True)
        self.running = True
        max_workers = max(2, multiprocessing.cpu_count() - 1)
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    log_error("Failed to grab frame.")
                    break
                frame_start = time.time()
                frame = cv2.flip(frame, 1)
                # Crop to selected ROI if set
                if self.crop_roi is not None:
                    x, y, w, h = self.crop_roi
                    frame = frame[y:y+h, x:x+w]
                faces = []
                if self.face_overlay and face_cascade is not None:
                    try:
                        gray_for_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray_for_face, 1.1, 4)
                    except Exception as e:
                        log_error(f"Face detection error: {e}")
                        faces = []
                # Use GPU if available
                if self.use_cuda:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    gpu_resized = cv2.cuda.resize(gpu_frame, (self.width, int(self.width * self.height_factor / (frame.shape[1]/frame.shape[0]))))
                    gpu_gray = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2GRAY)
                    resized_gray = gpu_gray.download()
                else:
                    resized = self.resize_frame(frame, self.width)
                    resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                adjusted = np.clip(resized_gray * self.contrast + self.brightness, 0, 255).astype(np.uint8)
                ascii_frame = self.frame_to_ascii(adjusted)
                output_lines = []
                ascii_h, ascii_w = ascii_frame.shape
                adj_h, adj_w = adjusted.shape
                y_idx = np.clip((np.linspace(0, adj_h - 1, ascii_h)).astype(int), 0, adj_h - 1)
                x_idx = np.clip((np.linspace(0, adj_w - 1, ascii_w)).astype(int), 0, adj_w - 1)
                intensities = adjusted[np.ix_(y_idx, x_idx)]
                if self.color_mode and not self.use_cuda:
                    color_indices = np.minimum((intensities / 255.0 * len(self.color_map)).astype(int), len(self.color_map) - 1)
                    args_list = [
                        (ascii_frame[i], intensities[i], self.color_map, self.rainbow_mode, ascii_w, ascii_h, i)
                        for i in range(ascii_h)
                    ]
                    try:
                        output_lines = list(pool.map(color_line_worker, args_list))
                    except Exception as e:
                        log_error(f"Multiprocessing color mode error: {e}")
                        # Fallback to single-threaded
                        for i in range(ascii_h):
                            chars = ascii_frame[i]
                            line = ''
                            for j, char in enumerate(chars):
                                intensity = intensities[i, j]
                                if self.rainbow_mode:
                                    hue = j / ascii_w
                                    rgb = colorsys.hsv_to_rgb(hue, 1, 1)
                                    r, g, b = [int(x * 255) for x in rgb]
                                    line += f'\033[38;2;{r};{g};{b}m{char}\033[0m'
                                else:
                                    color_idx = min(int(intensity / 255.0 * len(self.color_map)), len(self.color_map) - 1)
                                    line += self.color_map[color_idx] + char
                            output_lines.append(line)
                else:
                    for i, row in enumerate(ascii_frame):
                        output_lines.append(''.join(row))
                # Face detection overlay
                if self.face_overlay and faces is not None and len(faces) > 0:
                    for (fx, fy, fw, fh) in faces:
                        # Map face rectangle to ASCII grid
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
                # Add stats
                self.frame_count += 1
                now = time.time()
                self.frame_times.append(now)
                # Keep only the last N frame times
                if len(self.frame_times) > self.fps_window:
                    self.frame_times.pop(0)
                if len(self.frame_times) > 1:
                    self.fps = (len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])
                else:
                    self.fps = 0
                if self.show_help:
                    output_lines = self.show_help_overlay() + [''] + output_lines
                if self.show_fps:
                    output_lines.append("")
                    output_lines.append(f"FPS: {self.fps:.1f} | Q: Quit | Size: {self.width} | C: Color {'ON' if self.color_mode else 'OFF'} | Palette: {self.palette_names[self.palette_idx]}")
                if self.frame_count % 30 == 0:
                    log_event(f"Current FPS: {self.fps:.1f}, width: {self.width}, color_mode: {self.color_mode}")

                # Render in terminal
                output_str = CLEAR_SCREEN + '\n'.join(output_lines)
                print(output_str, end='', flush=True)

                # Handle input without blocking
                key = getch()
                if key:
                    key = key.lower()
                    if key == 'q':
                        log_event("User quit with 'q'.")
                        break
                    elif key == 'c':
                        self.color_mode = not self.color_mode
                        log_event(f"Color mode toggled to: {self.color_mode}")
                        # Reset FPS stats on mode change
                        self.frame_times = [time.time()]
                        self.frame_count = 0
                        self.start_time = time.time()
                    elif key == '+':
                        self.width = min(self.width + 5, 160)
                        log_event(f"Width increased to: {self.width}")
                        self.frame_times = [time.time()]
                        self.frame_count = 0
                        self.start_time = time.time()
                    elif key == '-':
                        self.width = max(self.width - 5, 20)
                        log_event(f"Width decreased to: {self.width}")
                        self.frame_times = [time.time()]
                        self.frame_count = 0
                        self.start_time = time.time()
                    elif key == 's':
                        self.save_ascii(output_lines)
                    elif key == 'p':
                        self.save_image(frame)
                    elif key == 'i':
                        self.invert_palette()
                    elif key == 'a':
                        self.next_palette()
                    elif key == 'r':
                        self.toggle_rainbow()
                    elif key == 'f':
                        self.toggle_fps()
                    elif key == 'w':
                        self.auto_width()
                    elif key == 'h':
                        self.toggle_help()
                    elif key == 't':
                        self.toggle_face_overlay()
                    elif key == '[':
                        self.adjust_contrast(-0.1)
                    elif key == ']':
                        self.adjust_contrast(0.1)
                    elif key == '{':
                        self.adjust_brightness(-10)
                    elif key == '}':
                        self.adjust_brightness(10)

                # Frame rate limiting
                frame_time = time.time() - frame_start
                sleep_time = max(FLUSH_DELAY - frame_time, 0)
                time.sleep(sleep_time)

        finally:
            cap.release()
            log_event("Camera released. Goodbye!")
            print(Cursor.POS() + Style.RESET_ALL + "Camera released. Goodbye!")
            # Clear the terminal at the end
            if os.name == 'nt':
                os.system('cls')
            else:
                os.system('clear')

    def stop(self):
        self.running = False


def prompt_yes_no(prompt, default=False):
    while True:
        resp = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if not resp:
            return default
        if resp in ['y', 'yes']:
            return True
        if resp in ['n', 'no']:
            return False
        print("Please enter y or n.")

def prompt_int(prompt, default):
    while True:
        resp = input(f"{prompt} [{default}]: ").strip()
        if not resp:
            return default
        try:
            return int(resp)
        except ValueError:
            print("Please enter a valid integer.")

if __name__ == "__main__":
    # Optional command-line args
    import argparse
    parser = argparse.ArgumentParser(description="Live ASCII Webcam in Terminal with GPU support")
    parser.add_argument('--width', type=int, help='Width of ASCII output')
    parser.add_argument('--contrast', type=float, default=1.0, help='Contrast multiplier')
    parser.add_argument('--brightness', type=int, default=0, help='Brightness offset')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--color', action='store_true', help='Enable color ASCII output')

    args = parser.parse_args()

    # Interactive prompts if not provided
    # CUDA
    cuda_available = False
    try:
        cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        cuda_available = False

    # Show CUDA availability at the start
    print(f"CUDA available: {'YES' if cuda_available else 'NO'}")

    use_cuda = False
    if args.cuda:
        if not cuda_available:
            print("CUDA not available. Falling back to CPU.")
            use_cuda = False
        else:
            print("Using CUDA acceleration.")
            use_cuda = True
    else:
        if cuda_available:
            use_cuda = prompt_yes_no("CUDA is available. Use it?", default=False)
        else:
            use_cuda = False

    # Width
    width = args.width if args.width else prompt_int("ASCII output width", 80)
    # Color
    color_mode = args.color if args.color else prompt_yes_no("Enable color ASCII output?", default=False)

    # Create and run ASCII camera
    app = ASCIICamera(
        width=width,
        contrast=args.contrast,
        brightness=args.brightness,
        use_cuda=use_cuda,
        color_mode=color_mode
    )

    # Ensure cleanup
    atexit.register(lambda: print(Style.RESET_ALL))

    try:
        app.run()
    except KeyboardInterrupt:
        app.stop()
        print("\nInterrupted by user.")