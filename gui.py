import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import queue

class VisionGUI:
    def __init__(self, root, frame_queue):
        self.root = root
        self.frame_queue = frame_queue

        self.zoom = tk.DoubleVar(value=1.0)
        self.motion_threshold = tk.IntVar(value=5000)
        self.save_enabled = tk.BooleanVar(value=True)

        self.cached_zoom = self.zoom.get()
        self.cached_motion_threshold = self.motion_threshold.get()
        self.cached_save_enabled = self.save_enabled.get()

        # Видео метка
        self.video_label = tk.Label(root)
        self.video_label.pack()

        # FPS
        self.fps_label = ttk.Label(root, text="FPS: 0")
        self.fps_label.pack()

        # Контролы
        ttk.Checkbutton(root, text="Сохранять кадры", variable=self.save_enabled).pack(anchor='w')
        ttk.Label(root, text="Зум")
        ttk.Scale(root, from_=1.0, to=4.0, variable=self.zoom, orient='horizontal', length=200).pack()
        ttk.Label(root, text="Порог движения")
        ttk.Scale(root, from_=100, to=20000, variable=self.motion_threshold, orient='horizontal', length=200).pack()
        ttk.Button(root, text="Выйти", command=root.quit).pack(pady=10)

        self.update_cached_vars()
        self.update_video_loop()

    def update_cached_vars(self):
        self.cached_zoom = self.zoom.get()
        self.cached_motion_threshold = self.motion_threshold.get()
        self.cached_save_enabled = self.save_enabled.get()
        self.root.after(200, self.update_cached_vars)

    def update_video_loop(self):
        try:
            frame = self.frame_queue.get_nowait()
            img = Image.fromarray(frame[..., ::-1])  # BGR->RGB
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        except queue.Empty:
            pass
        self.root.after(30, self.update_video_loop)

    def update_fps(self, fps):
        self.fps_label.config(text=f"FPS: {fps:.1f}")
