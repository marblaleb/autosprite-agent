import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from PIL import Image, ImageTk

from src.orchestrator import Orchestrator


def load_config(path: str = "config.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


class AutoSpriteApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AutoSprite Agent")
        self.root.resizable(False, False)
        self.config = load_config()
        self.orchestrator = Orchestrator(self.config)
        self._photo = None
        self._build_ui()

    def _build_ui(self):
        left = ttk.Frame(self.root, padding=16)
        left.grid(row=0, column=0, sticky="nsew")

        ttk.Label(left, text="Personaje:").grid(row=0, column=0, sticky="w")
        self.character_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.character_var, width=30).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12)
        )

        ttk.Label(left, text="Animación:").grid(row=2, column=0, sticky="w")
        self.anim_var = tk.StringVar(value="walk")
        for i, anim in enumerate(["walk", "run", "jump", "attack"]):
            ttk.Radiobutton(
                left, text=anim.capitalize(), variable=self.anim_var, value=anim
            ).grid(row=3 + i // 2, column=i % 2, sticky="w")

        ttk.Label(left, text="Frames:").grid(row=5, column=0, sticky="w", pady=(12, 0))
        self.frames_var = tk.IntVar(value=4)
        ttk.Combobox(
            left, textvariable=self.frames_var, values=[2, 4, 6, 8], state="readonly", width=8
        ).grid(row=6, column=0, sticky="w")

        ttk.Label(left, text="Referencia (opc.):").grid(row=7, column=0, sticky="w", pady=(12, 0))
        self.ref_var = tk.StringVar()
        ref_frame = ttk.Frame(left)
        ref_frame.grid(row=8, column=0, columnspan=2, sticky="ew")
        ttk.Entry(ref_frame, textvariable=self.ref_var, width=22).pack(side="left")
        ttk.Button(ref_frame, text="Browse", command=self._browse_ref).pack(side="left", padx=4)

        self.generate_btn = ttk.Button(left, text="GENERAR", command=self._on_generate)
        self.generate_btn.grid(row=9, column=0, columnspan=2, pady=16, ipadx=8, ipady=4)

        ttk.Separator(left).grid(row=10, column=0, columnspan=2, sticky="ew")
        self.status_var = tk.StringVar(value="Listo.")
        self.status_label = ttk.Label(left, textvariable=self.status_var, foreground="gray")
        self.status_label.grid(row=11, column=0, columnspan=2, sticky="w", pady=(8, 0))

        right = ttk.Frame(self.root, padding=16)
        right.grid(row=0, column=1, sticky="nsew")
        self.preview_label = ttk.Label(
            right, text="Sin generar", width=45, anchor="center", relief="sunken"
        )
        self.preview_label.pack(fill="both", expand=True, ipady=150)

    def _browse_ref(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")]
        )
        if path:
            self.ref_var.set(path)

    def _on_generate(self):
        character = self.character_var.get().strip()
        if not character:
            messagebox.showwarning("Campo vacío", "Describe el personaje antes de generar.")
            return

        self.generate_btn.config(state="disabled", text="Generando…")
        self._set_status("Generando…", "gray")

        threading.Thread(
            target=self._run_generation,
            args=(
                character,
                self.anim_var.get(),
                self.frames_var.get(),
                self.ref_var.get() or None,
            ),
            daemon=True,
        ).start()

    def _run_generation(self, character, anim_type, num_frames, reference_path):
        def on_progress(current, total):
            self.root.after(0, self._set_status, f"Generando frame {current}/{total}…", "gray")

        try:
            output_path = self.orchestrator.run(
                anim_type, character, num_frames, reference_path, progress_callback=on_progress
            )
            self.root.after(0, self._on_success, output_path)
        except Exception as e:
            self.root.after(0, self._on_error, str(e))

    def _on_success(self, output_path):
        self.generate_btn.config(state="normal", text="GENERAR")
        self._set_status(f"Guardado: {output_path}", "green")
        self._show_preview(output_path)

    def _on_error(self, message):
        self.generate_btn.config(state="normal", text="GENERAR")
        self._set_status(f"Error: {message}", "red")

    def _show_preview(self, path):
        img = Image.open(path)
        img.thumbnail((420, 200))
        self._photo = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self._photo, text="")

    def _set_status(self, msg: str, color: str):
        self.status_var.set(msg)
        self.status_label.config(foreground=color)


if __name__ == "__main__":
    root = tk.Tk()
    app = AutoSpriteApp(root)
    root.mainloop()
