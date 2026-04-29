import os
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

from lpAppModel import LPAppModel

class LPApp:
	"""Giao diện Tkinter cho ứng dụng nhận dạng biển số xe."""

	WINDOW_TITLE = "Nhận dạng biển số xe"
	WINDOW_SIZE = "1100x700"

	def __init__(self, root):
		self.root = root
		self.root.title(self.WINDOW_TITLE)
		self.root.geometry(self.WINDOW_SIZE)
		self.root.resizable(True, True)

		self.lp_model = LPAppModel()

		# State
		self.img_path = None
		self.img_original = None       # ảnh gốc BGR (OpenCV)

		# Build UI
		self._build_ui()

	def _build_ui(self):
		# ---- Toolbar trên cùng -------------------------------------------
		toolbar = ttk.Frame(self.root, padding=8)
		toolbar.pack(side=tk.TOP, fill=tk.X)

		btn_open = ttk.Button(toolbar, text="📂 Chọn ảnh", command=self._open_image)
		btn_open.pack(side=tk.LEFT, padx=(0, 10))

		ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

		btn_detect = ttk.Button(
			toolbar, text="🔍 Nhận dạng và đọc", command=self._detect_plates
		)
		btn_detect.pack(side=tk.LEFT, padx=(0, 10))

		# ---- Nội dung chính ----------------------------------------------
		main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
		main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

		# Bên trái: hiển thị ảnh gốc
		left_frame = ttk.LabelFrame(main_pane, text="Ảnh gốc", padding=4)
		main_pane.add(left_frame, weight=3)

		self.canvas_original = tk.Canvas(left_frame, bg="#2b2b2b")
		self.canvas_original.pack(fill=tk.BOTH, expand=True)
		self._photo_original = None  # giữ tham chiếu

		# Bên phải: kết quả
		right_frame = ttk.Frame(main_pane, padding=4)
		main_pane.add(right_frame, weight=2)

		# Ảnh kết quả (có bounding box)
		result_img_frame = ttk.LabelFrame(right_frame, text="Kết quả phát hiện", padding=4)
		result_img_frame.pack(fill=tk.BOTH, expand=True)

		self.canvas_result = tk.Canvas(result_img_frame, bg="#2b2b2b")
		self.canvas_result.pack(fill=tk.BOTH, expand=True)
		self._photo_result = None

		# Biển số tách ra
		plates_frame = ttk.LabelFrame(right_frame, text="Biển số tách được", padding=4)
		plates_frame.pack(fill=tk.X, pady=(6, 0))

		# Scrollable frame cho nhiều biển số
		self.plates_container = ttk.Frame(plates_frame)
		self.plates_container.pack(fill=tk.X)
		self._plate_photos = []  # giữ tham chiếu

		# Kết quả văn bản
		text_frame = ttk.LabelFrame(right_frame, text="Kết quả nhận dạng", padding=4)
		text_frame.pack(fill=tk.X, pady=(6, 0))

		self.result_text = tk.Text(text_frame, height=4, font=("Consolas", 14), wrap=tk.WORD)
		self.result_text.pack(fill=tk.X)

		# Status bar
		self.status_var = tk.StringVar(value="Sẵn sàng. Hãy chọn ảnh để bắt đầu.")
		status_bar = ttk.Label(
			self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=4
		)
		status_bar.pack(side=tk.BOTTOM, fill=tk.X)

	# -- Xử lý sự kiện ----------------------------------------------------

	def _open_image(self):
		path = filedialog.askopenfilename(
			title="Chọn ảnh",
			filetypes=[
				("Ảnh", "*.jpg *.jpeg *.png *.bmp *.tiff"),
				("Tất cả", "*.*"),
			],
		)
		if not path:
			return

		self.img_path = path
		self.img_original = cv2.imread(path)
		if self.img_original is None:
			self.status_var.set(f"Không thể đọc ảnh: {path}")
			return

		# Hiển thị ảnh gốc
		self._display_cv_image(self.img_original, self.canvas_original, 'original')

		# Xóa kết quả cũ
		self._clear_results()
		self.status_var.set(f"Đã tải: {os.path.basename(path)}. Bấm Nhận dạng để bắt đầu.")

	def _detect_plates(self):
		"""Bắt đầu nhận dạng YOLOv8."""
		if self.img_original is None:
			self.status_var.set("Chưa chọn ảnh!")
			return

		self.status_var.set("Đang nhận dạng…")
		self.root.update_idletasks()

		self.lp_model.detect_n_read(self.img_original)

		# Vẽ kết quả lên ảnh
		img_result = self.lp_model.draw_rect()
		self._display_cv_image(img_result, self.canvas_result, 'result')

		# Hiển thị từng biển số
		self._clear_plate_display()
		self.result_text.delete('1.0', tk.END)

		detected_confs = self.lp_model.detected_confs
		lp_texts = self.lp_model.lp_texts
		lp_confs = self.lp_model.lp_confs

		if not detected_confs:
			self.result_text.insert(tk.END, "Không tìm thấy biển số nào.")
			self.status_var.set("Không tìm thấy biển số.")
			return

		num_of_lps = len(detected_confs)
		for i in range(num_of_lps):
			lp_img = self.lp_model.lp_imgs[i]
			detected_conf = detected_confs[i]
			lp_text = lp_texts[i]
			lp_conf = lp_confs[i]

			display_text = lp_text if lp_text else '(không đọc được)'
			self._add_plate_image(lp_img, i, display_text)
			self.result_text.insert(
				tk.END,
				f"Biển {i + 1}: {display_text}  "
				f"(YOLO: {detected_conf:.2f} | OCR: {lp_conf:.2f})\n"
			)

		self.status_var.set(f"Tìm thấy {num_of_lps} biển số.")

	# -- Hiển thị ảnh ------------------------------------------------------

	def _display_cv_image(self, cv_img, canvas, tag):
		"""Hiển thị ảnh OpenCV (BGR) lên Canvas Tkinter, co giãn vừa khung."""
		canvas.update_idletasks()
		cw = canvas.winfo_width() or 400
		ch = canvas.winfo_height() or 400

		# Chuyển BGR → RGB
		rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		pil_img = Image.fromarray(rgb)

		# Co giãn giữ tỉ lệ
		pil_img.thumbnail((cw, ch), Image.LANCZOS)
		photo = ImageTk.PhotoImage(pil_img)

		canvas.delete("all")
		canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=photo)

		# Giữ tham chiếu tránh bị garbage collect
		if tag == 'original':
			self._photo_original = photo
		else:
			self._photo_result = photo

	def _add_plate_image(self, gray_img, index, text):
		"""Thêm ảnh biển số tách được vào container."""
		frame = ttk.Frame(self.plates_container)
		frame.pack(side=tk.LEFT, padx=4, pady=4)

		# Resize cho vừa hiển thị
		h, w = gray_img.shape[:2]
		scale = min(150 / w, 80 / h, 1.0)
		new_w, new_h = int(w * scale), int(h * scale)
		resized = cv2.resize(gray_img, (new_w, new_h))

		if len(resized.shape) == 3:
			resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

		pil_img = Image.fromarray(resized)
		photo = ImageTk.PhotoImage(pil_img)
		self._plate_photos.append(photo)

		lbl_img = ttk.Label(frame, image=photo)
		lbl_img.pack()
		# lbl_text = ttk.Label(frame, text=text or "?", font=("Consolas", 11, "bold"))
		# lbl_text.pack()

	def _clear_results(self):
		self._clear_plate_display()
		self.result_text.delete('1.0', tk.END)
		self.canvas_result.delete("all")
		self._photo_result = None

	def _clear_plate_display(self):
		for child in self.plates_container.winfo_children():
			child.destroy()
		self._plate_photos.clear()
