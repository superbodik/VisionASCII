import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import time
import os
from PIL import Image, ImageTk

class InteractiveTrainer:
    """Интерактивная система обучения с пользовательской разметкой"""
    
    def __init__(self, smart_tracker):
        self.smart_tracker = smart_tracker
        self.training_window = None
        self.current_frame = None
        self.current_detections = []
        self.user_corrections = []
        self.manual_selections = []  # Новый список для ручных выделений
        
        # Переменные для рисования
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None
        self.manual_class = 'shadow'  # По умолчанию
        
        # Счетчики обучения
        self.corrections_count = {
            'shadow': 0,
            'person': 0, 
            'vehicle': 0,
            'unknown': 0
        }
        
        self.manual_count = {
            'shadow': 0,
            'person': 0, 
            'vehicle': 0,
            'unknown': 0
        }
        
    def show_training_interface(self, frame, detections):
        """Показывает интерфейс для ручной разметки"""
        self.current_frame = frame.copy()
        self.current_detections = detections
        
        # Создаем окно обучения
        if self.training_window is None or not self.training_window.winfo_exists():
            self.create_training_window()
            
        self.update_training_display()
    
    def create_training_window(self):
        """Создает окно интерактивного обучения"""
        self.training_window = tk.Toplevel()
        self.training_window.title("Обучение нейронной сети - Интерактивная разметка")
        self.training_window.geometry("900x1000")
        
        # Фрейм для изображения
        image_frame = tk.Frame(self.training_window)
        image_frame.pack(pady=10)
        
        self.image_label = tk.Label(image_frame)
        self.image_label.pack()
        
        # === СЕКЦИЯ 1: ИСПРАВЛЕНИЕ АВТОМАТИЧЕСКИХ ДЕТЕКЦИЙ ===
        correction_frame = tk.LabelFrame(self.training_window, text="1. Исправление автоматических детекций", 
                                       font=("Arial", 11, "bold"))
        correction_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(correction_frame, text="Кликните на объект чтобы исправить его класс:",
                font=("Arial", 10)).pack(pady=2)
        
        # Кнопки классов для исправлений
        correction_buttons_frame = tk.Frame(correction_frame)
        correction_buttons_frame.pack(pady=5)
        
        self.class_buttons = {}
        classes = [
            ('Тень/Шум', 'shadow', '#808080'),
            ('Человек', 'person', '#FF0000'),
            ('Машина', 'vehicle', '#00FF00'),
            ('Неизвестно', 'unknown', '#FF00FF')
        ]
        
        for i, (name, class_key, color) in enumerate(classes):
            btn = tk.Button(correction_buttons_frame, text=name, bg=color, fg='white',
                           font=("Arial", 9, "bold"), width=10,
                           command=lambda k=class_key: self.set_correction_class(k))
            btn.grid(row=0, column=i, padx=3)
            self.class_buttons[class_key] = btn
            
        # Информация о текущем выборе для исправлений
        self.selection_label = tk.Label(correction_frame, 
                                       text="Выберите объект и класс для исправления",
                                       font=("Arial", 9))
        self.selection_label.pack(pady=2)
        
        # === СЕКЦИЯ 2: РУЧНОЕ ВЫДЕЛЕНИЕ НОВЫХ ОБЪЕКТОВ ===
        manual_frame = tk.LabelFrame(self.training_window, text="2. Ручное выделение новых объектов", 
                                   font=("Arial", 11, "bold"))
        manual_frame.pack(fill='x', padx=10, pady=5)
        
        # Инструкции для ручного выделения
        instructions_frame = tk.Frame(manual_frame)
        instructions_frame.pack(pady=5)
        
        tk.Label(instructions_frame, text="1) Выберите класс  2) Зажмите ЛКМ и выделите область  3) Отпустите ЛКМ",
                font=("Arial", 10, "bold"), fg='blue').pack()
        
        # Выбор класса для ручного выделения
        class_selection_frame = tk.Frame(manual_frame)
        class_selection_frame.pack(pady=5)
        
        tk.Label(class_selection_frame, text="Класс для выделения:", 
                font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.manual_class_var = tk.StringVar(value='shadow')
        manual_class_buttons = {}
        
        for i, (name, class_key, color) in enumerate(classes):
            btn = tk.Radiobutton(class_selection_frame, text=name, variable=self.manual_class_var,
                               value=class_key, bg=color, fg='white', selectcolor=color,
                               font=("Arial", 9, "bold"), width=8,
                               command=lambda: self.set_manual_class())
            btn.pack(side=tk.LEFT, padx=2)
            manual_class_buttons[class_key] = btn
        
        # Информация о ручном выделении
        self.manual_label = tk.Label(manual_frame, 
                                    text="Выберите класс и выделите область мышью",
                                    font=("Arial", 9), fg='green')
        self.manual_label.pack(pady=2)
        
        # === СЕКЦИЯ 3: СТАТИСТИКА ===
        stats_frame = tk.LabelFrame(self.training_window, text="3. Статистика", 
                                  font=("Arial", 11, "bold"))
        stats_frame.pack(fill='x', padx=10, pady=5)
        
        self.stats_label = tk.Label(stats_frame, text="", font=("Arial", 9))
        self.stats_label.pack(pady=5)
        
        # === СЕКЦИЯ 4: ДЕЙСТВИЯ ===
        actions_frame = tk.LabelFrame(self.training_window, text="4. Действия", 
                                    font=("Arial", 11, "bold"))
        actions_frame.pack(fill='x', padx=10, pady=5)
        
        actions_buttons_frame = tk.Frame(actions_frame)
        actions_buttons_frame.pack(pady=5)
        
        tk.Button(actions_buttons_frame, text="Применить ВСЕ изменения", 
                 command=self.apply_all_changes, bg='#4CAF50', fg='white',
                 font=("Arial", 10, "bold"), width=18).pack(side=tk.LEFT, padx=5)
                 
        tk.Button(actions_buttons_frame, text="Очистить всё", 
                 command=self.clear_all, bg='#f44336', fg='white',
                 font=("Arial", 10, "bold"), width=12).pack(side=tk.LEFT, padx=5)
                 
        tk.Button(actions_buttons_frame, text="Закрыть", 
                 command=self.close_training_window, bg='#2196F3', fg='white',
                 font=("Arial", 10, "bold"), width=8).pack(side=tk.LEFT, padx=5)
        
        # Переменные для выбора
        self.selected_detection = None
        self.correction_class = None
        
        # Привязываем события к изображению
        self.image_label.bind("<Button-1>", self.on_mouse_down)
        self.image_label.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_label.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        self.update_stats_display()
    
    def set_manual_class(self):
        """Устанавливает класс для ручного выделения"""
        self.manual_class = self.manual_class_var.get()
        class_names = {'shadow': 'тень/шум', 'person': 'человек', 'vehicle': 'машина', 'unknown': 'неизвестно'}
        self.manual_label.configure(
            text=f"Режим выделения: {class_names[self.manual_class]}. Зажмите ЛКМ и выделите область.",
            fg='green'
        )
    
    def on_mouse_down(self, event):
        """Начало выделения области или клик по объекту"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            return
            
        # Преобразуем координаты клика с проверкой границ
        click_x = max(0, min(int(event.x / self.scale_x), self.current_frame.shape[1] - 1))
        click_y = max(0, min(int(event.y / self.scale_y), self.current_frame.shape[0] - 1))
        
        # Сначала проверяем, кликнули ли мы по существующему объекту
        clicked_detection = None
        for i, detection in enumerate(self.current_detections):
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            if x <= click_x <= x + w and y <= click_y <= y + h:
                clicked_detection = i
                break
        
        if clicked_detection is not None:
            # Это клик по существующему объекту - режим исправления
            self.selected_detection = clicked_detection
            detection = self.current_detections[clicked_detection]
            
            class_names = ['тень/шум', 'человек', 'машина', 'неизвестно']
            current_class = class_names[detection['class']]
            
            self.selection_label.configure(
                text=f"Выбран объект #{clicked_detection} (сейчас: {current_class}). "
                     f"Выберите правильный класс."
            )
            
            self.drawing = False
        else:
            # Это начало ручного выделения
            self.drawing = True
            self.start_x = click_x
            self.start_y = click_y
            self.current_rect = None
            
            self.manual_label.configure(
                text="Выделяете область... Отпустите ЛКМ чтобы завершить.",
                fg='orange'
            )
        
        self.update_training_display()
    
    def on_mouse_drag(self, event):
        """Перетаскивание мыши - рисование прямоугольника"""
        if not self.drawing or not hasattr(self, 'current_frame') or self.current_frame is None:
            return
            
        # Преобразуем координаты с проверкой границ
        frame_height, frame_width = self.current_frame.shape[:2]
        current_x = max(0, min(int(event.x / self.scale_x), frame_width - 1))
        current_y = max(0, min(int(event.y / self.scale_y), frame_height - 1))
        
        # Вычисляем прямоугольник с проверкой границ
        x1 = max(0, min(self.start_x, current_x))
        y1 = max(0, min(self.start_y, current_y))
        x2 = min(frame_width - 1, max(self.start_x, current_x))
        y2 = min(frame_height - 1, max(self.start_y, current_y))
        
        w = x2 - x1
        h = y2 - y1
        
        # Обновляем текущий прямоугольник только если он имеет разумный размер
        if w > 5 and h > 5:
            self.current_rect = (x1, y1, w, h)
            
            # Обновляем информацию о выделении
            self.manual_label.configure(
                text=f"Выделяете область: {w}x{h}px. Отпустите ЛКМ чтобы завершить.",
                fg='orange'
            )
        else:
            self.current_rect = None
            self.manual_label.configure(
                text="Область слишком маленькая... Продолжайте выделение.",
                fg='red'
            )
        
        self.update_training_display()
    
    def on_mouse_up(self, event):
        """Завершение выделения области"""
        if not self.drawing:
            return
            
        self.drawing = False
        
        # Проверяем финальные координаты
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            frame_height, frame_width = self.current_frame.shape[:2]
            end_x = max(0, min(int(event.x / self.scale_x), frame_width - 1))
            end_y = max(0, min(int(event.y / self.scale_y), frame_height - 1))
            
            # Финальный расчет прямоугольника
            x1 = max(0, min(self.start_x, end_x))
            y1 = max(0, min(self.start_y, end_y))
            x2 = min(frame_width - 1, max(self.start_x, end_x))
            y2 = min(frame_height - 1, max(self.start_y, end_y))
            
            w = x2 - x1
            h = y2 - y1
            
            # Проверяем минимальный размер
            if w >= 15 and h >= 15:
                final_rect = (x1, y1, w, h)
                
                # Добавляем ручное выделение
                manual_selection = {
                    'bbox': final_rect,
                    'class': self.manual_class,
                    'class_id': {'shadow': 0, 'person': 1, 'vehicle': 2, 'unknown': 3}[self.manual_class],
                    'frame': self.current_frame.copy()
                }
                
                self.manual_selections.append(manual_selection)
                self.manual_count[self.manual_class] += 1
                
                class_names = {'shadow': 'тень/шум', 'person': 'человек', 'vehicle': 'машина', 'unknown': 'неизвестно'}
                self.manual_label.configure(
                    text=f"✅ Добавлено: {class_names[self.manual_class]} "
                         f"({w}x{h}px) в позиции ({x1},{y1})",
                    fg='blue'
                )
                
                self.update_stats_display()
            else:
                self.manual_label.configure(
                    text=f"❌ Область слишком маленькая ({w}x{h}px). Минимум: 15x15px. Попробуйте еще раз.",
                    fg='red'
                )
        else:
            self.manual_label.configure(
                text="❌ Ошибка: нет активного кадра. Попробуйте еще раз.",
                fg='red'
            )
        
        self.current_rect = None
        self.update_training_display()
    
    def _apply_manual_selection(self, manual_sel):
        """Применяет одно ручное выделение с дополнительной проверкой"""
        bbox = manual_sel['bbox']
        class_id = manual_sel['class_id']
        frame = manual_sel['frame']
        
        # Извлекаем область изображения с проверкой границ
        x, y, w, h = bbox
        frame_height, frame_width = frame.shape[:2]
        
        # Двойная проверка границ
        x1 = max(0, min(x, frame_width - 1))
        y1 = max(0, min(y, frame_height - 1))
        x2 = max(x1 + 1, min(x + w, frame_width))
        y2 = max(y1 + 1, min(y + h, frame_height))
        
        if x2 > x1 and y2 > y1:
            try:
                image_patch = frame[y1:y2, x1:x2]
                
                # Проверяем что patch не пустой
                if image_patch.size > 0 and image_patch.shape[0] > 0 and image_patch.shape[1] > 0:
                    # Добавляем в систему обучения с максимальной уверенностью (ручная разметка)
                    self.smart_tracker.learner.add_sample(
                        image_patch, (x1, y1, x2-x1, y2-y1), frame.shape, 
                        class_id, 0.99, true_label=class_id
                    )
                    
                    # Сохраняем ручное выделение
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    class_name = manual_sel['class']
                    
                    os.makedirs(f"training_data/{class_name}", exist_ok=True)
                    filename = f"training_data/{class_name}/manual_{timestamp}_{hash(str(bbox))}.jpg"
                    cv2.imwrite(filename, image_patch)
                    
                    print(f"✅ Сохранен ручной образец: {class_name} ({x2-x1}x{y2-y1}px)")
                else:
                    print(f"❌ Пустой patch для ручного выделения: {bbox}")
            except Exception as e:
                print(f"❌ Ошибка при сохранении ручного выделения: {e}")
        else:
            print(f"❌ Некорректные границы для ручного выделения: {bbox}")
    
    def update_training_display(self):
        """Обновляет отображение кадра с детекциями и выделениями"""
        if self.current_frame is None:
            return
            
        display_frame = self.current_frame.copy()
        frame_height, frame_width = display_frame.shape[:2]
        
        # Рисуем автоматические детекции
        for i, detection in enumerate(self.current_detections):
            bbox = detection['bbox']
            class_id = detection['class']
            confidence = detection['confidence']
            
            x, y, w, h = bbox
            
            # Проверяем границы для автоматических детекций
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            w = min(w, frame_width - x)
            h = min(h, frame_height - y)
            
            # Цвета для классов
            colors = {0: (128, 128, 128), 1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 255)}
            color = colors.get(class_id, (255, 255, 255))
            
            # Рамка (толще если выбран)
            thickness = 3 if i == self.selected_detection else 2
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Подпись
            class_names = ['SHADOW', 'PERSON', 'VEHICLE', 'UNKNOWN']
            label = f"{class_names[class_id]} {confidence:.2f}"
            
            # Если есть исправление для этого объекта
            correction = next((c for c in self.user_corrections if c['detection_id'] == i), None)
            if correction:
                label += f" -> {correction['new_class'].upper()}"
                
            cv2.putText(display_frame, label, (x, max(15, y - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Номер объекта
            cv2.putText(display_frame, f"A{i}", (x + 5, y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Рисуем ручные выделения
        for i, manual_sel in enumerate(self.manual_selections):
            bbox = manual_sel['bbox']
            class_name = manual_sel['class']
            
            x, y, w, h = bbox
            
            # Проверяем границы для ручных выделений
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            w = min(w, frame_width - x)
            h = min(h, frame_height - y)
            
            # Цвета для ручных выделений (пунктирная линия)
            colors = {'shadow': (128, 128, 128), 'person': (0, 0, 255), 'vehicle': (0, 255, 0), 'unknown': (255, 0, 255)}
            color = colors[class_name]
            
            # Пунктирная рамка для ручных выделений
            self.draw_dashed_rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # Подпись
            label = f"MANUAL {class_name.upper()}"
            cv2.putText(display_frame, label, (x, max(15, y - 25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Номер ручного выделения
            cv2.putText(display_frame, f"M{i}", (x + 5, y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Рисуем текущее выделение
        if self.current_rect:
            x, y, w, h = self.current_rect
            
            # Проверяем границы для текущего выделения
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            w = min(w, frame_width - x)
            h = min(h, frame_height - y)
            
            color = {'shadow': (128, 128, 128), 'person': (0, 0, 255), 'vehicle': (0, 255, 0), 'unknown': (255, 0, 255)}[self.manual_class]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, f"SELECTING {w}x{h}", (x, max(15, y - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Конвертируем для отображения в tkinter
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(display_frame_rgb)
        
        # Масштабируем если нужно
        max_width, max_height = 850, 450
        if pil_image.width > max_width or pil_image.height > max_height:
            pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=self.tk_image)
        
        # Сохраняем коэффициент масштабирования
        self.scale_x = pil_image.width / self.current_frame.shape[1]
        self.scale_y = pil_image.height / self.current_frame.shape[0]
    
    def update_training_display(self):
        """Обновляет отображение кадра с детекциями и выделениями"""
        if self.current_frame is None:
            return
            
        display_frame = self.current_frame.copy()
        
        # Рисуем автоматические детекции
        for i, detection in enumerate(self.current_detections):
            bbox = detection['bbox']
            class_id = detection['class']
            confidence = detection['confidence']
            
            x, y, w, h = bbox
            
            # Цвета для классов
            colors = {0: (128, 128, 128), 1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 255)}
            color = colors.get(class_id, (255, 255, 255))
            
            # Рамка (толще если выбран)
            thickness = 3 if i == self.selected_detection else 2
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Подпись
            class_names = ['SHADOW', 'PERSON', 'VEHICLE', 'UNKNOWN']
            label = f"{class_names[class_id]} {confidence:.2f}"
            
            # Если есть исправление для этого объекта
            correction = next((c for c in self.user_corrections if c['detection_id'] == i), None)
            if correction:
                label += f" -> {correction['new_class'].upper()}"
                
            cv2.putText(display_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Номер объекта
            cv2.putText(display_frame, f"A{i}", (x + 5, y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Рисуем ручные выделения
        for i, manual_sel in enumerate(self.manual_selections):
            bbox = manual_sel['bbox']
            class_name = manual_sel['class']
            
            x, y, w, h = bbox
            
            # Цвета для ручных выделений (пунктирная линия)
            colors = {'shadow': (128, 128, 128), 'person': (0, 0, 255), 'vehicle': (0, 255, 0), 'unknown': (255, 0, 255)}
            color = colors[class_name]
            
            # Пунктирная рамка для ручных выделений
            self.draw_dashed_rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # Подпись
            label = f"MANUAL {class_name.upper()}"
            cv2.putText(display_frame, label, (x, y - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Номер ручного выделения
            cv2.putText(display_frame, f"M{i}", (x + 5, y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Рисуем текущее выделение
        if self.current_rect:
            x, y, w, h = self.current_rect
            color = {'shadow': (128, 128, 128), 'person': (0, 0, 255), 'vehicle': (0, 255, 0), 'unknown': (255, 0, 255)}[self.manual_class]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, "SELECTING...", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Конвертируем для отображения в tkinter
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(display_frame_rgb)
        
        # Масштабируем если нужно
        max_width, max_height = 850, 450
        if pil_image.width > max_width or pil_image.height > max_height:
            pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=self.tk_image)
        
        # Сохраняем коэффициент масштабирования
        self.scale_x = pil_image.width / self.current_frame.shape[1]
        self.scale_y = pil_image.height / self.current_frame.shape[0]
    
    def draw_dashed_rectangle(self, img, pt1, pt2, color, thickness):
        """Рисует пунктирный прямоугольник"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Горизонтальные линии
        dash_length = 10
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # Вертикальные линии
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def set_correction_class(self, class_key):
        """Устанавливает класс для исправления автоматической детекции"""
        if self.selected_detection is None:
            messagebox.showwarning("Предупреждение", "Сначала выберите объект!")
            return
            
        # Удаляем предыдущее исправление для этого объекта
        self.user_corrections = [c for c in self.user_corrections 
                                if c['detection_id'] != self.selected_detection]
        
        # Добавляем новое исправление
        detection = self.current_detections[self.selected_detection]
        old_class_id = detection['class']
        
        class_mapping = {'shadow': 0, 'person': 1, 'vehicle': 2, 'unknown': 3}
        new_class_id = class_mapping[class_key]
        
        if old_class_id != new_class_id:  # Только если класс действительно изменился
            correction = {
                'detection_id': self.selected_detection,
                'bbox': detection['bbox'],
                'old_class': old_class_id,
                'new_class': class_key,
                'new_class_id': new_class_id,
                'frame': self.current_frame.copy()
            }
            
            self.user_corrections.append(correction)
            self.corrections_count[class_key] += 1
            
            class_names = ['тень/шум', 'человек', 'машина', 'неизвестно']
            self.selection_label.configure(
                text=f"Объект #{self.selected_detection} изменен: "
                     f"{class_names[old_class_id]} -> {class_key}"
            )
            
            self.update_training_display()
            self.update_stats_display()
    
    def update_stats_display(self):
        """Обновляет отображение статистики"""
        corrections_total = sum(self.corrections_count.values())
        manual_total = sum(self.manual_count.values())
        
        stats_text = f"📝 ИСПРАВЛЕНИЯ автоматических детекций: {corrections_total}\n"
        stats_text += f"   Тени: {self.corrections_count['shadow']}, "
        stats_text += f"Люди: {self.corrections_count['person']}, "
        stats_text += f"Машины: {self.corrections_count['vehicle']}, "
        stats_text += f"Неизвестно: {self.corrections_count['unknown']}\n\n"
        
        stats_text += f"✋ РУЧНЫЕ выделения: {manual_total}\n"
        stats_text += f"   Тени: {self.manual_count['shadow']}, "
        stats_text += f"Люди: {self.manual_count['person']}, "
        stats_text += f"Машины: {self.manual_count['vehicle']}, "
        stats_text += f"Неизвестно: {self.manual_count['unknown']}\n\n"
        
        stats_text += f"🎯 ВСЕГО образцов для обучения: {corrections_total + manual_total}"
        
        if hasattr(self, 'stats_label'):
            self.stats_label.configure(text=stats_text)
    
    def apply_all_changes(self):
        """Применяет ВСЕ изменения (исправления + ручные выделения)"""
        total_changes = len(self.user_corrections) + len(self.manual_selections)
        
        if total_changes == 0:
            messagebox.showinfo("Информация", "Нет изменений для применения")
            return
        
        applied_count = 0
        
        # Применяем исправления автоматических детекций
        for correction in self.user_corrections:
            self._apply_single_correction(correction)
            applied_count += 1
        
        # Применяем ручные выделения
        for manual_sel in self.manual_selections:
            self._apply_manual_selection(manual_sel)
            applied_count += 1
        
        # Принудительно обновляем модель
        self.smart_tracker.learner.update_model()
        
        # Очищаем все
        self.user_corrections.clear()
        self.manual_selections.clear()
        
        messagebox.showinfo("Успех", 
                           f"Применено {applied_count} изменений!\n"
                           f"({len(self.user_corrections)} исправлений + "
                           f"{len(self.manual_selections)} ручных выделений)\n"
                           f"Нейронная сеть обновлена.")
        
        self.update_training_display()
    
    def _apply_single_correction(self, correction):
        """Применяет одно исправление"""
        bbox = correction['bbox']
        new_class_id = correction['new_class_id']
        frame = correction['frame']
        
        # Извлекаем область изображения
        x, y, w, h = bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        
        if x2 > x1 and y2 > y1:
            image_patch = frame[y1:y2, x1:x2]
            
            # Добавляем в систему обучения с высокой уверенностью
            self.smart_tracker.learner.add_sample(
                image_patch, bbox, frame.shape, 
                new_class_id, 0.95, true_label=new_class_id
            )
            
            # Сохраняем исправленный образец
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            class_name = correction['new_class']
            
            os.makedirs(f"training_data/{class_name}", exist_ok=True)
            filename = f"training_data/{class_name}/corrected_{timestamp}_{hash(str(bbox))}.jpg"
            cv2.imwrite(filename, image_patch)
    
    def _apply_manual_selection(self, manual_sel):
        """Применяет одно ручное выделение"""
        bbox = manual_sel['bbox']
        class_id = manual_sel['class_id']
        frame = manual_sel['frame']
        
        # Извлекаем область изображения
        x, y, w, h = bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        
        if x2 > x1 and y2 > y1:
            image_patch = frame[y1:y2, x1:x2]
            
            # Добавляем в систему обучения с максимальной уверенностью (ручная разметка)
            self.smart_tracker.learner.add_sample(
                image_patch, bbox, frame.shape, 
                class_id, 0.99, true_label=class_id
            )
            
            # Сохраняем ручное выделение
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            class_name = manual_sel['class']
            
            os.makedirs(f"training_data/{class_name}", exist_ok=True)
            filename = f"training_data/{class_name}/manual_{timestamp}_{hash(str(bbox))}.jpg"
            cv2.imwrite(filename, image_patch)
    
    def clear_all(self):
        """Очищает все изменения"""
        self.user_corrections.clear()
        self.manual_selections.clear()
        self.selected_detection = None
        
        self.selection_label.configure(text="Выберите объект и класс для исправления")
        self.manual_label.configure(text="Выберите класс и выделите область мышью")
        
        # Сбрасываем счетчики
        for key in self.corrections_count:
            self.corrections_count[key] = 0
            self.manual_count[key] = 0
            
        self.update_stats_display()
        self.update_training_display()
    
    def close_training_window(self):
        """Закрывает окно обучения"""
        if self.training_window:
            self.training_window.destroy()
            self.training_window = None
            ('Неизвестно', 'unknown', '#FF00FF')
        
        
        for i, (name, class_key, color) in enumerate(classes):
            btn = tk.Button(buttons_frame, text=name, bg=color, fg='white',
                           font=("Arial", 10, "bold"), width=12,
                           command=lambda k=class_key: self.set_correction_class(k))
            btn.grid(row=0, column=i, padx=5)
            self.class_buttons[class_key] = btn
            
        # Информация о текущем выборе
        self.selection_label = tk.Label(self.training_window, 
                                       text="Выберите объект и класс для исправления",
                                       font=("Arial", 10))
        self.selection_label.pack(pady=5)
        
        # Статистика исправлений
        stats_frame = tk.Frame(self.training_window)
        stats_frame.pack(pady=10)
        
        tk.Label(stats_frame, text="Статистика исправлений:", 
                font=("Arial", 11, "bold")).pack()
        
        self.stats_label = tk.Label(stats_frame, text="", font=("Arial", 9))
        self.stats_label.pack()
        
        # Кнопки действий
        actions_frame = tk.Frame(self.training_window)
        actions_frame.pack(pady=10)
        
        tk.Button(actions_frame, text="Применить исправления", 
                 command=self.apply_corrections, bg='#4CAF50', fg='white',
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
                 
        tk.Button(actions_frame, text="Отменить все", 
                 command=self.clear_corrections, bg='#f44336', fg='white',
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
                 
        tk.Button(actions_frame, text="Закрыть", 
                 command=self.close_training_window, bg='#2196F3', fg='white',
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Переменные для выбора
        self.selected_detection = None
        self.correction_class = None
        
        # Привязываем клик к изображению
        self.image_label.bind("<Button-1>", self.on_image_click)
        
        self.update_stats_display()
    
    def update_training_display(self):
        """Обновляет отображение кадра с детекциями"""
        if self.current_frame is None:
            return
            
        display_frame = self.current_frame.copy()
        
        # Рисуем все детекции
        for i, detection in enumerate(self.current_detections):
            bbox = detection['bbox']
            class_id = detection['class']
            confidence = detection['confidence']
            
            x, y, w, h = bbox
            
            # Цвета для классов
            colors = {0: (128, 128, 128), 1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 255)}
            color = colors.get(class_id, (255, 255, 255))
            
            # Рамка
            thickness = 3 if i == self.selected_detection else 2
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Подпись
            class_names = ['SHADOW', 'PERSON', 'VEHICLE', 'UNKNOWN']
            label = f"{class_names[class_id]} {confidence:.2f}"
            
            # Если есть исправление для этого объекта
            correction = next((c for c in self.user_corrections if c['detection_id'] == i), None)
            if correction:
                label += f" -> {correction['new_class'].upper()}"
                
            cv2.putText(display_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Номер объекта
            cv2.putText(display_frame, str(i), (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Конвертируем для отображения в tkinter
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(display_frame_rgb)
        
        # Масштабируем если нужно
        max_width, max_height = 750, 500
        if pil_image.width > max_width or pil_image.height > max_height:
            pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=self.tk_image)
        
        # Сохраняем коэффициент масштабирования
        self.scale_x = pil_image.width / self.current_frame.shape[1]
        self.scale_y = pil_image.height / self.current_frame.shape[0]
    
    def on_image_click(self, event):
        """Обработка клика по изображению"""
        if not self.current_detections:
            return
            
        # Преобразуем координаты клика обратно к исходному изображению
        click_x = int(event.x / self.scale_x)
        click_y = int(event.y / self.scale_y)
        
        # Находим ближайшую детекцию
        min_distance = float('inf')
        closest_detection = None
        
        for i, detection in enumerate(self.current_detections):
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            # Центр объекта
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Проверяем, попадает ли клик в область объекта
            if x <= click_x <= x + w and y <= click_y <= y + h:
                distance = np.sqrt((click_x - center_x)**2 + (click_y - center_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_detection = i
        
        if closest_detection is not None:
            self.selected_detection = closest_detection
            detection = self.current_detections[closest_detection]
            
            class_names = ['тень/шум', 'человек', 'машина', 'неизвестно']
            current_class = class_names[detection['class']]
            
            self.selection_label.configure(
                text=f"Выбран объект #{closest_detection} (сейчас: {current_class}). "
                     f"Выберите правильный класс."
            )
            
            self.update_training_display()
    
    def set_correction_class(self, class_key):
        """Устанавливает класс для исправления"""
        if self.selected_detection is None:
            messagebox.showwarning("Предупреждение", "Сначала выберите объект!")
            return
            
        # Удаляем предыдущее исправление для этого объекта
        self.user_corrections = [c for c in self.user_corrections 
                                if c['detection_id'] != self.selected_detection]
        
        # Добавляем новое исправление
        detection = self.current_detections[self.selected_detection]
        old_class_id = detection['class']
        
        class_mapping = {'shadow': 0, 'person': 1, 'vehicle': 2, 'unknown': 3}
        new_class_id = class_mapping[class_key]
        
        if old_class_id != new_class_id:  # Только если класс действительно изменился
            correction = {
                'detection_id': self.selected_detection,
                'bbox': detection['bbox'],
                'old_class': old_class_id,
                'new_class': class_key,
                'new_class_id': new_class_id,
                'frame': self.current_frame.copy()
            }
            
            self.user_corrections.append(correction)
            self.corrections_count[class_key] += 1
            
            class_names = ['тень/шум', 'человек', 'машина', 'неизвестно']
            self.selection_label.configure(
                text=f"Объект #{self.selected_detection} изменен: "
                     f"{class_names[old_class_id]} -> {class_key}"
            )
            
            self.update_training_display()
            self.update_stats_display()
    
    def update_stats_display(self):
        """Обновляет отображение статистики"""
        total = sum(self.corrections_count.values())
        stats_text = f"Всего исправлений: {total}\n"
        stats_text += f"Тени: {self.corrections_count['shadow']}, "
        stats_text += f"Люди: {self.corrections_count['person']}, "
        stats_text += f"Машины: {self.corrections_count['vehicle']}, "
        stats_text += f"Неизвестно: {self.corrections_count['unknown']}"
        
        if hasattr(self, 'stats_label'):
            self.stats_label.configure(text=stats_text)
    
    def apply_corrections(self):
        """Применяет исправления для обучения нейронной сети"""
        if not self.user_corrections:
            messagebox.showinfo("Информация", "Нет исправлений для применения")
            return
        
        # Отправляем исправления в систему обучения
        for correction in self.user_corrections:
            bbox = correction['bbox']
            new_class_id = correction['new_class_id']
            frame = correction['frame']
            
            # Извлекаем область изображения
            x, y, w, h = bbox
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            
            if x2 > x1 and y2 > y1:
                image_patch = frame[y1:y2, x1:x2]
                
                # Добавляем в систему обучения с высокой уверенностью
                self.smart_tracker.learner.add_sample(
                    image_patch, bbox, frame.shape, 
                    new_class_id, 0.95, true_label=new_class_id
                )
                
                # Сохраняем исправленный образец
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                class_name = correction['new_class']
                
                # Сохраняем в training_data
                os.makedirs(f"training_data/{class_name}", exist_ok=True)
                filename = f"training_data/{class_name}/corrected_{timestamp}_{len(self.user_corrections)}.jpg"
                cv2.imwrite(filename, image_patch)
        
        # Принудительно обновляем модель
        self.smart_tracker.learner.update_model()
        
        applied_count = len(self.user_corrections)
        self.user_corrections.clear()
        
        messagebox.showinfo("Успех", 
                           f"Применено {applied_count} исправлений!\n"
                           f"Нейронная сеть обновлена.")
        
        self.update_training_display()
    
    def clear_corrections(self):
        """Очищает все исправления"""
        self.user_corrections.clear()
        self.selected_detection = None
        self.selection_label.configure(text="Выберите объект и класс для исправления")
        self.update_training_display()
    
    def close_training_window(self):
        """Закрывает окно обучения"""
        if self.training_window:
            self.training_window.destroy()
            self.training_window = None

# Интеграция с основной системой
class SmartTrackerWithTraining:
    """Расширенный умный трекер с интерактивным обучением"""
    
    def __init__(self, smart_tracker):
        self.smart_tracker = smart_tracker
        self.trainer = InteractiveTrainer(smart_tracker)
        self.last_training_time = 0
        self.training_interval = 10  # показывать интерфейс каждые 10 секунд
    
    def should_show_training(self):
        """Определяет, нужно ли показать интерфейс обучения"""
        current_time = time.time()
        if current_time - self.last_training_time > self.training_interval:
            self.last_training_time = current_time
            return True
        return False
    
    def process_frame(self, frame, detections):
        """Обрабатывает кадр и показывает интерфейс обучения при необходимости"""
        # Формируем список детекций для обучения
        training_detections = []
        for tracker_data in detections:
            if 'bbox' in tracker_data:  # Проверяем что есть bbox
                detection = {
                    'bbox': tracker_data.get('bbox', (0, 0, 0, 0)),
                    'class': tracker_data.get('neural_class', 3),
                    'confidence': tracker_data.get('confidence', 0.0)
                }
                training_detections.append(detection)
        
        # Показываем интерфейс обучения если есть детекции и прошло достаточно времени
        if training_detections and self.should_show_training():
            try:
                self.trainer.show_training_interface(frame, training_detections)
            except Exception as e:
                print(f"Training interface error: {e}")
    
    def classify_object(self, frame, bbox, movement_speed=0):
        """Прокси-метод для классификации"""
        return self.smart_tracker.classify_object(frame, bbox, movement_speed)