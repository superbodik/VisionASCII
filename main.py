import threading
import queue
import time
import os

import cv2
import numpy as np
import tkinter as tk
from gui import VisionGUI
from camera import open_camera, get_frame, close_camera, apply_zoom
from neural_classifier import  SmartTrackerWithTraining
from interactive_training import SmartObjectTracker

frame_queue = queue.Queue(maxsize=10)
stop_flag = threading.Event()

last_time = time.time()
frame_count = 0

# Умный трекер с нейронной сетью и интерактивным обучением
smart_tracker = SmartObjectTracker()
trainer_system = SmartTrackerWithTraining(smart_tracker)

# Список трекеров
trackers = []
max_trackers = 8
next_tracker_id = 1

floor = 3
max_zoom = 5.0
prev_gray = None

def detect_motion(prev_gray, current_gray, threshold=5000):
    diff = cv2.absdiff(prev_gray, current_gray)
    changed_pixels = np.count_nonzero(diff > 25)
    return changed_pixels > threshold

def create_tracker():
    """Создает новый трекер"""
    try:
        return cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        return cv2.TrackerCSRT_create()

def calculate_distance(box1, box2):
    """Вычисляет расстояние между центрами двух прямоугольников"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    center1 = (x1 + w1//2, y1 + h1//2)
    center2 = (x2 + w2//2, y2 + h2//2)
    
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def boxes_overlap_iou(box1, box2, threshold=0.25):
    """Проверяет пересечение по IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    iou = intersection / union if union > 0 else 0
    return iou > threshold

def is_too_close_to_existing(new_box, existing_boxes, min_distance=80):
    """Проверяет близость к существующим объектам"""
    for existing_box in existing_boxes:
        if calculate_distance(new_box, existing_box) < min_distance:
            return True
        if boxes_overlap_iou(new_box, existing_box, threshold=0.2):
            return True
    return False

def get_display_info(class_id, confidence):
    """Возвращает информацию для отображения класса"""
    class_names = ['SHADOW', 'PERSON', 'VEHICLE', 'UNKNOWN']
    class_colors = [
        (128, 128, 128),  # серый для теней
        (0, 0, 255),      # красный для людей  
        (0, 255, 0),      # зеленый для машин
        (255, 0, 255)     # пурпурный для неизвестных
    ]
    
    name = class_names[class_id] if 0 <= class_id < len(class_names) else 'UNKNOWN'
    color = class_colors[class_id] if 0 <= class_id < len(class_colors) else (255, 255, 255)
    
    return name, color

def capture_thread_func(gui):
    global last_time, frame_count, trackers, prev_gray, next_tracker_id, smart_tracker

    cap = open_camera()
    os.makedirs("snapshots", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)

    # Счетчик для периодического сохранения модели
    save_counter = 0

    while not stop_flag.is_set():
        frame = get_frame(cap)
        if frame is None:
            time.sleep(0.01)
            continue

        zoom_factor = gui.cached_zoom
        motion_threshold = gui.cached_motion_threshold
        save_enabled = gui.cached_save_enabled

        frame = apply_zoom(frame, zoom_factor)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        if prev_gray is None:
            prev_gray = blurred
            continue

        # Обновляем существующие трекеры
        active_trackers = []
        active_boxes = []
        
        for tracker_data in trackers:
            tracker = tracker_data['tracker']
            tracker_id = tracker_data['id']
            neural_class = tracker_data['neural_class']
            confidence = tracker_data['confidence']
            position_history = tracker_data['history']
            fail_count = tracker_data.get('fail_count', 0)
            
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                current_box = (x, y, w, h)
                
                # Периодически переклассифицируем объект (каждые 30 кадров)
                if frame_count % 30 == 0:
                    # Вычисляем текущую скорость для переклассификации
                    current_speed = 0
                    if len(position_history) >= 2:
                        dx = position_history[-1][0] - position_history[-2][0]
                        dy = position_history[-1][1] - position_history[-2][1]
                        current_speed = np.sqrt(dx*dx + dy*dy)
                    
                    new_class, new_confidence = trainer_system.classify_object(frame, current_box, current_speed)
                    # Обновляем класс с более мягким порогом
                    if new_confidence > 0.4:
                        neural_class = new_class
                        confidence = new_confidence
                        tracker_data['neural_class'] = neural_class
                        tracker_data['confidence'] = confidence
                
                # Игнорируем тени и шум (класс 0)
                if neural_class == 0:  # shadow/noise
                    continue
                
                # Проверяем размер объекта
                if w * h >= 200 and w >= 10 and h >= 10:
                    center = (x + w // 2, y + h // 2)
                    
                    # Обновляем историю позиций
                    position_history.append(center)
                    if len(position_history) > 10:
                        position_history.pop(0)
                    
                    # Проверяем движение (разные пороги для разных классов)
                    movement_threshold = {
                        1: 0.5,  # person - очень низкий порог
                        2: 1.2,  # vehicle - средний порог
                        3: 0.8   # unknown - низкий порог
                    }.get(neural_class, 0.8)
                    
                    is_moving = False
                    if len(position_history) >= 3:
                        recent_movements = []
                        for i in range(max(1, len(position_history)-3), len(position_history)):
                            dx = position_history[i][0] - position_history[i-1][0]
                            dy = position_history[i][1] - position_history[i-1][1]
                            movement = np.sqrt(dx*dx + dy*dy)
                            recent_movements.append(movement)
                        
                        avg_movement = np.mean(recent_movements)
                        is_moving = avg_movement > movement_threshold
                    
                    if is_moving:
                        # Вычисляем скорость
                        if len(position_history) >= 2:
                            dx = position_history[-1][0] - position_history[-2][0]
                            dy = position_history[-1][1] - position_history[-2][1]
                            speed = np.sqrt(dx*dx + dy*dy)
                        else:
                            speed = 0
                        
                        # Получаем информацию для отображения
                        class_name, color = get_display_info(neural_class, confidence)
                        
                        # Толщина рамки зависит от уверенности
                        thickness = max(1, int(confidence * 4))
                        
                        # Рисуем трекер
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                        
                        # Подпись с классом, ID, скоростью и уверенностью
                        label = f"{class_name}-{tracker_id} S:{speed:.1f} C:{confidence:.2f}"
                        cv2.putText(frame, label, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Сохраняем активный трекер
                        tracker_data['history'] = position_history
                        tracker_data['fail_count'] = 0
                        active_trackers.append(tracker_data)
                        active_boxes.append(current_box)
                    else:
                        # Объект не движется
                        fail_count += 1
                        max_static_frames = 25 if neural_class == 2 else 20  # больше шансов для машин
                        if fail_count < max_static_frames:
                            tracker_data['fail_count'] = fail_count
                            active_trackers.append(tracker_data)
                else:
                    # Неподходящий размер
                    pass
            else:
                # Трекер потерял объект
                fail_count += 1
                if fail_count < 10:
                    tracker_data['fail_count'] = fail_count
                    active_trackers.append(tracker_data)

        # Обновляем список трекеров
        trackers = active_trackers

        # Ищем новые объекты
        if len(trackers) < max_trackers and detect_motion(prev_gray, blurred, threshold=motion_threshold):
            diff = cv2.absdiff(prev_gray, blurred)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Морфологические операции
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            new_trackers_added = 0
            for c in contours:
                if new_trackers_added >= 2:
                    break
                    
                area = cv2.contourArea(c)
                if area > 250:  # еще больше снижена минимальная площадь
                    x, y, w, h = cv2.boundingRect(c)
                    new_box = (x, y, w, h)
                    
                    # Классифицируем объект с помощью нейронной сети
                    neural_class, confidence = trainer_system.classify_object(frame, new_box, movement_speed=0)
                    
                    # Более мягкие критерии принятия объектов
                    if neural_class != 0 and confidence > 0.25:  # значительно снижен порог
                        # Проверяем близость к существующим
                        min_distance = 100 if neural_class == 2 else 60  # расстояния для машин и людей
                        
                        if not is_too_close_to_existing(new_box, active_boxes, min_distance):
                            # Создаем новый трекер
                            new_tracker = create_tracker()
                            if new_tracker.init(frame, (x, y, w, h)):
                                center = (x + w // 2, y + h // 2)
                                tracker_data = {
                                    'tracker': new_tracker,
                                    'history': [center],
                                    'id': next_tracker_id,
                                    'neural_class': neural_class,
                                    'confidence': confidence,
                                    'fail_count': 0
                                }
                                trackers.append(tracker_data)
                                next_tracker_id += 1
                                new_trackers_added += 1
                                
                                # Сохраняем образец для дальнейшего обучения
                                if save_enabled:
                                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                                    microseconds = str(int(time.time() * 1000000) % 1000000)
                                    full_timestamp = f"{timestamp}_{microseconds}"
                                    class_name = ['shadow', 'person', 'vehicle', 'unknown'][neural_class]
                                    
                                    # Сохраняем весь кадр в snapshots
                                    cv2.imwrite(f"snapshots/{class_name}_{full_timestamp}.jpg", frame)
                                    
                                    # Сохраняем вырезанную область для обучения в training_data
                                    x1, y1 = max(0, x), max(0, y)
                                    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                                    crop = frame[y1:y2, x1:x2]
                                    if crop.size > 0:
                                        cv2.imwrite(f"training_data/{class_name}_{full_timestamp}_crop.jpg", crop)

        prev_gray = blurred

        # Отображаем статистику
        class_counts = [0, 0, 0, 0]  # shadow, person, vehicle, unknown
        for t in trackers:
            class_id = t['neural_class']
            if 0 <= class_id < len(class_counts):
                class_counts[class_id] += 1
        
        # Отображаем статистику обучения
        learning_stats = trainer_system.smart_tracker.get_stats()
        
        # Интерактивное обучение - показываем интерфейс при необходимости
        try:
            # Подготавливаем данные о текущих трекерах для обучения
            current_detections = []
            for tracker_data in trackers:
                if 'neural_class' in tracker_data and 'confidence' in tracker_data:
                    # Получаем текущий bbox из трекера
                    tracker = tracker_data['tracker']
                    success, box = tracker.update(frame.copy())
                    if success:
                        x, y, w, h = [int(v) for v in box]
                        detection_data = {
                            'bbox': (x, y, w, h),
                            'neural_class': tracker_data['neural_class'],
                            'confidence': tracker_data['confidence']
                        }
                        current_detections.append(detection_data)
            
            # Обрабатываем кадр в системе обучения
            trainer_system.process_frame(frame, current_detections)
        except Exception as e:
            print(f"Training system error: {e}")
        
        # Статистика обучения
        
        # Отображаем информацию на экране
        info_lines = [
            f"People: {class_counts[1]} | Vehicles: {class_counts[2]} | Unknown: {class_counts[3]}",
            f"Shadows filtered: {class_counts[0]} | Learning updates: {learning_stats['updates']}",
            f"Model accuracy: {learning_stats['accuracy']:.3f} | Confidence threshold: 0.5+"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 25 + i * 25
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Умное сохранение модели с улучшениями
        save_counter += 1
        
        # Сохраняем чаще если модель активно обучается
        try:
            progress = trainer_system.smart_tracker.learner.get_learning_progress()
            
            # Адаптивная частота сохранения
            if progress['manual_samples'] + progress['correction_samples'] > 0:
                # Если есть ручные данные - сохраняем чаще
                save_interval = 200
            else:
                # Обычная частота
                save_interval = 500
                
            if save_counter % save_interval == 0:
                trainer_system.smart_tracker.save_model()
                print(f"🔄 Model auto-saved: {progress['total_samples']} samples, "
                      f"{progress['accuracy']:.1%} accuracy")
                      
        except Exception as e:
            # Fallback к обычному сохранению
            if save_counter % 500 == 0:
                try:
                    trainer_system.smart_tracker.save_model()
                except Exception as save_error:
                    print(f"Error saving model: {save_error}")

        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass

        frame_count += 1
        now_time = time.time()
        if now_time - last_time >= 1.0:
            fps = frame_count / (now_time - last_time)
            frame_count = 0
            last_time = now_time
            if gui:
                gui.root.after(0, gui.update_fps, fps)

        time.sleep(0.008)

    # Сохраняем модель при завершении
    try:
        trainer_system.smart_tracker.save_model()
        print("Model saved on exit")
    except Exception as e:
        print(f"Error saving model on exit: {e}")

    close_camera(cap)

def main():
    print("🚀 Starting Smart AI Tracker with Dynamic Learning...")
    print("📚 Classes: SHADOW (filtered) | PERSON (red) | VEHICLE (green) | UNKNOWN (purple)")
    print("🧠 The system learns from your corrections and improves in real-time!")
    print("🎯 Interactive training window appears every 10 seconds")
    print("💡 Click objects to correct them, or draw new areas to train on")
    
    root = tk.Tk()
    root.title("Smart AI Tracker - Dynamic Learning System")
    
    gui = VisionGUI(root, frame_queue)

    t = threading.Thread(target=capture_thread_func, args=(gui,), daemon=True)
    t.start()

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("🛑 Shutting down...")
    finally:
        stop_flag.set()
        t.join()
        print("✅ System shutdown complete")

if __name__ == "__main__":
    main()