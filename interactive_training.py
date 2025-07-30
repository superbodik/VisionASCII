import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from collections import deque
import time
import os

class DynamicObjectClassifier(nn.Module):
    def __init__(self, num_classes=4):  # 0=noise/shadow, 1=person, 2=vehicle, 3=unknown
        super().__init__()
        
        # Конволюционные слои для извлечения признаков
        self.features = nn.Sequential(
            # Первый блок
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # убрал inplace=True
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # убрал inplace=True
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Второй блок
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # убрал inplace=True
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # убрал inplace=True
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
            
            # Третий блок
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # убрал inplace=True
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # убрал inplace=True
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Адаптивный пулинг для фиксированного размера
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Классификатор на основе только визуальных признаков
        self.visual_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),  # убрал inplace=True
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU()   # убрал inplace=True
        )
        
        # Дополнительные признаки (геометрические)
        self.geometry_features = nn.Sequential(
            nn.Linear(8, 32),  # x, y, w, h, aspect_ratio, area, density, position_ratio
            nn.ReLU(),  # убрал inplace=True
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()   # убрал inplace=True
        )
        
        # Финальный классификатор с объединенными признаками
        self.final_classifier = nn.Sequential(
            nn.Linear(64 + 16, 32),  # 64 визуальных + 16 геометрических = 80
            nn.ReLU(),  # убрал inplace=True
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
    def extract_geometry_features(self, bbox, frame_shape):
        """Извлекает геометрические признаки из bounding box"""
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        
        # Защита от деления на ноль
        frame_w = max(frame_w, 1)
        frame_h = max(frame_h, 1)
        h = max(h, 1)
        
        features = [
            x / frame_w,  # нормализованная x координата
            y / frame_h,  # нормализованная y координата
            w / frame_w,  # нормализованная ширина
            h / frame_h,  # нормализованная высота
            w / h,  # соотношение сторон
            (w * h) / (frame_w * frame_h),  # нормализованная площадь
            (w * h) / (w + h),  # плотность
            (y + h/2) / frame_h  # относительная позиция по вертикали
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self, image, bbox=None, frame_shape=None):
        # Извлекаем визуальные признаки
        visual_features = self.features(image)
        visual_features = visual_features.view(visual_features.size(0), -1)
        visual_output = self.visual_classifier(visual_features)
        
        # Если геометрические данные не предоставлены, используем только визуальные признаки
        if bbox is None or frame_shape is None:
            # Добавляем нулевые геометрические признаки
            batch_size = image.size(0)
            geo_features = torch.zeros(batch_size, 16, device=image.device)
        else:
            # Извлекаем геометрические признаки
            batch_size = image.size(0)
            
            if batch_size == 1:
                # Для одиночного элемента bbox является кортежем (x, y, w, h)
                try:
                    geo_features = self.extract_geometry_features(bbox, frame_shape).unsqueeze(0)
                except:
                    geo_features = torch.zeros(1, 8, device=image.device)
            else:
                # Для батча bbox должен быть списком кортежей
                geo_features_list = []
                for i in range(batch_size):
                    try:
                        if isinstance(bbox, list) and len(bbox) > i:
                            geo_feat = self.extract_geometry_features(bbox[i], frame_shape)
                        else:
                            geo_feat = torch.zeros(8, device=image.device)
                        geo_features_list.append(geo_feat)
                    except:
                        geo_features_list.append(torch.zeros(8, device=image.device))
                        
                geo_features = torch.stack(geo_features_list)
            
            geo_features = geo_features.to(image.device)
            geo_features = self.geometry_features(geo_features)
        
        # Объединяем признаки
        combined = torch.cat([visual_output, geo_features], dim=1)
        final_output = self.final_classifier(combined)
        
        return final_output

class ShadowDetector:
    """Сбалансированный детектор теней"""
    
    @staticmethod
    def is_likely_shadow(image_patch, bbox):
        """Проверяет, является ли область тенью (более консервативно)"""
        if image_patch is None or image_patch.size == 0:
            return False
            
        try:
            # Анализ размера - очень маленькие объекты скорее тени
            x, y, w, h = bbox
            area = w * h
            aspect_ratio = w / h if h > 0 else 1
            
            # Если объект слишком маленький или странной формы
            if area < 400 or aspect_ratio > 8 or aspect_ratio < 0.1:
                return True
            
            # Анализ цвета и яркости
            gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Очень темные и однородные области
            if mean_brightness < 40 and std_brightness < 15:
                return True
                
            # Анализ насыщенности
            hsv = cv2.cvtColor(image_patch, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            mean_saturation = np.mean(saturation)
            
            # Только если ВСЕ критерии указывают на тень
            is_very_dark = mean_brightness < 50
            very_low_contrast = std_brightness < 12
            very_low_saturation = mean_saturation < 15
            
            # Тень только если ВСЕ 3 критерия выполняются
            return is_very_dark and very_low_contrast and very_low_saturation
            
        except Exception as e:
            return False  # При ошибке не считаем тенью
    
    @staticmethod
    def analyze_shadow_context(frame, bbox):
        """Анализ контекста - более мягкий"""
        try:
            x, y, w, h = bbox
            area = w * h
            
            # Не применяем контекстный анализ к объектам среднего размера
            if area > 800:  # Люди обычно больше 800 пикселей
                return False
                
            frame_h, frame_w = frame.shape[:2]
            
            # Расширяем область для анализа контекста
            context_margin = 15
            x1 = max(0, x - context_margin)
            y1 = max(0, y - context_margin)
            x2 = min(frame_w, x + w + context_margin)
            y2 = min(frame_h, y + h + context_margin)
            
            if x2 <= x1 or y2 <= y1:
                return False
                
            context_region = frame[y1:y2, x1:x2]
            object_region = frame[y:y+h, x:x+w]
            
            context_brightness = np.mean(cv2.cvtColor(context_region, cv2.COLOR_BGR2GRAY))
            object_brightness = np.mean(cv2.cvtColor(object_region, cv2.COLOR_BGR2GRAY))
            
            # Только очень темные объекты относительно контекста
            brightness_ratio = object_brightness / (context_brightness + 1)
            return brightness_ratio < 0.4  # Более строгий критерий
            
        except:
            return False

class AdvancedObjectClassifier:
    """Классификатор с улучшенной логикой для точного определения"""
    
    @staticmethod
    def classify_by_rules(bbox, frame_shape, movement_speed=0):
        """Более точная классификация с четкими критериями"""
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        
        area = w * h
        aspect_ratio = w / h if h > 0 else 1
        
        # Анализ размера относительно кадра
        relative_width = w / frame_w
        relative_height = h / frame_h
        center_y = (y + h/2) / frame_h
        
        print(f"🔍 Анализ объекта: {w}x{h}px, area={area}, ratio={aspect_ratio:.2f}, speed={movement_speed:.1f}")
        
        # === ЧЕТКИЕ КРИТЕРИИ ДЛЯ МАШИН ===
        vehicle_score = 0
        vehicle_reasons = []
        
        # 1. Размер машины (строгие критерии)
        if area >= 4000:
            vehicle_score += 4
            vehicle_reasons.append(f"Large area ({area})")
        elif area >= 2500:
            vehicle_score += 2
            vehicle_reasons.append(f"Medium area ({area})")
            
        # 2. Горизонтальные пропорции машины
        if 1.5 <= aspect_ratio <= 3.5:
            vehicle_score += 4
            vehicle_reasons.append(f"Car proportions ({aspect_ratio:.2f})")
        elif 1.2 <= aspect_ratio <= 4.0:
            vehicle_score += 2
            vehicle_reasons.append(f"Wide proportions ({aspect_ratio:.2f})")
            
        # 3. Минимальные размеры машины
        if w >= 90 and h >= 60:
            vehicle_score += 3
            vehicle_reasons.append(f"Car size ({w}x{h})")
        elif w >= 70 and h >= 45:
            vehicle_score += 1
            vehicle_reasons.append(f"Medium size ({w}x{h})")
            
        # 4. Скорость машины
        if movement_speed > 4.0:
            vehicle_score += 3
            vehicle_reasons.append(f"Fast speed ({movement_speed:.1f})")
        elif movement_speed > 2.5:
            vehicle_score += 2
            vehicle_reasons.append(f"Car speed ({movement_speed:.1f})")
            
        # 5. Позиция на дороге
        if center_y >= 0.4:
            vehicle_score += 1
            vehicle_reasons.append("Road position")
            
        # === ЧЕТКИЕ КРИТЕРИИ ДЛЯ ЛЮДЕЙ ===
        person_score = 0
        person_reasons = []
        
        # 1. Размер человека
        if 800 <= area <= 6000:
            person_score += 4
            person_reasons.append(f"Person area ({area})")
        elif 500 <= area <= 10000:
            person_score += 2
            person_reasons.append(f"Human-like area ({area})")
            
        # 2. Вертикальные пропорции человека  
        if 0.4 <= aspect_ratio <= 0.8:
            person_score += 4
            person_reasons.append(f"Person proportions ({aspect_ratio:.2f})")
        elif 0.3 <= aspect_ratio <= 1.0:
            person_score += 2
            person_reasons.append(f"Vertical proportions ({aspect_ratio:.2f})")
            
        # 3. Размеры человека
        if 25 <= w <= 80 and 50 <= h <= 200:
            person_score += 3
            person_reasons.append(f"Person size ({w}x{h})")
        elif 20 <= w <= 100 and 40 <= h <= 250:
            person_score += 1
            person_reasons.append(f"Human-like size ({w}x{h})")
            
        # 4. Скорость человека
        if 0.5 <= movement_speed <= 2.5:
            person_score += 3
            person_reasons.append(f"Walking speed ({movement_speed:.1f})")
        elif 0.2 <= movement_speed <= 4.0:
            person_score += 1
            person_reasons.append(f"Human speed ({movement_speed:.1f})")
        elif movement_speed > 5.0:
            person_score -= 2
            person_reasons.append(f"Too fast for person ({movement_speed:.1f})")
            
        # === КРИТЕРИИ ДЛЯ ТЕНЕЙ ===
        shadow_score = 0
        shadow_reasons = []
        
        # 1. Очень маленькие объекты
        if area < 400:
            shadow_score += 3
            shadow_reasons.append(f"Very small ({area})")
            
        # 2. Экстремальные пропорции
        if aspect_ratio > 6 or aspect_ratio < 0.15:
            shadow_score += 3
            shadow_reasons.append(f"Extreme ratio ({aspect_ratio:.2f})")
            
        # 3. Очень узкие объекты
        if w < 15 or h < 15:
            shadow_score += 2
            shadow_reasons.append(f"Too narrow ({w}x{h})")
            
        # === ПРИНЯТИЕ РЕШЕНИЯ С ЛОГИРОВАНИЕМ ===
        
        print(f"🚗 Vehicle score: {vehicle_score} - {vehicle_reasons}")
        print(f"👤 Person score: {person_score} - {person_reasons}")
        print(f"🌑 Shadow score: {shadow_score} - {shadow_reasons}")
        
        # Четкие пороги для принятия решений
        if shadow_score >= 5:
            print(f"✅ DECISION: SHADOW (score: {shadow_score})")
            return 0, 0.8 + min(0.15, shadow_score * 0.02)
            
        elif vehicle_score >= 8:
            confidence = 0.85 + min(0.1, vehicle_score * 0.01)
            print(f"✅ DECISION: VEHICLE (score: {vehicle_score}, confidence: {confidence:.2f})")
            return 2, confidence
            
        elif person_score >= 7:
            confidence = 0.8 + min(0.15, person_score * 0.02)
            print(f"✅ DECISION: PERSON (score: {person_score}, confidence: {confidence:.2f})")
            return 1, confidence
            
        elif vehicle_score >= 5 and vehicle_score > person_score + 2:
            confidence = 0.6 + vehicle_score * 0.03
            print(f"✅ DECISION: VEHICLE (lower confidence, score: {vehicle_score})")
            return 2, confidence
            
        elif person_score >= 4 and person_score > vehicle_score + 1:
            confidence = 0.6 + person_score * 0.03
            print(f"✅ DECISION: PERSON (lower confidence, score: {person_score})")
            return 1, confidence
            
        elif shadow_score >= 2:
            print(f"✅ DECISION: SHADOW (low confidence, score: {shadow_score})")
            return 0, 0.6
            
        else:
            print(f"❓ DECISION: UNKNOWN (vehicle: {vehicle_score}, person: {person_score})")
            return 3, 0.4
    """Детектор теней и артефактов"""
    
    @staticmethod
    def is_likely_shadow(image_patch, bbox):
        """Проверяет, является ли область тенью"""
        if image_patch is None or image_patch.size == 0:
            return False
            
        # Конвертируем в различные цветовые пространства
        hsv = cv2.cvtColor(image_patch, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_patch, cv2.COLOR_BGR2LAB)
        
        # Анализ яркости
        gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Анализ насыщенности
        saturation = hsv[:, :, 1]
        mean_saturation = np.mean(saturation)
        
        # Анализ градиентов
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        mean_gradient = np.mean(gradient_magnitude)
        
        # Критерии тени
        is_dark = mean_brightness < 80  # темная область
        low_contrast = std_brightness < 20  # низкий контраст
        low_saturation = mean_saturation < 30  # низкая насыщенность
        weak_edges = mean_gradient < 15  # слабые границы
        
        # Тень, если выполняется большинство критериев
        shadow_score = sum([is_dark, low_contrast, low_saturation, weak_edges])
        return shadow_score >= 3

class DynamicLearner:
    """Система динамического обучения с мгновенными обновлениями"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Буферы для онлайн обучения (увеличены)
        self.image_buffer = deque(maxlen=200)  # Больше образцов
        self.label_buffer = deque(maxlen=200)
        self.bbox_buffer = deque(maxlen=200)
        self.frame_shape_buffer = deque(maxlen=200)
        self.confidence_buffer = deque(maxlen=200)  # Новый буфер для уверенности
        
        # Статистика обучения
        self.learning_stats = {
            'updates': 0,
            'accuracy': 0.0,
            'last_update': time.time(),
            'total_samples': 0,
            'manual_samples': 0,
            'correction_samples': 0
        }
        
        # Настройки для непрерывного обучения
        self.auto_update_threshold = 5  # Обновляем каждые 5 новых образцов
        self.confidence_threshold = 0.8
        self.samples_since_update = 0
        
        # Веса для разных типов обучения
        self.sample_weights = {
            'manual': 2.0,      # Ручные выделения важнее
            'correction': 1.5,  # Исправления важнее
            'auto': 1.0        # Автоматические предсказания
        }
        
    def add_sample(self, image_patch, bbox, frame_shape, predicted_class, confidence, true_label=None, sample_type='auto'):
        """Добавляет образец для обучения с типом образца и защитой от ошибок"""
        
        try:
            # Определяем истинную метку
            final_label = true_label if true_label is not None else predicted_class
            
            # Автоматическая маркировка высоко уверенных предсказаний
            if true_label is None and confidence > self.confidence_threshold and sample_type == 'auto':
                final_label = predicted_class
            
            # Дополнительная проверка на тени
            if true_label is None and sample_type == 'auto':
                try:
                    if ShadowDetector.is_likely_shadow(image_patch, bbox):
                        final_label = 0  # класс "шум/тень"
                except:
                    pass  # Если детектор теней не работает, продолжаем
            
            if final_label is not None:
                # Преобразование изображения с защитой от ошибок
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                if len(image_patch.shape) == 3 and image_patch.size > 0:
                    try:
                        # Клонируем image_patch для избежания inplace операций
                        image_copy = image_patch.copy()
                        image_tensor = transform(image_copy)
                        
                        # Проверяем что тензор валидный
                        if not torch.isnan(image_tensor).any() and not torch.isinf(image_tensor).any():
                            # Клонируем тензор для безопасности
                            self.image_buffer.append(image_tensor.clone())
                            self.label_buffer.append(final_label)
                            self.bbox_buffer.append(bbox)
                            self.frame_shape_buffer.append(frame_shape)
                            self.confidence_buffer.append((confidence, sample_type))
                            
                            self.learning_stats['total_samples'] += 1
                            if sample_type == 'manual':
                                self.learning_stats['manual_samples'] += 1
                            elif sample_type == 'correction':
                                self.learning_stats['correction_samples'] += 1
                            
                            self.samples_since_update += 1
                            
                            # Обучение если накопилось достаточно данных или это важный образец
                            should_update = (
                                self.samples_since_update >= self.auto_update_threshold or
                                sample_type in ['manual', 'correction'] or
                                (len(self.image_buffer) >= 10 and time.time() - self.learning_stats['last_update'] > 3.0)
                            )
                            
                            if should_update:
                                self.update_model()
                        else:
                            print("⚠️ Invalid tensor detected, skipping sample")
                            
                    except Exception as transform_error:
                        print(f"⚠️ Error transforming image: {transform_error}")
                        
        except Exception as e:
            print(f"⚠️ Error adding sample: {e}")
            # Очищаем CUDA кэш при ошибках
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def update_model(self):
        """Обновляет модель на накопленных данных с защитой от градиентных ошибок"""
        if len(self.image_buffer) < 5:
            return
        
        try:
            self.model.train()
            
            # Подготовка батча с учетом важности образцов
            available_samples = len(self.image_buffer)
            batch_size = min(16, available_samples)
            
            # Создаем веса для выборки (важные образцы чаще попадают в батч)
            sample_weights_list = []
            for i in range(available_samples):
                confidence, sample_type = self.confidence_buffer[i]
                weight = self.sample_weights.get(sample_type, 1.0)
                
                # Увеличиваем вес для высокоуверенных образцов
                if confidence > 0.9:
                    weight *= 1.2
                elif confidence > 0.8:
                    weight *= 1.1
                    
                sample_weights_list.append(weight)
            
            # Нормализуем веса
            total_weight = sum(sample_weights_list)
            if total_weight > 0:
                sample_probs = [w / total_weight for w in sample_weights_list]
            else:
                sample_probs = [1.0 / available_samples] * available_samples
            
            # Выбираем образцы с учетом весов
            try:
                indices = np.random.choice(available_samples, batch_size, 
                                         replace=False, p=sample_probs)
            except:
                indices = np.random.choice(available_samples, batch_size, replace=False)
            
            # Создаем новые тензоры для избежания inplace операций
            images = torch.stack([self.image_buffer[i].clone() for i in indices]).to(self.device)
            labels = torch.tensor([self.label_buffer[i] for i in indices], dtype=torch.long).to(self.device)
            
            # Для батча создаем список bbox'ов и используем одну frame_shape
            bboxes = [self.bbox_buffer[i] for i in indices]
            frame_shapes = self.frame_shape_buffer[indices[0]]  # Используем первую frame_shape
            
            # Очищаем градиенты перед обучением
            self.optimizer.zero_grad()
            
            # Обучение с несколькими эпохами для важных образцов
            num_epochs = 1
            manual_or_correction_in_batch = any(
                self.confidence_buffer[i][1] in ['manual', 'correction'] for i in indices
            )
            if manual_or_correction_in_batch:
                num_epochs = 2  # Больше эпох для важных данных
            
            total_loss = 0
            for epoch in range(num_epochs):
                # Обязательно очищаем градиенты каждую эпоху
                self.optimizer.zero_grad()
                
                # Форвард пасс
                outputs = self.model(images, bboxes, frame_shapes)
                loss = self.criterion(outputs, labels)
                
                # Применяем веса к loss для важных образцов
                if manual_or_correction_in_batch:
                    loss = loss * 1.2  # Увеличиваем loss для важных образцов
                
                # Обратное распространение
                loss.backward()
                
                # Градиентное клиппинг для стабильности
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Обновление весов
                self.optimizer.step()
                total_loss += loss.item()
                
                # Очищаем кэш CUDA если доступен
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Обновление статистики
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == labels).float().mean().item()
                
            self.learning_stats['updates'] += 1
            self.learning_stats['accuracy'] = accuracy
            self.learning_stats['last_update'] = time.time()
            self.samples_since_update = 0
            
            avg_loss = total_loss / num_epochs
            print(f"🧠 Model updated: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, "
                  f"Updates={self.learning_stats['updates']}, "
                  f"Samples={self.learning_stats['total_samples']} "
                  f"(M:{self.learning_stats['manual_samples']}, "
                  f"C:{self.learning_stats['correction_samples']})")
            
            self.model.eval()
            
        except RuntimeError as e:
            if "inplace operation" in str(e):
                print("⚠️ Gradient error detected, skipping this update...")
                # Очищаем градиенты и продолжаем
                self.optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model.eval()
            else:
                print(f"❌ Training error: {e}")
                self.model.eval()
        except Exception as e:
            print(f"❌ Unexpected training error: {e}")
            self.model.eval()
    
    def force_update(self):
        """Принудительное обновление модели"""
        if len(self.image_buffer) > 0:
            print("🔄 Принудительное обновление модели...")
            self.update_model()
        else:
            print("⚠️ Нет данных для обновления модели")
    
    def get_learning_progress(self):
        """Возвращает прогресс обучения"""
        if self.learning_stats['total_samples'] == 0:
            return "Обучение не начато"
        
        manual_pct = (self.learning_stats['manual_samples'] / self.learning_stats['total_samples']) * 100
        correction_pct = (self.learning_stats['correction_samples'] / self.learning_stats['total_samples']) * 100
        
        return {
            'total_samples': self.learning_stats['total_samples'],
            'accuracy': self.learning_stats['accuracy'],
            'updates': self.learning_stats['updates'],
            'manual_percentage': manual_pct,
            'correction_percentage': correction_pct,
            'buffer_size': len(self.image_buffer)
        }

class SmartObjectTracker:
    """Умный трекер с нейронной сетью"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = DynamicObjectClassifier().to(device)
        self.learner = DynamicLearner(self.model, device)
        
        # Загружаем предобученную модель если есть
        model_path = 'smart_tracker_model.pth'
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print("Loaded pretrained model")
            except:
                print("Could not load pretrained model, using random weights")
        
        self.model.eval()
        
        # Преобразования для изображений
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Маппинг классов
        self.class_names = ['shadow/noise', 'person', 'vehicle', 'unknown']
        self.class_colors = [
            (128, 128, 128),  # серый для теней/шума
            (255, 0, 0),      # красный для людей
            (0, 255, 0),      # зеленый для машин
            (0, 0, 255)       # синий для неизвестных
        ]
    
    def classify_object(self, frame, bbox, movement_speed=0):
        """Классифицирует объект с улучшенной логикой приоритетов"""
        try:
            x, y, w, h = bbox
            
            # Извлекаем область изображения
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            
            if x2 <= x1 or y2 <= y1:
                return 0, 0.0  # шум
                
            image_patch = frame[y1:y2, x1:x2]
            
            if image_patch.size == 0:
                return 0, 0.0
            
            print(f"\n🎯 Классификация объекта: {w}x{h}px, скорость: {movement_speed:.1f}")
            
            # 1. БЫСТРАЯ ПРОВЕРКА НА ЯВНЫЕ ТЕНИ
            if ShadowDetector.is_likely_shadow(image_patch, bbox):
                print("✅ Быстрое определение: ТЕНЬ (детектор)")
                self.learner.add_sample(image_patch, bbox, frame.shape, 0, 0.95, true_label=0, sample_type='auto')
                return 0, 0.95
            
            # 2. КОНТЕКСТНЫЙ АНАЛИЗ
            if ShadowDetector.analyze_shadow_context(frame, bbox):
                print("✅ Контекстное определение: ТЕНЬ")
                self.learner.add_sample(image_patch, bbox, frame.shape, 0, 0.85, true_label=0, sample_type='auto')
                return 0, 0.85
            
            # 3. ОСНОВНАЯ КЛАССИФИКАЦИЯ ПО ПРАВИЛАМ
            rule_class, rule_confidence = AdvancedObjectClassifier.classify_by_rules(
                bbox, frame.shape, movement_speed
            )
            
            # 4. НЕЙРОННАЯ СЕТЬ ДЛЯ ДОПОЛНИТЕЛЬНОЙ ПРОВЕРКИ
            neural_class, neural_confidence = 3, 0.1
            
            try:
                image_tensor = self.transform(image_patch).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(image_tensor, bbox, frame.shape)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    neural_class = predicted.item()
                    neural_confidence = confidence.item()
                    
                class_names = ['SHADOW', 'PERSON', 'VEHICLE', 'UNKNOWN']
                print(f"🤖 Нейронка: {class_names[neural_class]} ({neural_confidence:.2f})")
                    
            except Exception as e:
                print(f"⚠️ Ошибка нейронки: {e}")
                neural_class, neural_confidence = 3, 0.1
            
            # 5. УМНАЯ КОМБИНАЦИЯ РЕЗУЛЬТАТОВ
            
            # Если правила уверенно определили класс - доверяем им
            if rule_confidence > 0.8:
                final_class = rule_class
                final_confidence = rule_confidence
                print(f"🎯 ФИНАЛЬНОЕ РЕШЕНИЕ: Доверяем правилам")
                
            # Если правила и нейронка согласны - объединяем уверенность
            elif rule_class == neural_class and rule_confidence > 0.6:
                final_class = rule_class
                final_confidence = min(0.95, (rule_confidence + neural_confidence) / 2 + 0.1)
                print(f"🎯 ФИНАЛЬНОЕ РЕШЕНИЕ: Согласие между методами")
                
            # Если нейронка очень уверена, а правила сомневаются
            elif neural_confidence > 0.85 and rule_confidence < 0.7:
                final_class = neural_class
                final_confidence = neural_confidence * 0.9
                print(f"🎯 ФИНАЛЬНОЕ РЕШЕНИЕ: Доверяем нейронке (высокая уверенность)")
                
            # Разрешение конфликтов по приоритетам
            elif rule_class != neural_class:
                print(f"⚡ КОНФЛИКТ: Правила={class_names[rule_class]}({rule_confidence:.2f}) vs Нейронка={class_names[neural_class]}({neural_confidence:.2f})")
                
                # Приоритет 1: Тени (безопасность превыше всего)
                if rule_class == 0 and rule_confidence > 0.6:
                    final_class = 0
                    final_confidence = rule_confidence
                    print("  → Приоритет теням (безопасность)")
                elif neural_class == 0 and neural_confidence > 0.7:
                    final_class = 0
                    final_confidence = neural_confidence * 0.9
                    print("  → Нейронка уверена в тени")
                
                # Приоритет 2: Машины vs Люди (более строго к машинам)
                elif rule_class == 2 and rule_confidence > 0.7:  # правила говорят машина
                    final_class = 2
                    final_confidence = rule_confidence
                    print("  → Правила уверены в машине")
                elif neural_class == 2 and neural_confidence > 0.8:  # нейронка уверена в машине
                    final_class = 2
                    final_confidence = neural_confidence * 0.85
                    print("  → Нейронка уверена в машине")
                
                # Приоритет 3: Люди (более мягкие критерии)
                elif rule_class == 1 and rule_confidence > 0.6:
                    final_class = 1
                    final_confidence = rule_confidence
                    print("  → Правила говорят человек")
                elif neural_class == 1 and neural_confidence > 0.7:
                    final_class = 1
                    final_confidence = neural_confidence * 0.85
                    print("  → Нейронка говорит человек")
                
                # По умолчанию - выбираем более уверенный метод
                else:
                    if rule_confidence > neural_confidence:
                        final_class = rule_class
                        final_confidence = rule_confidence * 0.8
                        print("  → Выбираем правила (выше уверенность)")
                    else:
                        final_class = neural_class
                        final_confidence = neural_confidence * 0.8
                        print("  → Выбираем нейронку (выше уверенность)")
            
            # Если правила дали низкую уверенность
            else:
                final_class = rule_class
                final_confidence = rule_confidence
                print(f"🎯 ФИНАЛЬНОЕ РЕШЕНИЕ: По умолчанию правила")
            
            # 6. ФИНАЛЬНАЯ ПРОВЕРКА НА АРТЕФАКТЫ
            area = w * h
            aspect_ratio = w / h if h > 0 else 1
            
            if (area < 400 or aspect_ratio > 8 or aspect_ratio < 0.12) and final_class != 0:
                print("🚫 Принудительная фильтрация: слишком маленький/странный")
                final_class = 0
                final_confidence = 0.8
            
            # Логируем финальное решение
            final_class_name = ['ТЕНЬ', 'ЧЕЛОВЕК', 'МАШИНА', 'НЕИЗВЕСТНО'][final_class]
            print(f"🏁 ИТОГ: {final_class_name} (уверенность: {final_confidence:.2f})")
            
            # Добавляем для обучения только уверенные предсказания
            if final_confidence > 0.75 or final_class == 0:
                self.learner.add_sample(image_patch, bbox, frame.shape, 
                                      final_class, final_confidence, sample_type='auto')
            
            return final_class, final_confidence
            
        except Exception as e:
            print(f"❌ Ошибка классификации: {e}")
            return 0, 0.7
    
    def save_model(self):
        """Сохраняет модель"""
        torch.save(self.model.state_dict(), 'smart_tracker_model.pth')
        print("Model saved")
    
    def get_stats(self):
        """Возвращает статистику обучения"""
        return self.learner.learning_stats