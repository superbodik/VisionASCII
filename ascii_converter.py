import cv2

ASCII_CHARS = "@%#*+=-:. "

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def resize_image(image, new_width=100):
    height, width = image.shape
    ratio = height / width
    new_height = int(new_width * ratio * 0.55)  # Коэффициент для пропорций шрифта
    return cv2.resize(image, (new_width, new_height))

def image_to_ascii(image):
    ascii_str = ""
    for row in image:
        for pixel in row:
            ascii_str += ASCII_CHARS[int(pixel) * len(ASCII_CHARS) // 256]
        ascii_str += "\n"
    return ascii_str
