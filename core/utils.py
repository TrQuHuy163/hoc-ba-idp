# core/utils.py
import cv2
import numpy as np
import re
import unicodedata
from PIL import Image, ImageOps
from core.config import NAME_DICTIONARY

def remove_accents(input_str):
    if not input_str:
        return ""
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s

def order_points(pts):
    """
    Sắp xếp 4 tọa độ góc theo thứ tự chuẩn:
    0: Trái-trên (Top-Left), 1: Phải-trên (Top-Right), 
    2: Phải-dưới (Bottom-Right), 3: Trái-dưới (Bottom-Left)
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Tổng (x + y): Trái-trên có tổng nhỏ nhất, Phải-dưới có tổng lớn nhất
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Hiệu (y - x): Phải-trên có hiệu nhỏ nhất, Trái-dưới có hiệu lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def unwarp_document(img_bgr):
    """
    Tự động tìm khung viền tờ giấy bị nghiêng/méo và nắn phẳng lại (Perspective Transform)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)
    
    # Tìm các đường viền (contours)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    doc_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # Nếu đường viền có đúng 4 góc thì khả năng cao là mép tờ giấy
        if len(approx) == 4:
            doc_cnt = approx
            break 

    if doc_cnt is not None:
        pts = doc_cnt.reshape(4, 2)
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Tính chiều rộng mới cho ảnh sau khi kéo phẳng
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # Tính chiều cao mới cho ảnh sau khi kéo phẳng
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Tọa độ khung ảnh phẳng đích
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        # Tính ma trận biến đổi góc nhìn và thực hiện kéo phẳng
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img_bgr, M, (maxWidth, maxHeight))
        return warped

    # Nếu không tìm thấy đủ 4 góc (ví dụ chụp quá sát viền), trả lại ảnh gốc
    return img_bgr

def load_and_preprocess_image(uploaded_file, max_dim=1920):
    image_pil = Image.open(uploaded_file)
    image_pil = ImageOps.exif_transpose(image_pil)
    image_pil = image_pil.convert('RGB')
    
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # 1. Tự động làm phẳng ảnh nếu ảnh bị nghiêng/méo
    img_bgr = unwarp_document(img_bgr)
    
    # 2. Resize ảnh nếu kích thước quá lớn để tăng tốc độ xử lý AI
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
    image_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        
    return img_bgr, image_pil

def fix_ocr_class_name(class_name):
    if not class_name:
        return ""
    c = str(class_name).strip().upper()
    c = re.sub(r'^(1[012][A-Z])S$', r'\g<1>5', c)
    c = re.sub(r'^(1[012])8(\d*)$', r'\1B\2', c)
    c = re.sub(r'^(1[012])0(\d*)$', r'\1D\2', c)
    return c

def check_image_quality(img_bgr, blur_threshold=60):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = blur_score < blur_threshold
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    is_skewed = False
    skew_angle = 0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angle = abs(angle) % 90
            if angle > 45: 
                angle = 90 - angle
            angles.append(angle)
            
        if angles:
            skew_angle = np.mean(angles)
            if skew_angle > 10.0:
                is_skewed = True

    return is_blurry, blur_score, is_skewed, skew_angle

def remove_stamps_signatures(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 50, 50]), np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    lower_blue, upper_blue = np.array([100, 50, 50]), np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    mask_combined = mask_red1 | mask_red2 | mask_blue
    kernel = np.ones((3,3), np.uint8)
    mask_combined = cv2.dilate(mask_combined, kernel, iterations=1)
    
    cleaned_img = img_bgr.copy()
    cleaned_img[mask_combined > 0] = (255, 255, 255)
    return cleaned_img

def fix_vietnamese_name(name_str):
    if not name_str:
        return name_str
        
    words = name_str.split()
    fixed_words = []
    for w in words:
        norm_w = remove_accents(w)
        if norm_w in NAME_DICTIONARY:
            fixed_words.append(NAME_DICTIONARY[norm_w])
        else:
            fixed_words.append(w.capitalize())
            
    return " ".join(fixed_words)