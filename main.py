from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cv2
import numpy as np
import pandas as pd
import re
import io

from core.utils import (
    load_and_preprocess_image, check_image_quality, 
    fix_ocr_class_name, remove_stamps_signatures
)
from core.ai_engine import (
    load_models, extract_student_info, count_valid_subjects, 
    check_garbage_table, clean_and_fix_table, clean_scores,
    validate_ocr_results
)

app = FastAPI(title="SOTA OCR Học Bạ API")

# Cho phép Web kết nối API không bị chặn CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tải các mô hình AI khi khởi động server
print("[INFO] Đang tải các mô hình AI...")
yolo_model, table_engine, forced_engine, text_engine, vietocr_engine = load_models()
print("[INFO] Tải mô hình hoàn tất!")

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.get("/config.js")
async def serve_config():
    return FileResponse("config.js")

@app.post("/api/process")
async def process_student_record(file: UploadFile = File(...)):
    try:
        # 1. Đọc file ảnh từ Request
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_cv_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv_bgr is None:
            return {"success": False, "message": "File ảnh không hợp lệ."}

        # 2. Kiểm tra chất lượng ảnh
        is_blurry, blur_score, is_skewed, skew_angle = check_image_quality(img_cv_bgr)
        if is_blurry or is_skewed:
            return {
                "success": False, 
                "message": "Ảnh mờ hoặc quá nghiêng. Vui lòng chụp thẳng và rõ nét hơn.",
                "quality": {"is_blurry": is_blurry, "is_skewed": is_skewed}
            }

        # 3. Nhận diện YOLO
        results = yolo_model(img_cv_bgr)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return {"success": False, "message": "Không tìm thấy bảng điểm hoặc thông tin trong ảnh."}

        info_boxes = [b for b in boxes if yolo_model.names[int(b.cls[0])] == "thong_tin"]
        table_boxes = [b for b in boxes if yolo_model.names[int(b.cls[0])] == "bang_diem"]

        student_info = {"ho_ten": "", "lop": "", "nam_hoc": "", "ban": ""}

        # 4. Trích xuất thông tin cá nhân
        if info_boxes:
            best_info_box = sorted(info_boxes, key=lambda b: b.xyxy[0][1])[0]
            x1, y1, x2, y2 = map(int, best_info_box.xyxy[0])
            cropped_info_bgr = img_cv_bgr[y1:y2, x1:x2]
            
            ext_info = extract_student_info(cropped_info_bgr, text_engine, vietocr_engine)
            student_info = {
                "ho_ten": ext_info.get("ho_ten", ""),
                "lop": fix_ocr_class_name(ext_info.get("lop", "")),
                "nam_hoc": ext_info.get("nam_hoc", ""),
                "ban": ext_info.get("ban", "")
            }

        # 5. Trích xuất bảng điểm
        table_data = []
        warnings = []
        
        if table_boxes:
            best_box = max(table_boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cropped_table_bgr = img_cv_bgr[y1:y2, x1:x2]
            cleaned_bgr = remove_stamps_signatures(cropped_table_bgr)

            h_crop, w_crop = cleaned_bgr.shape[:2]
            pad_h, pad_w = max(int(h_crop * 0.05), 15), max(int(w_crop * 0.05), 15)
            padded_bgr = cv2.copyMakeBorder(cleaned_bgr, top=pad_h, bottom=pad_h, left=pad_w, right=pad_w, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

            def run_ocr(target_img):
                res = table_engine(target_img)
                for region in res:
                    if region.get('type', '').lower() == 'table' and 'res' in region and 'html' in region['res']:
                        return True, region['res']['html']
                forced_res = forced_engine(target_img)
                for region in forced_res:
                    if 'res' in region and 'html' in region['res']:
                        return True, region['res']['html']
                return False, ""

            table_found, html_code = run_ocr(padded_bgr)
            
            if table_found and html_code:
                df_raw = pd.read_html(html_code)[0]
                if count_valid_subjects(df_raw) < 3:
                    rotated_bgr = cv2.rotate(padded_bgr, cv2.ROTATE_180)
                    t_found_rot, html_rot = run_ocr(rotated_bgr)
                    if t_found_rot and html_rot:
                        df_rot = pd.read_html(html_rot)[0]
                        if count_valid_subjects(df_rot) >= 3:
                            df_raw = df_rot

                df_cleaned = clean_scores(clean_and_fix_table(df_raw))
                is_garbage, reason = check_garbage_table(df_cleaned)

                if is_garbage:
                    return {"success": False, "message": f"Bảng điểm bị mờ hoặc không thể căn chỉnh cột ({reason})."}

                # Kích hoạt rà soát toàn diện và thêm vào danh sách warnings gửi cho Frontend
                ocr_warnings = validate_ocr_results(student_info, df_cleaned)
                warnings.extend(ocr_warnings)

                # Kiểm tra logic chéo (Cross-validation) tính điểm trung bình
                if df_cleaned.shape[1] >= 4:
                    for idx, row in df_cleaned.iterrows():
                        try:
                            subj = str(row[0]).strip()
                            if subj and subj != "nan" and subj != "ĐTB các môn học":
                                hk1 = float(row[1]) if str(row[1]).replace('.','').isdigit() else None
                                hk2 = float(row[2]) if str(row[2]).replace('.','').isdigit() else None
                                tbm = float(row[3]) if str(row[3]).replace('.','').isdigit() else None
                                if hk1 is not None and hk2 is not None and tbm is not None:
                                    calc_tbm = round((hk1 + hk2 * 2) / 3, 1)
                                    if abs(calc_tbm - tbm) > 0.1:
                                        warnings.append(f"🧮 **Sai lệch tính toán**: Môn {subj} AI đọc ĐTB là {tbm} nhưng tính từ HK1 và HK2 ra {calc_tbm}")
                        except:
                            continue

                # Chuyển DataFrame thành list dictionary trả về cho Web
                headers = [str(c) for c in df_cleaned.columns]
                rows = df_cleaned.fillna("").values.tolist()
                table_data = {"headers": headers, "rows": rows}

        return {
            "success": True,
            "student_info": student_info,
            "table_data": table_data,
            "warnings": warnings
        }

    except Exception as e:
        return {"success": False, "message": f"Lỗi máy chủ: {str(e)}"}