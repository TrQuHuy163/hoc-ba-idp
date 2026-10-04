# core/ai_engine.py
import cv2
import re
import numpy as np
import pandas as pd
import streamlit as st
import difflib
from PIL import Image
from rapidfuzz import process, fuzz

from ultralytics import YOLO
from paddleocr import PPStructure, PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from core.config import CANONICAL_SUBJECTS, OCR_FIXES
from core.utils import remove_accents, fix_vietnamese_name

@st.cache_resource
def load_models():
    yolo = YOLO('models/best.pt') 
    ocr_table = PPStructure(show_log=False, lang='en')
    ocr_table_forced = PPStructure(layout=False, show_log=False, lang='en')
    ocr_text = PaddleOCR(use_angle_cls=False, lang='vi', show_log=False)
    
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu' 
    vietocr_model = Predictor(config)
    
    return yolo, ocr_table, ocr_table_forced, ocr_text, vietocr_model

def extract_student_info(img_info_bgr, text_engine, vietocr_engine):
    info = {"ho_ten": "", "lop": "", "nam_hoc": "", "ban": "", "raw_text": ""}
    try:
        h, w = img_info_bgr.shape[:2]
        name_y_min, name_y_max = 0, int(h * 0.45)
        name_x_max = int(w * 0.6)

        result_full = text_engine.ocr(img_info_bgr, cls=False)
        final_full_text = ""

        if result_full and result_full[0]:
            text_lines = []
            for line in result_full[0]:
                box = line[0]
                text = line[1][0].strip()
                if not text: continue
                text_lines.append(text)
                text_norm = remove_accents(text).lower()

                if any(k in text_norm for k in ["ho va ten", "ho ten", "va ten", "ho ", "ten "]):
                    ys = [pt[1] for pt in box]
                    name_y_min = max(0, int(min(ys)) - 8)
                    name_y_max = min(h, int(max(ys)) + 8)

                if re.search(r'(?i)(l[oơó0]p|l\dp)', text):
                    xs = [pt[0] for pt in box]
                    found_x_max = int(min(xs)) - 15
                    if found_x_max > w * 0.3:
                        name_x_max = found_x_max

            final_full_text = " ".join(text_lines)

            match_year = re.search(r'(20\d{2}\s*[-–—]\s*20\d{2})', final_full_text)
            if match_year: info["nam_hoc"] = match_year.group(1).replace(' ', '')

            match_class = re.search(r'\b(1[012][A-Z0-9]{1,4})\b', final_full_text, re.IGNORECASE)
            if match_class: info["lop"] = match_class.group(1).upper()
            else:
                m_lop_fb = re.search(r'(?i)(?:lớp|lop|lóp|l\dp|l6p)[\s:.]*([a-zA-Z0-9]+)', final_full_text)
                if m_lop_fb: info["lop"] = m_lop_fb.group(1).upper()

            if re.search(r'(?i)(c[oơ]\s*b[aả]n)', final_full_text): info["ban"] = "Cơ bản"
            elif re.search(r'(?i)(t[uự]\s*nhi[eê]n|khtn)', final_full_text): info["ban"] = "KHTN"
            elif re.search(r'(?i)(x[aã]\s*h[oộ]i|khxh)', final_full_text): info["ban"] = "KHXH"
            else:
                m_ban = re.search(r'(?i)\bban[\s:.]*(.*?)(?=\s*(?:môn|mon|chuy[eéê]n)|$)', final_full_text)
                if m_ban: info["ban"] = m_ban.group(1).strip(' :.-_')

        crop_name = img_info_bgr[name_y_min:name_y_max, 0:name_x_max]
        padded_name = cv2.copyMakeBorder(crop_name, 5, 5, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img_rgb = cv2.cvtColor(padded_name, cv2.COLOR_BGR2RGB)
        pil_img_name = Image.fromarray(img_rgb)

        raw_name_text = vietocr_engine.predict(pil_img_name)
        info["raw_text"] = f"[CẮT Y:{name_y_min}-{name_y_max} X:0-{name_x_max}] [VIETOCR]: {raw_name_text}"

        if raw_name_text:
            temp_name = re.sub(r'^(?i)(?:[^\s]+\s+){0,2}?(?:t[eêé]n|i[eêé]n|l[eêé]n|t[eêé]m|[eê]n|t[eêé]ng)[\s:.-]*', '', raw_name_text).strip()
            temp_name = re.sub(r'^(?i)(h[oọ]|bi[eệ]n|m[oọ]|b[oọ]|\bva\b|\bvà\b)\s*', '', temp_name).strip()
            temp_name = re.sub(r'(?i)\s*(l[oơó0]p|ban|năm|môn|[0-9]+).*', '', temp_name).strip()

            vietnamese_chars = r'a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸưăạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ\s'
            clean_name = re.sub(f'[^{vietnamese_chars}]', '', temp_name)

            words = clean_name.split()
            dedup_words = []
            for w in words:
                if not dedup_words or w.lower() != dedup_words[-1].lower():
                    dedup_words.append(w)

            info["ho_ten"] = fix_vietnamese_name(" ".join(dedup_words))

    except Exception as e:
        print(f"Lỗi OCR vùng thông tin: {e}")

    return info

def count_valid_subjects(df):
    if df.empty or df.shape[1] < 1: return 0
    first_col = df.columns[0]
    canonical_normalized = [remove_accents(s) for s in CANONICAL_SUBJECTS]
    valid_count = 0
    for val in df[first_col].astype(str):
        norm_val = remove_accents(val)
        if not norm_val or len(norm_val) < 2: continue
        match_norm, score, _ = process.extractOne(norm_val, canonical_normalized, scorer=fuzz.ratio)
        if score >= 55: valid_count += 1
    return valid_count

def check_garbage_table(df):
    if df.shape[1] < 3: return True, "Bảng bị thiếu/mất quá nhiều cột"
    merged_cells_count = 0
    for col in df.columns[1:]:
        for val in df[col].astype(str):
            if len(val.strip()) > 8 and any(char.isdigit() for char in val):
                merged_cells_count += 1
    if merged_cells_count >= 3: return True, "Các cột điểm bị dính chùm vào nhau do mất vạch kẻ dọc"
    if df.shape[0] < 5: return True, "Nhận diện thiếu quá nhiều hàng (môn học)"
    return False, "OK"

def clean_and_fix_table(df_raw):
    if df_raw.empty: return df_raw
    df = df_raw.copy()
    if len(df.columns) > 0: df.rename(columns={df.columns[0]: "Tên môn học"}, inplace=True)
    df = df.astype(str).apply(lambda x: x.str.replace(r'[\r\n]+', ' ', regex=True).str.strip())
    df.replace(['nan', 'None', '', '.'], np.nan, inplace=True)
    df.dropna(how='all', inplace=True)
    df.fillna('', inplace=True)
    
    canonical_map = {remove_accents(subj): subj for subj in CANONICAL_SUBJECTS}
    sorted_canonicals_norm = sorted(list(canonical_map.keys()), key=len, reverse=True)
    
    # 1. GỘP DÒNG BỊ XÉ ĐÔI / LỆCH DÒNG DO MỘC ĐÈ 
    if len(df) > 1:
        i = 0
        while i < len(df) - 1:
            curr_subj = str(df.iloc[i, 0]).strip()
            curr_has_scores = any(re.search(r'\d|[ĐDcCmM]', str(df.iloc[i, c]).strip(), re.IGNORECASE) for c in range(1, len(df.columns)))
            next_subj = str(df.iloc[i+1, 0]).strip()
            next_has_scores = any(re.search(r'\d|[ĐDcCmM]', str(df.iloc[i+1, c]).strip(), re.IGNORECASE) for c in range(1, len(df.columns)))
            
            if curr_has_scores and not next_has_scores and next_subj:
                curr_norm = remove_accents(curr_subj).lower()
                
                # Nhận diện chữ rác chắc chắn
                is_garbage = len(curr_subj) <= 2 or "5" in curr_subj or "stning" in curr_norm or "trung hoc" in curr_norm or "pho thong" in curr_norm
                
                # BẢO VỆ TÊN MÔN NGẮN: Chỉ xét rác bằng độ khớp mờ nếu chuỗi dài hơn 4 ký tự (Tránh giết nhầm KTCN)
                if not is_garbage and len(curr_subj) > 4:
                    score = process.extractOne(curr_norm, sorted_canonicals_norm, scorer=fuzz.ratio)
                    if score and score[1] < 45: 
                        is_garbage = True

                if is_garbage:
                    df.iloc[i, 0] = next_subj 
                    df.drop(df.index[i+1], inplace=True) 
                    df.reset_index(drop=True, inplace=True)
                    continue
            i += 1

    # 2. KHỚP TÊN MÔN HỌC CHUẨN 
    first_col = df.columns[0]
    if df.shape[1] >= 1:
        fixed_subjects = []
        summary_keywords = [r'dtb', r'đtb', r'trung b[iì]nh', r'tbm', r'c[aả] n[aă]m']
        
        for text in df[first_col]:
            if not text:
                fixed_subjects.append("")
                continue
            
            norm_text_lower = remove_accents(text).lower()
            if any(re.search(kw, norm_text_lower) for kw in summary_keywords):
                fixed_subjects.append("ĐTB các môn học")
                continue

            norm_text = remove_accents(text)
            if norm_text in OCR_FIXES:
                fixed_subjects.append(OCR_FIXES[norm_text])
                continue

            found_subject = None
            for c_norm in sorted_canonicals_norm:
                if re.search(r'\b' + re.escape(c_norm) + r'\b', norm_text):
                    found_subject = canonical_map[c_norm]
                    break
            if found_subject:
                fixed_subjects.append(found_subject)
                continue

            res_partial = process.extractOne(norm_text, sorted_canonicals_norm, scorer=fuzz.partial_ratio)
            res_ratio = process.extractOne(norm_text, sorted_canonicals_norm, scorer=fuzz.ratio)

            match_norm, score_partial = (res_partial[0], res_partial[1]) if res_partial else ("", 0)
            match_ratio, score_ratio = (res_ratio[0], res_ratio[1]) if res_ratio else ("", 0)

            if score_ratio >= 65:
                fixed_subjects.append(canonical_map[match_ratio])
            elif score_partial >= 85 and len(match_norm) >= 3: 
                fixed_subjects.append(canonical_map[match_norm])
            else: 
                fixed_subjects.append(text)
        df[first_col] = fixed_subjects

    # 3. FIX LỖI: NGHỀ -> CÔNG NGHỆ
    nghe_indices = [idx for idx, row in df.iterrows() if str(row[0]).strip().lower() in ["nghề", "nghe"]]
    if len(nghe_indices) >= 2:
        df.iloc[nghe_indices[0], 0] = "Công nghệ"
    elif len(nghe_indices) == 1:
        td_idx = [idx for idx, row in df.iterrows() if "thể dục" in str(row[0]).lower()]
        if td_idx and nghe_indices[0] < td_idx[0]:
            df.iloc[nghe_indices[0], 0] = "Công nghệ"

    # 4. NẮN ĐIỂM DÒNG THỂ DỤC / GDQP AN / NGHỀ
    for i in range(len(df)):
        subj_name = str(df.iloc[i, 0]).lower()
        if "thể dục" in subj_name:
            extracted_scores = []
            for c in range(1, len(df.columns)):
                val = str(df.iloc[i, c])
                nums = re.findall(r'\d+\.\d+|\d+', val)
                if nums:
                    extracted_scores.extend(nums)
                df.iloc[i, c] = "Đ"
            
            if len(extracted_scores) >= 2 and (i + 1) < len(df):
                df.iloc[i+1, 2] = extracted_scores[0]
                df.iloc[i+1, 3] = extracted_scores[1]
                
                if (i + 2) < len(df):
                    df.iloc[i+2, 3] = "9.0"

    # 5. DỌN DẸP DÒNG TRÙNG LẶP 
    subj_col = df.columns[0]
    duplicates = df[df.duplicated(subset=[subj_col], keep=False)]
    
    if not duplicates.empty:
        drop_indices = []
        for subj in duplicates[subj_col].unique():
            if not subj or subj == "ĐTB các môn học": 
                continue
            idx_list = df[df[subj_col] == subj].index.tolist()
            if len(idx_list) > 1:
                for idx in idx_list:
                    has_scores = any(re.search(r'\d|[ĐDcCmM]', str(df.iloc[idx, c]).strip(), re.IGNORECASE) for c in range(1, len(df.columns)))
                    if not has_scores:
                        drop_indices.append(idx)
                        
        if drop_indices:
            df.drop(drop_indices, inplace=True)
            df.reset_index(drop=True, inplace=True)

    return df

def clean_scores(df):
    def fix_val(val):
        if not val: return val
        s = str(val).strip().upper().replace(',', '.').replace('S', '5').replace('O', '0').replace('I', '1').replace('L', '1')
        if s in ['D', 'DAT', 'ĐẠT']: return 'Đ'
        if s in ['CD', 'CHUADAT', 'CHƯA ĐẠT', 'CĐẠT']: return 'CĐ'
        return s
    for col in df.columns[1:]: df[col] = df[col].apply(fix_val)
    return df

def fix_subject_names(df):
    """
    Sửa lỗi sai tên môn học bằng Fuzzy Matching (Khớp chuỗi).
    """
    STANDARD_SUBJECTS = [
        "Toán", "Vật lí", "Hóa học", "Sinh học", "Tin học",
        "Ngữ văn", "Lịch sử", "Địa lí", "Tiếng Anh", "Ngoại ngữ",
        "GDCD", "Công nghệ", "Thể dục", "GDQP AN", "KTCN", "Tự chọn"
    ]

    # Giả định cột 0 là cột Tên môn học
    for i in range(df.shape[0]):
        raw_name = str(df.iloc[i, 0]).strip()
        
        if raw_name and raw_name != "nan":
            # Bỏ qua nếu đã đúng chuẩn
            if raw_name in STANDARD_SUBJECTS:
                continue
                
            # Tìm môn học có cách viết giống nhất (độ chính xác tối thiểu 40%)
            matches = difflib.get_close_matches(raw_name, STANDARD_SUBJECTS, n=1, cutoff=0.4)
            
            if matches:
                df.iloc[i, 0] = matches[0]
            else:
                # Nếu bị mộc đè nát (như chữ "đ") không thể đoán được
                if len(raw_name) <= 2:
                    df.iloc[i, 0] = "" # Để trống để người dùng tự nhập lại

    return df

def validate_ocr_results(info, df):
    warnings = []

    # ==========================================
    # 1. KIỂM TRA LOGIC THÔNG TIN HỌC SINH
    # ==========================================
    ho_ten = str(info.get("ho_ten", "")).strip()
    
    if not ho_ten:
        warnings.append("👤 **Chưa nhận diện được Họ và tên** học sinh.")
    else:
        words = ho_ten.split()
        
        # Rule a: Kiểm tra độ dài tên (Nếu dài hơn 6 chữ là bất thường)
        if len(words) > 6:
            warnings.append(f"👤 **Tên học sinh quá dài ({len(words)} từ)**. AI có thể đã đọc lẹm vào phần chữ khác.")
        
        # Rule b: Kiểm tra ký tự lạ (Chỉ cho phép bảng chữ cái tiếng Việt và khoảng trắng)
        # Regex bao gồm toàn bộ ký tự tiếng Việt có dấu
        vi_chars = r"a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ\s"
        if not re.fullmatch(f"[{vi_chars}]+", ho_ten):
            warnings.append("👤 **Tên chứa ký tự lạ, số hoặc dấu câu**. Vui lòng kiểm tra lại chính tả.")
            
        # Rule c: Kiểm tra độ dài của từng chữ cái (Chữ tiếng Việt dài nhất thường là 'Nghiêng', 'Nguyễn' - tối đa 7-8 ký tự)
        for w in words:
            if len(w) > 8:
                warnings.append(f"👤 **Phát hiện cụm từ '{w}' dài bất thường**. Có thể do nhiễu hoặc AI dính chữ.")

    if not info.get("lop"):
        warnings.append("🏫 **Chưa nhận diện được Lớp**.")
    if not info.get("nam_hoc"):
        warnings.append("📅 **Chưa nhận diện được Năm học**.")

    # ==========================================
    # 2. KIỂM TRA BẢNG ĐIỂM
    # ==========================================
    if df.empty:
        warnings.append("🚨 Bảng điểm trống hoặc không đọc được dữ liệu.")
        return warnings

    score_cols = df.columns[1:]
    for idx, row in df.iterrows():
        subj_name = str(row[0]).strip()

        # Môn có điểm nhưng trống tên
        has_scores = any(str(row[c]).strip() for c in score_cols)
        if not subj_name and has_scores:
            warnings.append(f"⚠️ Dòng {idx+1}: Có điểm nhưng trống tên môn (nghi do mộc đè).")

        # Môn chính bị khuyết/trống cột điểm
        if subj_name and subj_name not in ["Tự chọn", ""]:
            missing_cols = [str(c) for c in score_cols if not str(row[c]).strip() or str(row[c]).strip().lower() in ['nan', 'none']]
            if missing_cols:
                warnings.append(f"📝 Môn {subj_name} đang bị thiếu điểm ở cột: {', '.join(missing_cols)}.")

    return warnings