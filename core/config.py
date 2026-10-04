# core/config.py

# Danh sách môn học chuẩn
CANONICAL_SUBJECTS = [
    "Toán", "Vật lí", "Hóa học", "Sinh học", "Tin học",
    "Ngữ văn", "Lịch sử", "Địa lí", "Tiếng Anh", "Ngoại ngữ", "GDCD",
    "Công nghệ", "Thể dục", "Giáo dục thể chất", "GDQP AN", 
    "Giáo dục quốc phòng và an ninh", "Nghề", "Hoạt động trải nghiệm", 
    "Nội dung giáo dục địa phương", "Âm nhạc", "Mĩ thuật", "Khoa học tự nhiên", 
    "Lịch sử và Địa lí", "Tự chọn", "ĐTB các môn học"
]

# Sửa lỗi OCR viết sai tên môn
OCR_FIXES = {
    "tyr chon": "Tự chọn",
    "tu chon": "Tự chọn",
    "lich sur": "Lịch sử",
    "diali": "Địa lí",
    "toan": "Toán"
}

# Từ điển hỗ trợ tự động phục hồi dấu cho Họ và Tên
NAME_DICTIONARY = {
    # Họ phổ biến
    "nguyen": "Nguyễn", "tran": "Trần", "le": "Lê", "pham": "Phạm",
    "hoang": "Hoàng", "huynh": "Huỳnh", "phan": "Phan", "vu": "Vũ",
    "vo": "Võ", "dang": "Đặng", "bui": "Bùi", "do": "Đỗ", "ho": "Hồ",
    "ngo": "Ngô", "duong": "Dương", "ly": "Lý", "dao": "Đào", "dinh": "Đinh",
    "trinh": "Trịnh", "doan": "Đoàn", "lam": "Lâm", "cao": "Cao", "luong": "Lương",
    # Tên đệm & Tên
    "van": "Văn", "thi": "Thị", "quoc": "Quốc", "duc": "Đức", "minh": "Minh",
    "tuan": "Tuấn", "huu": "Hữu", "xuan": "Xuân", "ngoc": "Ngọc", "phuoc": "Phước",
    "dinh": "Đình", "tan": "Tấn", "ba": "Bá", "trong": "Trọng", "huong": "Hương",
    "thuy": "Thúy", "phuong": "Phương", "ngan": "Ngân", "truong": "Trường",
    "khanh": "Khánh", "dat": "Đạt", "phat": "Phát", "loc": "Lộc", "bao": "Bảo",
    "anh": "Anh", "hieu": "Hiếu", "hoai": "Hoài", "thanh": "Thành", "hung": "Hùng",
    "dung": "Dũng", "khai": "Khải", "ha": "Hà", "hai": "Hải", "truc": "Trúc"
}