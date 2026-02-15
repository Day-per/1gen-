import io, re, os
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser

import cv2
import numpy as np
from PIL import Image
import pytesseract

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


SERVICE_ACCOUNT_JSON = "service_account.json"

SPREADSHEET_ID = os.environ["SPREADSHEET_ID"]
TRACKER_TAB = "tracker"

PROCESSED_FOLDER_ID = os.environ["PROCESSED_FOLDER_ID"]
NEEDS_REVIEW_FOLDER_ID = os.environ["NEEDS_REVIEW_FOLDER_ID"]

MAX_PER_RUN = int(os.environ.get("MAX_PER_RUN", "20"))

# Quota: limit total processed per window
QUOTA_PER_WINDOW = int(os.environ.get("QUOTA_PER_WINDOW", "30"))
QUOTA_WINDOW_HOURS = int(os.environ.get("QUOTA_WINDOW_HOURS", "12"))
CONTROL_ROW = int(os.environ.get("CONTROL_ROW", "2"))  # row where quota state is stored

# Crop boxes (tune if needed). Fractions of the warped image.
BOXES = {
    "receipt_number": (0.74, 0.22, 0.97, 0.27),
    "date":           (0.74, 0.27, 0.97, 0.33),
    "received_from":  (0.40, 0.34, 0.97, 0.47),
    "in_payment_for": (0.40, 0.52, 0.97, 0.64),
    "amount":         (0.33, 0.72, 0.63, 0.80),
}


def get_creds():
    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    return Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=scopes)

def drive_service(creds):
    return build("drive", "v3", credentials=creds)

def download_file_bytes(drive, file_id):
    req = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()

def move_file(drive, file_id, new_parent):
    meta = drive.files().get(fileId=file_id, fields="parents").execute()
    prev = ",".join(meta.get("parents", []))
    drive.files().update(
        fileId=file_id,
        addParents=new_parent,
        removeParents=prev,
        fields="id,parents"
    ).execute()

def open_tracker(creds):
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(SPREADSHEET_ID)
    return sh.worksheet(TRACKER_TAB)

def get_headers(ws):
    headers = ws.row_values(1)
    return {h: i+1 for i, h in enumerate(headers)}

def find_queued_rows(ws, hdr, limit):
    status_col = hdr["status"]
    values = ws.get_all_values()
    queued = []
    for r in range(2, len(values)+1):
        row = values[r-1]
        status = (row[status_col-1] or "").strip().upper()
        if status == "QUEUED":
            queued.append(r)
            if len(queued) >= limit:
                break
    return queued

def update_fields(ws, row, hdr, updates: dict):
    for k, v in updates.items():
        if k in hdr:
            ws.update_cell(row, hdr[k], v)

# ---- Quota helpers (stored in CONTROL_ROW) ----
def iso_now():
    return datetime.now(timezone.utc)

def parse_iso(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def ensure_quota_columns(hdr):
    missing = [c for c in ["quota_window_start", "quota_count_in_window"] if c not in hdr]
    if missing:
        raise RuntimeError(f"Missing quota columns in tracker sheet: {missing}")

def get_quota_state(ws, hdr):
    ensure_quota_columns(hdr)

    start_s = ws.cell(CONTROL_ROW, hdr["quota_window_start"]).value
    count_s = ws.cell(CONTROL_ROW, hdr["quota_count_in_window"]).value

    start_dt = parse_iso(start_s) if start_s else None
    try:
        count = int(count_s) if count_s else 0
    except Exception:
        count = 0

    if not start_dt:
        start_dt = iso_now()
        count = 0
        ws.update_cell(CONTROL_ROW, hdr["quota_window_start"], start_dt.isoformat())
        ws.update_cell(CONTROL_ROW, hdr["quota_count_in_window"], str(count))

    return start_dt, count

def reset_quota_state(ws, hdr):
    start_dt = iso_now()
    ws.update_cell(CONTROL_ROW, hdr["quota_window_start"], start_dt.isoformat())
    ws.update_cell(CONTROL_ROW, hdr["quota_count_in_window"], "0")
    return start_dt, 0

def update_quota_count(ws, hdr, new_count):
    ws.update_cell(CONTROL_ROW, hdr["quota_count_in_window"], str(new_count))

# --- Image / OCR ---
def pil_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def find_and_warp_document(cv_img):
    img = cv_img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    doc = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc = approx
            break

    if doc is None:
        return cv_img

    pts = doc.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    src = np.array([tl, tr, br, bl], dtype="float32")
    W, H = 2000, 1400
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(cv_img, M, (W, H))

def crop_norm(cv_img, box):
    h, w = cv_img.shape[:2]
    l, t, r, b = box
    x1, y1 = int(l*w), int(t*h)
    x2, y2 = int(r*w), int(b*h)
    return cv_img[y1:y2, x1:x2]

def ocr_image(cv_img, psm=6, whitelist=None):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = f"--oem 1 --psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    text = pytesseract.image_to_string(th, lang="eng", config=config)
    return re.sub(r"\s+", " ", text).strip()

def parse_amount(text):
    m = re.search(r"(\d{1,3}(?:[,\s]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2})?)", text)
    if not m:
        return ""
    return m.group(1).replace(" ", "").replace(",", "")

def parse_date(text):
    if not text:
        return ""
    try:
        dt = dateparser.parse(text, dayfirst=True, fuzzy=True)
        return dt.date().isoformat()
    except Exception:
        return text

def extract_fields(pil_img):
    cv_img = pil_to_cv(pil_img)
    warped = find_and_warp_document(cv_img)

    receipt_no = ocr_image(crop_norm(warped, BOXES["receipt_number"]), psm=7)
    date_raw   = ocr_image(crop_norm(warped, BOXES["date"]), psm=7, whitelist="0123456789/-.")
    recv_from  = ocr_image(crop_norm(warped, BOXES["received_from"]), psm=6)
    pay_for    = ocr_image(crop_norm(warped, BOXES["in_payment_for"]), psm=6)
    amt_raw    = ocr_image(crop_norm(warped, BOXES["amount"]), psm=7, whitelist="0123456789.,")
    amount     = parse_amount(amt_raw)

    fields = {
        "ocr_receipt_number": receipt_no,
        "ocr_date": parse_date(date_raw),
        "ocr_received_from": recv_from,
        "ocr_in_payment_for": pay_for,
        "ocr_amount": amount,
    }
    raw = (
        f"receipt_no={receipt_no} | date_raw={date_raw} | amount_raw={amt_raw} | "
        f"received_from={recv_from} | payment_for={pay_for}"
    )
    return fields, raw

def needs_review(fields):
    must = ["ocr_receipt_number", "ocr_date", "ocr_received_from", "ocr_in_payment_for", "ocr_amount"]
    return any(not (fields.get(k) or "").strip() for k in must)

def main():
    creds = get_creds()
    drive = drive_service(creds)
    ws = open_tracker(creds)
    hdr = get_headers(ws)

    # --- quota gate ---
    window_start, used = get_quota_state(ws, hdr)
    if iso_now() - window_start >= timedelta(hours=QUOTA_WINDOW_HOURS):
        window_start, used = reset_quota_state(ws, hdr)

    remaining = QUOTA_PER_WINDOW - used
    if remaining <= 0:
        print(f"Quota reached: {used}/{QUOTA_PER_WINDOW} in current {QUOTA_WINDOW_HOURS}h window.")
        return

    limit = min(MAX_PER_RUN, remaining)

    queued_rows = find_queued_rows(ws, hdr, limit)
    if not queued_rows:
        print("No queued rows.")
        return

    for row in queued_rows:
        file_id = ws.cell(row, hdr["drive_file_id"]).value.strip()
        now = iso_now().isoformat()

        update_fields(ws, row, hdr, {
            "status": "PROCESSING",
            "last_processed_at": now,
            "notes": ""
        })

        try:
            img_bytes = download_file_bytes(drive, file_id)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            fields, raw = extract_fields(pil_img)
            review = needs_review(fields)
            status = "NEEDS_REVIEW" if review else "PARSED"

            update_fields(ws, row, hdr, {
                **fields,
                "ocr_raw": raw,
                "status": status,
                "last_processed_at": iso_now().isoformat(),
                "notes": "Missing/uncertain fields" if review else ""
            })

            move_file(drive, file_id, NEEDS_REVIEW_FOLDER_ID if review else PROCESSED_FOLDER_ID)
            print(f"Row {row} processed: {status}")

            used += 1
            update_quota_count(ws, hdr, used)

            if used >= QUOTA_PER_WINDOW:
                print(f"Quota reached mid-run: {used}/{QUOTA_PER_WINDOW}. Stopping.")
                return

        except Exception as ex:
            update_fields(ws, row, hdr, {
                "status": "ERROR",
                "last_processed_at": iso_now().isoformat(),
                "notes": str(ex)[:500]
            })
            print(f"Row {row} ERROR: {ex}")

if __name__ == "__main__":
    main()
