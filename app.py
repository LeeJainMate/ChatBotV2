from flask import Flask, request, jsonify, render_template, session
import os
import pandas as pd
from google import genai
from google.genai import types

# โหลด .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
app.secret_key = "statbot_secret_key_change_this_2026"

# ดึง API Key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ ไม่พบ GEMINI_API_KEY ในไฟล์ .env!")

client = genai.Client(api_key=API_KEY)

# ============ โหลดฐานข้อมูลสารสนเทศจาก CSV ============
DATA_FILE = "data/Questions_Answer.csv"  # เปลี่ยนชื่อไฟล์ตามที่คุณใช้จริง
df_data = None

if os.path.exists(DATA_FILE):
    try:
        df_data = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
        # ทำความสะอาดข้อมูลเบื้องต้น
        df_data = df_data.dropna(how='all')  # ลบแถวว่างทั้งแถว
        df_data = df_data.fillna('')  # แทน NaN ด้วยค่าว่าง
        print(f"✅ โหลดฐานข้อมูลสำเร็จ: {len(df_data)} แถว, {len(df_data.columns)} คอลัมน์")
        print("คอลัมน์ที่มี:", list(df_data.columns))
    except Exception as e:
        print(f"⚠️ อ่านไฟล์ข้อมูลไม่สำเร็จ: {e}")
        df_data = None
else:
    print(f"⚠️ ไม่พบไฟล์ {DATA_FILE} — บอทจะใช้ Gemini ทั่วไปโดยไม่มีข้อมูลเฉพาะ")
    df_data = None

# ============ Chat History ============
def get_text_history():
    if 'history' not in session:
        session['history'] = [
            {"role": "user", "text": "สวัสดี"},
            {"role": "model", "text": "สวัสดีครับ! ผมคือ Statbot ผู้ช่วยของมหาวิทยาลัยมหาสารคาม พร้อมตอบคำถามจากข้อมูลล่าสุดแล้วนะครับ 😊"}
        ]
    return session['history']

def build_contents_with_data_context(user_message):
    history = get_text_history()
    history.append({"role": "user", "text": user_message})

    # สร้าง context จากข้อมูลทั้งหมดใน CSV
    context = ""
    if df_data is not None and not df_data.empty:
        # แปลงข้อมูลเป็นข้อความที่ Gemini อ่านเข้าใจง่าย
        data_text = df_data.to_string(index=False, max_rows=100)  # จำกัดเพื่อไม่ให้เกิน token limit
        context = (
            "นี่คือข้อมูลจากฐานข้อมูลมหาวิทยาลัยมหาสารคาม (ข้อมูลล่าสุด):\n"
            f"{data_text}\n\n"
            "โปรดใช้ข้อมูลนี้ตอบคำถามของผู้ใช้ให้ถูกต้องที่สุด "
            "ถ้าไม่มีข้อมูลที่เกี่ยวข้อง ให้บอกสุภาพว่า 'ขออภัยครับ ข้อมูลนี้ยังไม่มีในระบบ'\n\n"
        )

    # สร้าง contents สำหรับ Gemini
    contents = []
    for msg in history[:-1]:  # ข้อความเก่า
        contents.append(types.Content(
            role=msg["role"],
            parts=[types.Part.from_text(text=msg["text"])]
        ))

    # ข้อความล่าสุด + context
    full_message = context + "คำถามจากผู้ใช้: " + user_message
    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=full_message)]
    ))

    return contents

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "กรุณาพิมพ์ข้อความ"}), 400

    text_history = get_text_history()
    text_history.append({"role": "user", "text": user_message})

    contents = build_contents_with_data_context(user_message)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=300,
                temperature=0.7,
                system_instruction=(
                    "คุณคือ Statbot ผู้ช่วยอัจฉริยะของมหาวิทยาลัยมหาสารคาม\n"
                    "ตอบคำถามจากข้อมูลในฐานข้อมูลที่ให้มาเป็นหลัก\n"
                    "ตอบเป็นภาษาไทย สุภาพ เป็นมิตร ชัดเจน\n"
                    "ตอบสั้นกระชับ อ่านง่าย ใช้ emoji นิดหน่อย 😊\n"
                    "ถ้าไม่มีข้อมูลในฐานข้อมูล ให้บอกว่า 'ขออภัยครับ ข้อมูลนี้ยังไม่มีในระบบ ลองติดต่องานทะเบียนโดยตรงนะครับ'"
                )
            )
        )
        reply = response.text.strip()
    except Exception as e:
        print("Gemini error:", e)
        reply = "ขออภัยครับ ระบบมีปัญหาชั่วคราว ลองใหม่ในอีกสักครู่นะครับ 🙏"

    text_history.append({"role": "model", "text": reply})

    # จำกัด history
    if len(text_history) > 30:
        session['history'] = text_history[-30:]

    session.modified = True
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)