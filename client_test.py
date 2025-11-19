import requests
import socketio
import base64
import time
from PIL import Image
import io
import os

# ====== 設定 ======
RASPI_IP = "192.168.210.90"      # ← Raspberry Pi の IP
PORT = 5001

START_URL = f"http://{RASPI_IP}:{PORT}/start_scan"
GET_IMAGE_URL = f"http://{RASPI_IP}:{PORT}/get_scan_image"
WS_URL = f"http://{RASPI_IP}:{PORT}"

# ====== Socket.IO クライアント ======
sio = socketio.Client()

@sio.event
def connect():
    print("🔗 WebSocket 接続成功")

@sio.event
def disconnect():
    print("❌ WebSocket 切断")

@sio.on("scan_completed")
def on_scan_completed(data):
    print("📩 スキャン完了通知を受信:", data)

    # 画像を取得
    print("📡 /get_scan_image を取得中...")
    r = requests.get(GET_IMAGE_URL)
    
    if r.status_code != 200:
        print("❌ /get_scan_image エラー:", r.text)
        return
    
    json_data = r.json()
    img_base64 = json_data.get("image")

    # Base64 → バイナリへ
    img_bytes = base64.b64decode(img_base64)

    # ファイル保存
    file_path = "received_scan.jpg"
    with open(file_path, "wb") as f:
        f.write(img_bytes)

    print(f"💾 画像を保存しました → {file_path}")

    # ファイルサイズ表示
    file_size = os.path.getsize(file_path)
    print(f"📏 ファイルサイズ: {file_size} bytes ({file_size/1024:.1f} KB)")

    # 画像のピクセルサイズ取得
    try:
        img = Image.open(io.BytesIO(img_bytes))
        print(f"🖼 画像の解像度: {img.width} x {img.height} px")
    except Exception as e:
        print("⚠ 画像解析エラー:", e)


# ====== メイン ======
def main():
    print("🔌 WebSocket に接続します...")
    sio.connect(WS_URL, transports=["websocket"])

    print("\n▶ /start_scan を送信してスキャンを開始")
    r = requests.post(START_URL)
    print("レスポンス:", r.json())

    print("\n📡 スキャン完了通知を待機中...\n")

    # WebSocket を待機（Ctrl+C で終了）
    try:
        sio.wait()
    except KeyboardInterrupt:
        print("\n終了します")

if __name__ == "__main__":
    main()
