import cv2
import numpy as np
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import base64
import threading
from flask_cors import CORS
import tkinter as tk

from flask import Flask, request, jsonify
from flask_socketio import SocketIO

# ============================================================
# グローバル状態（Flask / SocketIO とカメラ処理で共有）
# ============================================================
scan_enabled = False               # False: 待機中 / True: スキャン許可中
previous_corners = None
stable_start_time = None
stable_threshold = 2.0             # 2秒間安定したら撮影
position_threshold = 15            # 位置変化が15px以内なら安定
image_saved = False
completion_message_time = None     # 「送信完了」表示開始時刻
completion_message_duration = 3.0  # 送信完了メッセージ表示時間（秒）

# ディスプレイサイズ（動的に取得）
DISPLAY_WIDTH = None
DISPLAY_HEIGHT = None

# 全画面表示状態
is_fullscreen = True

# ============================================================
# Flask + SocketIO 設定
# ============================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ============================================================
# Flask API（タブレットからの制御用）
# ============================================================
@app.route("/start_scan", methods=["POST"])
def start_scan():
    """
    タブレット側からスキャン開始要求を受けたときに呼ばれる。
    """
    global scan_enabled, previous_corners, stable_start_time, image_saved, completion_message_time
    scan_enabled = True
    previous_corners = None
    stable_start_time = None
    image_saved = False
    completion_message_time = None
    print("📡 /start_scan 受信 → スキャンモードに移行")
    return {"status": "OK", "mode": "scan_enabled"}

@app.route("/stop_scan", methods=["POST"])
def stop_scan():
    """
    任意でタブレットからスキャン中止を行いたい場合に使用。
    """
    global scan_enabled, previous_corners, stable_start_time, image_saved
    scan_enabled = False
    previous_corners = None
    stable_start_time = None
    image_saved = False
    print("📡 /stop_scan 受信 → 待機モードへ移行")
    return {"status": "OK", "mode": "scan_disabled"}

@app.route("/get_scan_image", methods=["GET"])
def get_scan_image():
    """
    最新のスキャン画像をBase64形式で返す
    """
    try:
        image_path = "scan_file.jpg"

        if not os.path.exists(image_path):
            return jsonify({
                "status": "error",
                "message": "スキャン画像が見つかりません"
            }), 404

        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        image_base64 = base64.b64encode(image_data).decode("utf-8")
        image_mime = "image/jpeg"

        print("📡 /get_scan_image 受信 → Base64形式で画像を返送")

        return jsonify({
            "status": "OK",
            "image": image_base64,
            "mime_type": image_mime,
            "format": "base64"
        })

    except Exception as e:
        print(f"❌ /get_scan_image エラー: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ============================================================
# ユーティリティ関数
# ============================================================
def get_display_size():
    """ディスプレイサイズを取得"""
    global DISPLAY_WIDTH, DISPLAY_HEIGHT
    if DISPLAY_WIDTH is not None and DISPLAY_HEIGHT is not None:
        return DISPLAY_WIDTH, DISPLAY_HEIGHT
    
    try:
        root = tk.Tk()
        DISPLAY_WIDTH = root.winfo_screenwidth()
        DISPLAY_HEIGHT = root.winfo_screenheight()
        root.destroy()
        print(f"📺 ディスプレイサイズ: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
        return DISPLAY_WIDTH, DISPLAY_HEIGHT
    except Exception as e:
        print(f"⚠ ディスプレイサイズ取得エラー: {e} → デフォルト値を使用")
        DISPLAY_WIDTH = 1920
        DISPLAY_HEIGHT = 1080
        return DISPLAY_WIDTH, DISPLAY_HEIGHT

def resize_for_display(frame):
    """フレームをディスプレイサイズに合わせてリサイズ（アスペクト比維持）"""
    display_w, display_h = get_display_size()
    h, w = frame.shape[:2]
    
    # ディスプレイサイズに合わせてリサイズ
    scale_w = display_w / w
    scale_h = display_h / h
    scale = min(scale_w, scale_h)
    new_width = int(w * scale)
    new_height = int(h * scale)

    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def find_japanese_font(font_size=60):
    """日本語フォントを検索して読み込む"""
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
        "/usr/share/fonts/truetype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/truetype/ipafont-mincho/ipam.ttf",
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                return font
            except Exception:
                continue

    font_dirs = [
        "/usr/share/fonts/opentype/noto/",
        "/usr/share/fonts/truetype/noto/",
        "/usr/share/fonts/truetype/takao-gothic/",
        "/usr/share/fonts/truetype/ipafont-gothic/",
        "/usr/share/fonts/truetype/ipafont-mincho/",
    ]
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for ext in ["*.ttf", "*.ttc", "*.otf"]:
                for font_file in glob.glob(os.path.join(font_dir, ext)):
                    try:
                        font = ImageFont.truetype(font_file, font_size)
                        return font
                    except Exception:
                        continue

    print("警告: 日本語フォントが見つかりません。デフォルトフォントを使用します。")
    return ImageFont.load_default()

def draw_rounded_rectangle(draw, xy, radius, fill=None, outline=None, width=1):
    """角丸矩形を描画（Pillowの古いバージョン対応、RGBA対応）"""
    x1, y1, x2, y2 = xy
    r = radius
    
    # RGBAタプルの場合はRGBのみを抽出
    if isinstance(fill, tuple) and len(fill) == 4:
        fill_rgb = fill[:3]
    else:
        fill_rgb = fill
    
    # 角丸の部分を描画
    draw.ellipse([x1, y1, x1 + 2*r, y1 + 2*r], fill=fill_rgb, outline=outline, width=width)  # 左上
    draw.ellipse([x2 - 2*r, y1, x2, y1 + 2*r], fill=fill_rgb, outline=outline, width=width)  # 右上
    draw.ellipse([x1, y2 - 2*r, x1 + 2*r, y2], fill=fill_rgb, outline=outline, width=width)  # 左下
    draw.ellipse([x2 - 2*r, y2 - 2*r, x2, y2], fill=fill_rgb, outline=outline, width=width)  # 右下
    
    # 矩形の中央部分を描画
    draw.rectangle([x1 + r, y1, x2 - r, y2], fill=fill_rgb, outline=None)  # 横
    draw.rectangle([x1, y1 + r, x2, y2 - r], fill=fill_rgb, outline=None)  # 縦

def put_japanese_text_modern(img, text, position, font_size=80, 
                            text_color=(255, 255, 255), 
                            bg_color=(0, 0, 0, 180),
                            shadow=True,
                            center=True):
    """モダンなデザインで日本語テキストを画像に描画"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # RGBAモードに変換（透明度対応）
    if pil_img.mode != 'RGBA':
        pil_img = pil_img.convert('RGBA')
    
    font = find_japanese_font(font_size)
    draw = ImageDraw.Draw(pil_img)

    # テキストのサイズを取得
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # パディング設定
    padding_x = int(font_size * 0.8)
    padding_y = int(font_size * 0.5)
    corner_radius = int(font_size * 0.3)
    
    # 位置の計算（中央揃えの場合）
    if center:
        x, y = position
        box_x1 = x - text_width // 2 - padding_x
        box_y1 = y - text_height // 2 - padding_y
        box_x2 = x + text_width // 2 + padding_x
        box_y2 = y + text_height // 2 + padding_y
        text_x = x - text_width // 2
        text_y = y - text_height // 2
    else:
        x, y = position
        box_x1 = x - padding_x
        box_y1 = y - padding_y
        box_x2 = x + text_width + padding_x
        box_y2 = y + text_height + padding_y
        text_x = x
        text_y = y
    
    # 影を描画（オプション）
    if shadow:
        shadow_offset = int(font_size * 0.05)
        shadow_box = (
            box_x1 + shadow_offset,
            box_y1 + shadow_offset,
            box_x2 + shadow_offset,
            box_y2 + shadow_offset
        )
        draw_rounded_rectangle(
            draw, shadow_box, corner_radius,
            fill=(0, 0, 0, 100)
        )
    
    # 背景ボックスを描画（半透明）
    if isinstance(bg_color, tuple) and len(bg_color) == 4:
        # RGBAモードで半透明背景を描画
        bg_img = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(bg_img)
        draw_rounded_rectangle(
            bg_draw, (box_x1, box_y1, box_x2, box_y2), corner_radius,
            fill=bg_color
        )
        pil_img = Image.alpha_composite(pil_img, bg_img)
        draw = ImageDraw.Draw(pil_img)
    else:
        # 不透明背景
        bg_rgb = bg_color[:3] if len(bg_color) > 3 else bg_color
        draw_rounded_rectangle(
            draw, (box_x1, box_y1, box_x2, box_y2), corner_radius,
            fill=bg_rgb
        )
    
    # テキストを描画
    draw.text((text_x, text_y), text, font=font, fill=text_color)
    
    # BGRに変換して返す
    if pil_img.mode == 'RGBA':
        pil_img = pil_img.convert('RGB')
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr

def calculate_corner_distance(corners1, corners2):
    """2つの四角形の角位置の平均距離を計算"""
    if corners1 is None or corners2 is None:
        return float("inf")
    if len(corners1) != 4 or len(corners2) != 4:
        return float("inf")

    total_distance = 0
    for i in range(4):
        x1, y1 = corners1[i][0]
        x2, y2 = corners2[i][0]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += distance

    return total_distance / 4

def order_points(pts):
    """4点を 左上,右上,右下,左下 の順に並べる"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

def four_point_transform(image, pts):
    """4点の透視変換を行い、矩形に変換"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    scale_factor = 1.2
    maxWidth = int(maxWidth * scale_factor)
    maxHeight = int(maxHeight * scale_factor)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped

def draw_minimize_button(frame, button_size=50, margin=20):
    """右上に最小化ボタンを描画（リサイズ後のフレーム用）"""
    h, w = frame.shape[:2]
    
    # ボタンの位置（右上）
    button_x = w - button_size - margin
    button_y = margin
    button_x2 = w - margin
    button_y2 = margin + button_size
    
    # 背景（半透明のダークグレー）
    overlay = frame.copy()
    cv2.rectangle(overlay, (button_x, button_y), (button_x2, button_y2), (40, 40, 40), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # ボーダー
    cv2.rectangle(frame, (button_x, button_y), (button_x2, button_y2), (200, 200, 200), 2)
    
    # ×アイコンを描画
    line_thickness = max(2, int(button_size * 0.06))
    line_length = button_size // 3
    center_x = button_x + button_size // 2
    center_y = button_y + button_size // 2
    
    # ×の線を描画
    cv2.line(frame, 
             (center_x - line_length // 2, center_y - line_length // 2),
             (center_x + line_length // 2, center_y + line_length // 2),
             (255, 255, 255), line_thickness)
    cv2.line(frame,
             (center_x - line_length // 2, center_y + line_length // 2),
             (center_x + line_length // 2, center_y - line_length // 2),
             (255, 255, 255), line_thickness)
    
    return frame, (button_x, button_y, button_x2, button_y2)

def mouse_callback(event, x, y, flags, param):
    """マウスクリックイベントのコールバック"""
    global is_fullscreen
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # ボタンエリアの取得（paramから）
        if param and len(param) == 4:
            button_x, button_y, button_x2, button_y2 = param
            
            # クリック位置がボタンエリア内かチェック
            if button_x <= x <= button_x2 and button_y <= y <= button_y2:
                # 全画面を解除
                window_name = "Corner Detection"
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                is_fullscreen = False
                print("🖥 全画面表示を解除しました")


# ============================================================
# カメラループ（別スレッドで起動）
# ============================================================
def camera_loop():
    global scan_enabled, previous_corners, stable_start_time
    global image_saved, completion_message_time

    # ---------- カメラ初期化（GStreamer ＋ V4L2 フォールバック） ----------
    resolutions = [
        (2560, 1440),  # 2K
        (1920, 1080),  # Full HD
        (1280, 720),   # HD
    ]

    cap = None
    actual_width = None
    actual_height = None

    # GStreamer で MJPEG 高解像度を優先
    for width, height in resolutions:
        pipeline = (
            f"v4l2src device=/dev/video0 ! "
            f"image/jpeg, width={width}, height={height}, framerate=30/1 ! "
            f"jpegdec ! videoconvert ! "
            f"appsink max-buffers=1 drop=true"
        )
        print(f"GStreamer で解像度設定を試行: {width}x{height}")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if cap.isOpened():
            for _ in range(5):
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    actual_height, actual_width = test_frame.shape[:2]
                    print(f"  実際の解像度: {actual_width}x{actual_height}")
                    break
                time.sleep(0.1)
            if actual_width and actual_width >= width * 0.9:
                print(f"✅ GStreamer で高解像度キャプチャ成功: {actual_width}x{actual_height}")
                break
            else:
                cap.release()
                cap = None

    # GStreamer 失敗 → V4L2 フォールバック
    if cap is None or not cap.isOpened():
        print("⚠ GStreamer 失敗 → V4L2 にフォールバックします...")
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
        except Exception:
            cap = cv2.VideoCapture(0)

        for width, height in resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            time.sleep(0.1)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  V4L2 実際の解像度: {actual_width}x{actual_height}")
            if actual_width >= width * 0.9 and actual_height >= height * 0.9:
                print(f"✅ V4L2 で解像度設定成功: {actual_width}x{actual_height}")
                break

    if not cap.isOpened():
        print("❌ エラー: カメラを開けませんでした（camera_loop終了）")
        return

    if actual_width is None:
        for _ in range(5):
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                actual_height, actual_width = test_frame.shape[:2]
                print(f"✅ 最終解像度: {actual_width}x{actual_height}")
                break
            time.sleep(0.1)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    try:
        for _ in range(3):
            ret, _ = cap.read()
            if not ret:
                break
    except Exception:
        pass

    print("🎥 カメラ初期化完了。待機モードで起動します。")
    
    # ウィンドウを全画面表示に設定
    window_name = "Corner Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("🖥 全画面表示モードに設定しました")
    
    # マウスイベントのコールバックを設定（初期値はNone、ループ内で更新）
    button_area = None
    cv2.setMouseCallback(window_name, mouse_callback, button_area)

    # ---------- メインループ ----------
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue

        h, w = frame.shape[:2]
        current_time = time.time()

        # 「送信完了」メッセージの寿命管理
        if completion_message_time is not None:
            if current_time - completion_message_time >= completion_message_duration:
                completion_message_time = None

        # --------------------------------------------------------
        # 待機モード（scan_enabled == False）
        # --------------------------------------------------------
        if not scan_enabled:
            overlay = frame.copy()
            gray_color = (50, 50, 50)
            alpha = 0.6
            cv2.rectangle(overlay, (0, 0), (w, h), gray_color, -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # 待機中 or 送信完了メッセージを表示（モダンなデザイン）
            if completion_message_time is not None:
                # 送信完了：緑色のテキスト、半透明のダークグレー背景
                frame = put_japanese_text_modern(
                    frame, "送信完了",
                    position=(w // 2, h // 2),
                    font_size=100,
                    text_color=(76, 175, 80),  # モダンな緑色
                    bg_color=(30, 30, 30, 200),  # 半透明のダークグレー
                    shadow=True,
                    center=True
                )
            else:
                # 待機中：白色のテキスト、半透明のダークグレー背景
                frame = put_japanese_text_modern(
                    frame, "待機中",
                    position=(w // 2, h // 2),
                    font_size=100,
                    text_color=(255, 255, 255),  # 白色
                    bg_color=(30, 30, 30, 200),  # 半透明のダークグレー
                    shadow=True,
                    center=True
                )

            display_frame = resize_for_display(frame)
            
            # リサイズ後のフレームに最小化ボタンを描画
            display_frame, button_area = draw_minimize_button(display_frame)
            
            # マウスコールバックにボタンエリアを渡す
            cv2.setMouseCallback(window_name, mouse_callback, button_area)
            
            cv2.imshow(window_name, display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            continue

        # --------------------------------------------------------
        # スキャンモード（scan_enabled == True）
        # --------------------------------------------------------
        # 1. グレースケール
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. しきい値で白い紙を強調
        _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # 3. ノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 4. 輪郭抽出
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        doc_cnt = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (w * h * 0.05):
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and area > max_area:
                doc_cnt = approx
                max_area = area

        if doc_cnt is not None:
            current_corners = doc_cnt

            if previous_corners is not None:
                distance = calculate_corner_distance(previous_corners, current_corners)

                if distance <= position_threshold:
                    if stable_start_time is None:
                        stable_start_time = current_time

                    stable_duration = current_time - stable_start_time

                    if stable_duration >= stable_threshold and not image_saved:
                        try:
                            pts = current_corners.reshape(4, 2)
                            scanned = four_point_transform(frame, pts)
                            
                            # archiveディレクトリが存在しない場合は作成
                            archive_dir = "archive"
                            os.makedirs(archive_dir, exist_ok=True)
                            
                            # タイムスタンプ付きファイル名を生成
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            archive_filename = os.path.join(archive_dir, f"scan_{timestamp}.jpg")
                            
                            # archiveディレクトリに保存
                            cv2.imwrite(archive_filename, scanned, [cv2.IMWRITE_JPEG_QUALITY, 100])
                            
                            # 最新ファイルとしてscan_file.jpgにも保存（/get_scan_image用）
                            cv2.imwrite("scan_file.jpg", scanned, [cv2.IMWRITE_JPEG_QUALITY, 100])
                            
                            print(
                                f"📸 画像を保存しました: {archive_filename} "
                                f"(安定時間: {stable_duration:.2f}秒, 解像度: {scanned.shape[1]}x{scanned.shape[0]})"
                            )
                        except Exception as e:
                            print(f"画像保存エラー: {e}")
                            # エラー時もarchiveディレクトリに保存を試みる
                            try:
                                archive_dir = "archive"
                                os.makedirs(archive_dir, exist_ok=True)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                archive_filename = os.path.join(archive_dir, f"scan_{timestamp}.jpg")
                                cv2.imwrite(archive_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                                cv2.imwrite("scan_file.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                            except Exception as e2:
                                print(f"エラー時の画像保存も失敗: {e2}")

                        # スキャン完了 → WebSocketで通知
                        socketio.emit("scan_completed", {"status": "ok"})
                        print("📡 WebSocket: scan_completed を送信")

                        completion_message_time = current_time
                        image_saved = True
                        scan_enabled = False   # スキャン完了後、自動で待機モードへ戻る
                        stable_start_time = None
                        previous_corners = None
                        print("➡ スキャン完了 → 待機モードへ移行")

                else:
                    stable_start_time = None
                    image_saved = False
            else:
                stable_start_time = current_time

            previous_corners = current_corners.copy()

            for p in doc_cnt:
                x, y = p[0]
                cv2.circle(frame, (x, y), 15, (255, 0, 0), -1)

            cv2.polylines(frame, [doc_cnt], True, (255, 0, 0), 2)

            if stable_start_time is not None and scan_enabled:
                stable_duration = current_time - stable_start_time
                if stable_duration < stable_threshold:
                    cv2.putText(
                        frame, f"安定中: {stable_duration:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
        else:
            previous_corners = None
            stable_start_time = None
            image_saved = False

        display_frame = resize_for_display(frame)
        
        # リサイズ後のフレームに最小化ボタンを描画
        display_frame, button_area = draw_minimize_button(display_frame)
        
        # マウスコールバックにボタンエリアを渡す
        cv2.setMouseCallback(window_name, mouse_callback, button_area)
        
        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🎥 カメラループ終了")


# ============================================================
# メインエントリーポイント
# ============================================================
if __name__ == "__main__":
    # カメラ処理を別スレッドで起動
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()

    print("✅ Flask + SocketIO サーバ起動（/start_scan, /stop_scan, /get_scan_image, WebSocket: scan_completed）")

    # SocketIO（= Flask + WebSocket）サーバを起動
    # eventlet か gevent を入れておくとより安定
socketio.run(
    app,
    host="0.0.0.0",
    port=5001,
    # ssl_context=(
    #     "/home/to-murakami/Documents/reception-scan-camera/certs/cert.pem",
    #     "/home/to-murakami/Documents/reception-scan-camera/certs/key.pem"
    # )
    ssl_context=(
        "/home/to-murakami/Documents/reception-scan-camera/certs/fullchain.pem",
        "/home/to-murakami/Documents/reception-scan-camera/certs/server.key"
    )
)
