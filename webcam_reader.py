import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont
import logging
import os
from gaussian_reconstruction import HighResReconstructor
import pickle
import time

# ロギングの設定（エラーのみ表示）
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def put_japanese_text(img, text, position, font_size=32, color=(255, 0, 0)):
    # PILイメージに変換
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # フォントの設定（Windowsのデフォルトフォントを使用）
    try:
        font = ImageFont.truetype("msgothic.ttc", font_size)
    except:
        font = ImageFont.load_default()
    
    # テキストを描画
    draw.text(position, text, font=font, fill=color)
    
    # OpenCVイメージに戻す
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def preprocess_image(image):
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ノイズ除去
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # コントラスト改善
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 二値化
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def process_frame(frame, focus_fixed=False, ocr_enabled=True, high_res_enabled=False, reconstructor=None, show_segmentation=True):
    try:
        # 画像の前処理
        processed = preprocess_image(frame)
        
        # --- YOLOv8-segによる物体セグメンテーション ---
        if show_segmentation:
            try:
                from ultralytics import YOLO
                model = YOLO('yolov8n-seg.pt')
                results = model(frame)
                h, w = frame.shape[:2]
                mask = None
                if hasattr(results[0], 'masks') and results[0].masks is not None:
                    mask_data = results[0].masks.data.cpu().numpy()
                    if mask_data.shape[0] > 0:
                        areas = mask_data.sum(axis=(1,2))
                        idx = areas.argmax()
                        mask = (mask_data[idx] * 255).astype(np.uint8)
                        mask = cv2.resize(mask, (w, h))
                if mask is not None:
                    color_mask = np.zeros_like(frame)
                    color_mask[:, :, 1] = 255  # 緑色
                    alpha = 0.4
                    mask_bool = mask > 0
                    frame = np.where(mask_bool[..., None], (frame * (1 - alpha) + color_mask * alpha).astype(np.uint8), frame)
            except Exception as e:
                logger.warning(f"YOLOv8-seg segmentation error: {str(e)}")
        
        # OCRが有効な場合のみ文字認識を実行
        if ocr_enabled:
            # 文字認識（OCR）- 日本語と英数字を別々に認識
            try:
                jpn_text = pytesseract.image_to_string(processed, lang='jpn', config='--psm 6')
                jpn_text = jpn_text.strip()
            except Exception as e:
                logger.error(f"日本語認識エラー: {str(e)}")
                jpn_text = ""
            
            try:
                eng_text = pytesseract.image_to_string(processed, lang='eng', config='--psm 6')
                eng_text = eng_text.strip()
            except Exception as e:
                logger.error(f"英数字認識エラー: {str(e)}")
                eng_text = ""
            
            # 認識結果を表示
            y_offset = 30
            if jpn_text:
                frame = put_japanese_text(frame, f"日本語: {jpn_text}", (10, y_offset), color=(0, 0, 255))
                y_offset += 40
            
            if eng_text:
                frame = put_japanese_text(frame, f"英数字: {eng_text}", (10, y_offset), color=(255, 0, 0))
                y_offset += 40
        
        # バーコード認識
        barcodes = decode(frame)
        for barcode in barcodes:
            try:
                # バーコードの位置情報を取得
                (x, y, w, h) = barcode.rect
                
                # バーコードの種類を取得
                barcode_type = barcode.type
                
                # バーコードのデータを取得
                barcode_data = barcode.data.decode('utf-8')
                
                # バーコードの周りに矩形を描画
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # バーコードのデータと種類を表示
                text = f"{barcode_type}: {barcode_data}"
                cv2.putText(frame, text, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                logger.warning(f"Error processing barcode: {str(e)}")
                continue
        
        # フレーム数表示のみ（add_frameはmainループでのみ呼ぶ）
        if high_res_enabled and reconstructor is not None:
            frame_count = len(reconstructor.frames)
            frame = put_japanese_text(frame, f"フレーム数: {frame_count}", 
                                    (10, frame.shape[0] - 80),
                                    font_size=24, color=(255, 255, 255))
        
        return frame
    
    except Exception as e:
        logger.error(f"フレーム処理エラー: {str(e)}")
        return frame

def select_four_points(image):
    points = []
    clone = image.copy()
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Select Label Corners', clone)
    cv2.imshow('Select Label Corners', clone)
    cv2.setMouseCallback('Select Label Corners', click_event)
    while True:
        cv2.imshow('Select Label Corners', clone)
        key = cv2.waitKey(1) & 0xFF
        if len(points) == 4:
            break
        if key == 27:  # ESCでキャンセル
            points = []
            break
    cv2.destroyWindow('Select Label Corners')
    return points

def main():
    try:
        # カメラを開く
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("カメラを開けませんでした。")
            return
        
        # カメラの設定
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # フォーカス固定をデフォルト
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 解像度を設定
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        focus_fixed = True  # デフォルトでフォーカス固定
        ocr_enabled = False  # OCRの初期状態を無効に設定
        high_res_enabled = False  # 高精細画像生成の初期状態を無効に設定
        # 合成モードリスト
        blend_modes = ["min", "median", "max"]
        blend_mode_idx = 0
        # 高精細画像再構築器の初期化
        reconstructor = HighResReconstructor(blend_mode=blend_modes[blend_mode_idx])
        
        # キー操作の説明
        key_instructions = [
            "キー操作:",
            "o: OCR有効/無効",
            "f: フォーカス固定/解除",
            "h: 高精細画像生成 有効/無効",
            "r: 高精細画像を再構築/更新",
            "b: 合成方法切替 (min/median/max)",
            "a: ラベル検出モード切替 (自動/手動)",
            "q: 終了"
        ]
        
        # ラベル検出モード（True: 自動, False: 手動）
        auto_detect_mode = True
        
        # 最初のフレームでラベルの4点を選択
        label_corners = None
        while label_corners is None or len(label_corners) != 4:
            ret, frame = cap.read()
            if not ret:
                logger.error("フレームを取得できませんでした。")
                return
                
            if auto_detect_mode:
                # 自動検出を試みる
                label_corners = reconstructor.auto_detect_label_corners(frame)
                if label_corners is None:
                    # 自動検出に失敗した場合、手動選択に切り替え
                    logger.warning("自動検出に失敗しました。手動選択に切り替えます。")
                    auto_detect_mode = False
                    label_corners = select_four_points(frame)
            else:
                # 手動選択
                label_corners = select_four_points(frame)
        
        # 4点をreconstructorに渡す
        reconstructor.label_corners = label_corners
        
        # 高精細画像ウィンドウの初期化
        cv2.namedWindow('High Resolution Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('High Resolution Image', 1280, 720)
        
        reconstruct_requested = False  # 再構築要求フラグ
        high_res_image = None  # 最新の高精細画像

        while True:
            # フレームを取得
            ret, frame = cap.read()
            if not ret:
                logger.error("フレームを取得できませんでした。")
                break

            # 高精細画像生成が有効な場合、生フレームをadd_frameに渡す
            if high_res_enabled and reconstructor is not None:
                reconstructor.add_frame(frame.copy())

            # フレームを処理（ライブビューはマスク表示ON）
            processed_frame = process_frame(frame, focus_fixed, ocr_enabled, high_res_enabled, reconstructor, show_segmentation=True)

            # 状態表示
            status_text = []
            if focus_fixed:
                status_text.append("フォーカス固定")
            if not ocr_enabled:
                status_text.append("OCR無効")
            if high_res_enabled:
                status_text.append("高精細生成中")
            status_text.append(f"合成方法: {blend_modes[blend_mode_idx]}")
            status_text.append(f"ラベル検出: {'自動' if auto_detect_mode else '手動'}")

            # 状態テキストを表示
            if status_text:
                status = " | ".join(status_text)
                processed_frame = put_japanese_text(processed_frame, status, 
                                                  (10, processed_frame.shape[0] - 40),
                                                  font_size=24, color=(255, 255, 255))

            # キー操作の説明を表示
            y_offset = 10
            for instruction in key_instructions:
                processed_frame = put_japanese_text(processed_frame, instruction,
                                                  (10, y_offset),
                                                  font_size=20, color=(255, 255, 255))
                y_offset += 25

            # 結果を表示
            cv2.imshow('Webcam Reader', processed_frame)

            # キー入力の処理（非同期的に即時反応）
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                if key == ord('q'):  # 'q'キーで終了
                    break
                elif key == ord('f'):  # 'f'キーでフォーカス固定/解除
                    focus_fixed = not focus_fixed
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0 if focus_fixed else 1)
                elif key == ord('o'):  # 'o'キーでOCR有効/無効
                    ocr_enabled = not ocr_enabled
                elif key == ord('h'):  # 'h'キーで高精細画像生成有効/無効
                    high_res_enabled = not high_res_enabled
                    if not high_res_enabled:
                        reconstructor = HighResReconstructor(blend_mode=blend_modes[blend_mode_idx])
                        high_res_image = None
                elif key == ord('b'):  # 'b'キーで合成方法切替
                    blend_mode_idx = (blend_mode_idx + 1) % len(blend_modes)
                    reconstructor = HighResReconstructor(blend_mode=blend_modes[blend_mode_idx])
                    high_res_image = None
                elif key == ord('a'):  # 'a'キーでラベル検出モード切替
                    auto_detect_mode = not auto_detect_mode
                    # モード切替時にラベル検出を再実行
                    label_corners = None
                    while label_corners is None or len(label_corners) != 4:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if auto_detect_mode:
                            label_corners = reconstructor.auto_detect_label_corners(frame)
                            if label_corners is None:
                                logger.warning("自動検出に失敗しました。手動選択に切り替えます。")
                                auto_detect_mode = False
                                label_corners = select_four_points(frame)
                        else:
                            label_corners = select_four_points(frame)
                    if label_corners is not None:
                        reconstructor.label_corners = label_corners
                elif key == ord('r'):  # 'r'キーで高精細画像を再構築/更新
                    reconstruct_requested = True

            # 再構築要求があれば即時再生成
            if reconstruct_requested and len(reconstructor.frames) > 0:
                high_res_image = reconstructor.reconstruct()
                if high_res_image is not None:
                    cv2.imshow('High Resolution Image', high_res_image)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'high_res_{timestamp}.png', high_res_image)
                reconstruct_requested = False
    
    except Exception as e:
        logger.error(f"メインループエラー: {str(e)}")
    
    finally:
        # リソースを解放
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 