import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from pyzbar.pyzbar import decode
from PIL import Image, ImageDraw, ImageFont
import logging
import os

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

def process_frame(frame, focus_fixed=False, ocr_enabled=True):
    try:
        # 画像の前処理
        processed = preprocess_image(frame)
        
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
        
        # バーコード読み取り
        try:
            barcodes = decode(frame)
            for barcode in barcodes:
                # バーコードの位置情報を取得
                (x, y, w, h) = barcode.rect
                
                # バーコードの周りに矩形を描画
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # バーコードのデータを取得
                barcode_data = barcode.data.decode('utf-8')
                barcode_type = barcode.type
                
                # バーコードの情報を表示
                text = f"{barcode_type}: {barcode_data}"
                frame = put_japanese_text(frame, text, (x, y - 30), font_size=24, color=(0, 255, 0))
        except Exception as e:
            logger.error(f"バーコード認識エラー: {str(e)}")
        
        return frame
    
    except Exception as e:
        logger.error(f"フレーム処理エラー: {str(e)}")
        return frame

def main():
    try:
        # カメラを開く
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("カメラを開けませんでした。")
            return
        
        # カメラの設定
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # オートフォーカスを有効化
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 解像度を設定
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        focus_fixed = False
        ocr_enabled = False  # OCRの初期状態を無効に設定
        
        # キー操作の説明
        key_instructions = [
            "キー操作:",
            "o: OCR有効/無効",
            "f: フォーカス固定/解除",
            "q: 終了"
        ]
        
        while True:
            # フレームを取得
            ret, frame = cap.read()
            
            if not ret:
                logger.error("フレームを取得できませんでした。")
                break
            
            # フレームを処理
            processed_frame = process_frame(frame, focus_fixed, ocr_enabled)
            
            # 状態表示
            status_text = []
            if focus_fixed:
                status_text.append("フォーカス固定")
            if not ocr_enabled:
                status_text.append("OCR無効")
            
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
            
            # キー入力の処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 'q'キーで終了
                break
            elif key == ord('f'):  # 'f'キーでフォーカス固定/解除
                focus_fixed = not focus_fixed
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0 if focus_fixed else 1)
            elif key == ord('o'):  # 'o'キーでOCR有効/無効
                ocr_enabled = not ocr_enabled
    
    except Exception as e:
        logger.error(f"メインループエラー: {str(e)}")
    
    finally:
        # リソースを解放
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 