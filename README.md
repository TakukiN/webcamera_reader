# Webcam Reader

リアルタイムでテキストとバーコードを認識するWebカメラアプリケーション

## 機能

- リアルタイムテキスト認識（OCR）
- バーコード読み取り
- 日本語テキスト表示対応
- フォーカス制御
- OCRの有効/無効切り替え

## 必要条件

- Python 3.x
- OpenCV
- Tesseract OCR
- pyzbar
- PIL (Pillow)

## インストール

1. 必要なパッケージをインストール:
```bash
pip install -r requirements.txt
```

2. Tesseract OCRをインストール:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- 日本語データをインストールすることを推奨

## 使用方法

1. プログラムを実行:
```bash
python webcam_reader.py
```

2. キー操作:
- 'o': OCRの有効/無効を切り替え
- 'f': フォーカスの固定/解除を切り替え
- 'q': プログラムを終了

## ライセンス

MIT License 