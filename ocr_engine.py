import pytesseract
import pandas as pd

from PIL import Image

class OCREngine:
    def __init__(self,lang='vie'):
        self.ocr = None
        self.lang = lang

    def predict(self,image:Image) -> pd.DataFrame:
        pass
    