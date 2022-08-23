## Evaluate Tesseract Engine 

### Evaluate text recognition
```
python evaluate.py
```

### Evaluate text detection
```
git clone https://github.com/rafaelpadilla/Object-Detection-Metrics
cd Object-Detection-Metrics
python pascalvoc.py -gt ../gt/ -det ../det/
```

### TODO
- evaluate in dir format
- refactor code
- evaluate wer for postprocessing
- implement mapping module (dict-bases/fuzz)
- evaluate full pipeline score (F1/AUC)
- centralize script for text-detection & text-recognition