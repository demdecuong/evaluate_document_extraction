from PIL import Image

import os
import pytesseract
import numpy as np
import json
import pandas as pd
import xml.etree.ElementTree as ET
from wer import get_word_error_rate

# https://github.com/rafaelpadilla/Object-Detection-Metrics

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

# sudo apt-get install tesseract-ocr-vie
def convert_to_tesseract_format(xmin,ymin,xmax,ymax,width,height):
    x = xmin
    y = ymin
    w = xmax - xmin
    h = ymax - ymin
    return x,y,w,h

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def get_ocr_output(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    width, height = image.size
    w_scale = 1000/width
    h_scale = 1000/height

    ocr_df = pytesseract.image_to_data(image, lang='vie', output_type='data.frame') \
                
    ocr_df = ocr_df.dropna() \
                .assign(left_scaled = ocr_df.left*w_scale,
                        width_scaled = ocr_df.width*w_scale,
                        top_scaled = ocr_df.top*h_scale,
                        height_scaled = ocr_df.height*h_scale,
                        right_scaled = lambda x: x.left_scaled + x.width_scaled,
                        bottom_scaled = lambda x: x.top_scaled + x.height_scaled)

    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    ocr_df['xmin'] = ocr_df['left']
    ocr_df['ymin'] = ocr_df['top']
    ocr_df['xmax'] = ocr_df['left'] + ocr_df['width']
    ocr_df['ymax'] = ocr_df['top'] + ocr_df['height']
    ocr_df = ocr_df[['xmin','ymin','xmax','ymax','conf','text']]
    return ocr_df

def xml_to_csv(path):
    xml_list = []
    # for xml_file in glob.glob(path + '/*.xml'):
    tree = ET.parse(path+'.xml')
    root = tree.getroot()
    for member in root.findall('object'):
        bbx = member.find('bndbox')
        xmin = int(bbx.find('xmin').text)
        ymin = int(bbx.find('ymin').text)
        xmax = int(bbx.find('xmax').text)
        ymax = int(bbx.find('ymax').text)
        label = member.find('name').text
    
        # x,y,w,h = convert_to_tesseract_format(xmin,ymin,xmax,ymax,\
        #         int(root.find('size')[0].text),\
        #         int(root.find('size')[1].text))
        value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    label,
                    xmin,
                    ymin,
                    xmax,
                    ymax
                    )
        xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    with open(path+'.json','r') as f:
        data = json.load(f)[0] 
    # print(data['transcription'])    
    xml_df['text'] = data['transcription']

    return xml_df

def refine_to_voc_format(label_df,pred_df):
    ''' Treat as object detection 
    '''
    # get text from pred that overlap with DOCUMENT_TITLE and HOSPITAL_NAME tags:
    pred_tags = []
    pred_df['class'] = [None] * len(pred_df)
    for tag in ['DOCUMENT_TITLE','HOSPITAL_NAME']:
        label_bbox = label_df.loc[label_df['class']==tag][['xmin','ymin','xmax','ymax']]
        # txmin, tymin, txmax, tymax = label_bbox.values.tolist()[0]
        for i, row in pred_df.iterrows():
            bbox = row[['xmin','ymin','xmax','ymax']]
            # print(get_iou(bbox.values.tolist(),label_bbox.values.tolist()[0]))
            if get_iou(bbox.values.tolist(),label_bbox.values.tolist()[0]) > 0:
                if tag == 'DOCUMENT_TITLE':
                    pred_df.at[i,'class'] = 'DOCUMENT_CONTENT'
                else:
                    pred_df.at[i,'class'] ='HOSPITAL_CONTENT'
                    
    # calculate mAP:  TODO
    pred_df = pred_df.loc[pred_df['class'].isin(['DOCUMENT_CONTENT','HOSPITAL_CONTENT'])]
    label_df = label_df.loc[label_df['class'].isin(['DOCUMENT_CONTENT','HOSPITAL_CONTENT'])]
    return label_df, pred_df

def save_to_txt(df,dir,is_det=False):
    data = []
    for _, row in df.iterrows():
        if is_det:
            # <class_name> <confidence> <left> <top> <width> <height>
            line = [row['class'], row['conf'], row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin']]
        else:
            # <class_name> <left> <top> <width> <height>
            line = [row['class'], row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin']]
        line = [str(item) for item in line]
        data.append(' '.join(line))
    with open(os.path.join(dir,'test.txt'),'w') as f:
        for line in data:
            f.write(line + '\n')

def calculate_ocr_metrics(label_df,pred_df):
    result = {
        'pred_document': [],
        'label_document' : [],
        'wer_document' : [],
        'pred_hospital': [],
        'label_hospital' : [],
        'wer_hospital' : [],
    }
    for tag in ['DOCUMENT_CONTENT','HOSPITAL_CONTENT']: # still not optimal
        for _, label_row in label_df.iterrows():
            if label_row['class'] == tag:
                label_bbox = label_row[['xmin','ymin','xmax','ymax']]
                label_text = label_row['text']
                for i, row in pred_df.iterrows():
                    bbox = row[['xmin','ymin','xmax','ymax']]
                    pred_text = row['text']
                    if get_iou(bbox.values.tolist(),label_bbox.values.tolist()) > 0: 
                        error_rate = get_word_error_rate(label_text,pred_text)
                        if tag == 'DOCUMENT_CONTENT':
                            result['pred_document'].append(pred_text)
                            result['label_document'].append(label_text)
                            result['wer_document'].append(error_rate)
                        else:
                            result['pred_hospital'].append(pred_text)
                            result['label_hospital'].append(label_text)
                            result['wer_hospital'].append(error_rate)

    # calculate all
    result['pred_document'].append(' '.join(result['pred_document']))
    result['label_document'].append(' '.join(result['label_document']))
    final_doc_wer = get_word_error_rate(result['label_document'][-1],result['pred_document'][-1])
    result['wer_document'].append(final_doc_wer)
    
    result['pred_hospital'].append(' '.join(result['pred_hospital']))
    result['label_hospital'].append(' '.join(result['label_hospital']))
    final_hos_wer = get_word_error_rate(result['label_hospital'][-1],result['pred_hospital'][-1])
    result['wer_hospital'].append(final_hos_wer)
    
    print('DOCUMENT WER =',final_doc_wer)
    print('HOSPITAL WER =',final_hos_wer)

    df = pd.DataFrame({
        'pred': result['pred_document'] + result['pred_hospital'],
        'label': result['label_document'] + result['label_hospital'],
        'class' : ['DOCUMENT'] * len(result['label_document']) + ['HOSPITAL'] * len(result['label_hospital']),
        'wer': result['wer_document'] + result['wer_hospital'],
    })
    df.to_csv('ocr_result.csv')
    return result

if __name__ == '__main__':
    label_df = xml_to_csv('label')

    pred_df = get_ocr_output('example.jpg')
    label_df,pred_df = refine_to_voc_format(label_df,pred_df)

    save_to_txt(label_df,'gt')
    save_to_txt(pred_df,'det',True)
    # test detection
    # python pascalvoc.py -gt ../gt/ -det ../det/

    # test ocr - WER
    result = calculate_ocr_metrics(label_df,pred_df)
    print(result)