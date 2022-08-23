import os
import pytesseract
import numpy as np
import json
import glob
import pandas as pd
import xml.etree.ElementTree as ET

from constants import LABEL_PATH, IMAGE_PATH, OCR_PATH
from utils import get_iou, get_mapped_ocr_transcript, remove_accent_vietnamese, is_same_string
from wer import get_word_error_rate
from PIL import Image
from typing import List,Dict, Tuple

def get_ocr_output(image_path_dir:str) -> List[pd.DataFrame]:
    ''' Parse image dir into a list of DataFrame
    '''
    result = []
    for image_path in sorted(glob.glob(image_path_dir+'/*')):
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
        ocr_df['filename'] = image_path.split('/')[-1].replace('.jpg','').replace('.png','').replace('.jpeg','')
        ocr_df = ocr_df[['filename','xmin','ymin','xmax','ymax','conf','text']]
        result.append(ocr_df)
    return result

def get_label_df(label_path_dir:str) -> List[pd.DataFrame]:
    ''' Parse label dir/files to dataframe
    '''
    with open(OCR_PATH,'r',encoding='utf-8') as f:
        ocr_data = json.load(f)
    result = []
    for xml_file in sorted(glob.glob(label_path_dir + '/*.xml')):
        xml_list = []
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(bbx.find('xmin').text)
            ymin = int(bbx.find('ymin').text)
            xmax = int(bbx.find('xmax').text)
            ymax = int(bbx.find('ymax').text)
            label = member.find('name').text

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
        fname = xml_file.split('/')[-1].replace('.xml','')
        # mapping label-ocr and xml 
        bbox_transcription = get_mapped_ocr_transcript(ocr_data,fname)
        xml_df['text'] = bbox_transcription
        result.append(xml_df)

    return result

def refine_to_voc_format(label_dfs:List[pd.DataFrame],pred_dfs:List[pd.DataFrame]) -> Tuple[List[pd.DataFrame],List[pd.DataFrame]]:
    ''' Treat as object detection 
    '''
    # get text from pred that overlap with DOCUMENT_TITLE and HOSPITAL_NAME tags:
    for pred_df, label_df in zip(pred_dfs,label_dfs):
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
                        
        pred_df = pred_df.loc[pred_df['class'].isin(['DOCUMENT_CONTENT','HOSPITAL_CONTENT'])]
        label_df = label_df.loc[label_df['class'].isin(['DOCUMENT_CONTENT','HOSPITAL_CONTENT'])]
    return label_dfs, pred_dfs

def save_to_txt(dfs:List[pd.DataFrame],dir:str,is_det=False):
    for df in dfs:
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
            fname = row['filename']
        with open(os.path.join(dir,fname+'.txt'),'w') as f:
            for line in data:
                f.write(line + '\n')

def calculate_ocr_metrics(label_dfs:List[pd.DataFrame],pred_dfs:List[pd.DataFrame]) -> Dict:
    result = {
        'pred_document': [],
        'label_document' : [],
        'wer_document' : [],
        'pred_hospital': [],
        'label_hospital' : [],
        'wer_hospital' : [],
    }
    final_doc_wer = []
    final_hos_wer = []
    final_doc_wer_rm_accent = []
    final_hos_wer_rm_accent = []
    final_doc_auc = []
    final_hos_auc = []

    for pred_df, label_df in zip(pred_dfs,label_dfs):
        out_dict, score = get_ocr_metrics(label_df,pred_df)
        f_doc_wer,f_doc_wer_rm_accent,is_doc_mapping,\
            f_hos_wer,f_hos_wer_rm_accent,is_hos_mapping = score
        fname = pred_df['filename'].values[0]
        result['pred_document'].extend(out_dict['pred_document'])
        result['label_document'].extend(out_dict['label_document'])
        result['wer_document'].extend(out_dict['wer_document'])
        result['pred_hospital'].extend(out_dict['pred_hospital'])
        result['label_hospital'].extend(out_dict['label_hospital'])
        result['wer_hospital'].extend(out_dict['wer_hospital'])

        final_doc_wer.append(f_doc_wer)
        final_hos_wer.append(f_hos_wer)
        final_doc_wer_rm_accent.append(f_doc_wer_rm_accent)
        final_hos_wer_rm_accent.append(f_hos_wer_rm_accent)
        final_doc_auc.append(is_doc_mapping)
        final_hos_auc.append(is_hos_mapping)

    print('DOCUMENT WER =',sum(final_doc_wer)/len(final_doc_wer))
    print('HOSPITAL WER =',sum(final_hos_wer)/len(final_hos_wer))
    print('DOCUMENT REMOVE ACCENT WER =',sum(final_doc_wer_rm_accent)/len(final_doc_wer_rm_accent))
    print('HOSPITAL REMOVE ACCENT WER =',sum(final_hos_wer_rm_accent)/len(final_hos_wer_rm_accent))
    print('DOCUMENT AUC =',sum(final_doc_auc)/len(final_doc_auc))
    print('HOSPITAL AUC =',sum(final_hos_auc)/len(final_hos_auc))

    df = pd.DataFrame({
        'pred': result['pred_document'] + result['pred_hospital'],
        'label': result['label_document'] + result['label_hospital'],
        'class' : ['DOCUMENT'] * len(result['label_document']) + ['HOSPITAL'] * len(result['label_hospital']),
        'wer': result['wer_document'] + result['wer_hospital'],
    })
    df.to_csv('result/ocr_result.csv')
    return result

def get_ocr_metrics(label_df,pred_df) -> Tuple[Dict,List]:
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
    # wer after remove accent
    final_doc_wer_rm_accent = get_word_error_rate(
        remove_accent_vietnamese(result['label_document'][-1]),\
        remove_accent_vietnamese(result['pred_document'][-1])
    )
    # AUC after fuzzy-mapping
    is_doc_mapping = int(is_same_string(result['label_document'][-1],result['pred_document'][-1]))

    result['pred_hospital'].append(' '.join(result['pred_hospital']))
    result['label_hospital'].append(' '.join(result['label_hospital']))
    final_hos_wer = get_word_error_rate(result['label_hospital'][-1],result['pred_hospital'][-1])
    result['wer_hospital'].append(final_hos_wer)
    # wer after remove accent
    final_hos_wer_rm_accent = get_word_error_rate(
        remove_accent_vietnamese(result['label_hospital'][-1]),\
        remove_accent_vietnamese(result['pred_hospital'][-1])
    )
    # AUC after fuzzy-mapping
    is_hos_mapping = int(is_same_string(result['label_hospital'][-1],result['pred_hospital'][-1]))

    score = (final_doc_wer,final_doc_wer_rm_accent,is_doc_mapping,\
        final_hos_wer,final_hos_wer_rm_accent,is_hos_mapping)
    return result, score

if __name__ == '__main__':

    # read annotations
    label_df = get_label_df(LABEL_PATH) 

    # get predictions
    pred_df = get_ocr_output(IMAGE_PATH)

    label_df,pred_df = refine_to_voc_format(label_df,pred_df)

    save_to_txt(label_df,'gt')
    save_to_txt(pred_df,'det',True)
    # test detection
    # python pascalvoc.py -gt ../gt/ -det ../det/

    # # test ocr - WER
    result = calculate_ocr_metrics(label_df,pred_df)
    # print(result)