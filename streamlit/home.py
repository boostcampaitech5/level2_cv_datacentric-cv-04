import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
import os.path as osp
import json

import torch
from imageio.v2 import imread
from model import EAST
from detect import detect
DATASET_DIR = '../data/medical'  # FIXME
# st.set_page_config(initial_sidebar_state="collapsed")
st.title("OCR with CV-04 Team model")

def load_model(model_file):
    model = EAST(pretrained=False).to('cpu')
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()

    return model

def do_inference(model, img, input_size=2048):

    image_fnames, by_sample_bboxes = [], []

    
    #img = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(dim=0)
    
    image_fnames, by_sample_bboxes = ['test'], []
    images = []
    images.append(img)
    
    by_sample_bboxes.extend(detect(model, images, input_size))
    #st.text(by_sample_bboxes)
    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result

def draw_bbox(image, bbox, color=(0, 0, 255), thickness=1, thickness_sub=None, double_lined=False,
              write_point_numbers=False):
    """이미지에 하나의 bounding box를 그려넣는 함수
    """
    thickness_sub = thickness_sub or thickness * 3
    basis = max(image.shape[:2])
    fontsize = basis / 4000
    x_offset, y_offset = int(fontsize * 12), int(fontsize * 10)
    color_sub = (255 - color[0], 255 - color[1], 255 - color[2])

    points = [(int(np.rint(p[0])), int(np.rint(p[1]))) for p in bbox]

    for idx in range(len(points)):
        if double_lined:
            cv2.line(image, points[idx], points[(idx + 1) % len(points)], color_sub,
                     thickness=thickness_sub)
        cv2.line(image, points[idx], points[(idx + 1) % len(points)], color, thickness=thickness)

    if write_point_numbers:
        for idx in range(len(points)):
            loc = (points[idx][0] - x_offset, points[idx][1] - y_offset)
            if double_lined:
                cv2.putText(image, str(idx), loc, cv2.FONT_HERSHEY_SIMPLEX, fontsize, color_sub,
                            thickness_sub, cv2.LINE_AA)
            cv2.putText(image, str(idx), loc, cv2.FONT_HERSHEY_SIMPLEX, fontsize, color, thickness,
                        cv2.LINE_AA)


def draw_bboxes(image, bboxes, color=(0, 0, 255), thickness=1, thickness_sub=None,
                double_lined=False, write_point_numbers=False):
    """이미지에 다수의 bounding box들을 그려넣는 함수
    """
    for bbox in bboxes:
        draw_bbox(image, bbox, color=color, thickness=thickness, thickness_sub=thickness_sub,
                  double_lined=double_lined, write_point_numbers=write_point_numbers)

@st.cache_data
def load_anno():
    # 데이터 셋 annotation 파일 불러오기
    ufo_fpath = osp.join(DATASET_DIR, 'ufo/train_merged.json')
    with open(ufo_fpath, 'r') as f:
        ufo_anno = json.load(f)
    # 정렬된 이미지이름 리스트 
    sample_ids = sorted(ufo_anno['images'])
    return ufo_anno, sample_ids

def main():
    # Using object notation
    add_selectbox = st.sidebar.selectbox(
        "원하는 옵션을 고르세요",
        ("model inference", "view data")
    )
    if add_selectbox == "model inference":
        uploaded_model = st.file_uploader("Upload your model.", accept_multiple_files=False, type=['pth', 'pt'])
        if uploaded_model is not None:
            model = load_model(uploaded_model)
            model.eval()

        
        uploaded_image = st.file_uploader("Choose an image...", accept_multiple_files=False)
        if uploaded_image is not None and uploaded_model is not None:
            img = np.array(Image.open(uploaded_image))
            fin_img = img.copy()

            ufo_result = do_inference(model, img)
            st.text(img.shape)

            for _, v in ufo_result['images']['test']['words'].items():
                v = v['points']
                v.append(v[0])
                cv2.polylines(fin_img, [np.array(v, dtype=np.int32)], True, (0, 0, 255), 2)
            st.image(fin_img)


    elif add_selectbox == "view data":
        #annotation json, 이미지 이름 불러오기
        ufo_anno, sample_ids = load_anno()

        SAMPLE_IDX = st.number_input('원하는 사진의 인덱스를 입력해주세요.(0~300)', 0, 300)

        sample_id = sample_ids[SAMPLE_IDX]  # `sample_id`가 곧 이미지 파일명
        image_fpath = osp.join(DATASET_DIR, 'img/img', sample_id)
        st.text(f"Image path: {image_fpath}")

        image = imread(image_fpath)
        st.text(f'Image shape:\t{image.shape}')

        ignore_tags = ['masked', 'excluded-region', 'maintable', 'stamp']

        bboxes, labels = [], []
        for word_info in ufo_anno['images'][sample_id]['words'].values():
            word_tags = word_info['tags']
            ignore_sample = any(elem for elem in word_tags if elem in ignore_tags)
            if ignore_sample:
                continue
            
            if len(word_info['points']) > 4:
                continue
                
            bboxes.append(np.array(word_info['points']))
            labels.append(int(not word_info['illegibility']))
        bboxes, labels = np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.float32)

        st.text('Bounding boxes:\t{}'.format(bboxes.shape))
        st.text('Labels:\t{}'.format(labels.shape))

        vis = image.copy()
        draw_bboxes(vis, bboxes, double_lined=True, thickness=2, thickness_sub=5, write_point_numbers=True)
        st.image(vis)

    elif add_selectbox == "test":
        ufo_anno, sample_ids = load_anno()
        uploaded_model = st.file_uploader("Upload your model.", accept_multiple_files=False, type=['pth', 'pt'])
        if uploaded_model is not None:
            model = load_model(uploaded_model)
            model.eval()
        
				# 파일 이름 리스트를 dir_list
        dir_list = os.listdir('/opt/ml/input/data/medical_fold1/img/val')
        
				# 아래는 복붙함
        for sample_id in dir_list:
            
            image_fpath = osp.join(DATASET_DIR, 'img/val', sample_id)
            st.text(f"Image path: {image_fpath}")

            image = imread(image_fpath)
            st.text(f'Image shape:\t{image.shape}')

            ignore_tags = ['masked', 'excluded-region', 'maintable', 'stamp']

            bboxes, labels = [], []
            for word_info in ufo_anno['images'][sample_id]['words'].values():
                word_tags = word_info['tags']
                ignore_sample = any(elem for elem in word_tags if elem in ignore_tags)
                if ignore_sample:
                    continue
                
                if len(word_info['points']) > 4:
                    continue
                    
                bboxes.append(np.array(word_info['points']))
                labels.append(int(not word_info['illegibility']))
            bboxes, labels = np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.float32)

            st.text('Bounding boxes:\t{}'.format(bboxes.shape))
            st.text('Labels:\t{}'.format(labels.shape))

            vis = image.copy()
            draw_bboxes(vis, bboxes, double_lined=True, thickness=2, thickness_sub=5, write_point_numbers=True)
            st.image(vis)
            
            st.text('uploaded image')
            if image_fpath is not None and image_fpath is not None:
                img = np.array(Image.open(image_fpath))
                fin_img = img.copy()

                ufo_result = do_inference(model, img)
                st.text(img.shape)

                for _, v in ufo_result['images']['test']['words'].items():
                    v = v['points']
                    v.append(v[0])
                    cv2.polylines(fin_img, [np.array(v, dtype=np.int32)], True, (0, 0, 255), 2)
                st.image(fin_img)

if __name__ == '__main__':
    main()