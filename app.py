from PIL import Image
import streamlit as st
from pathlib import Path

import os
import cv2
import time
import torch
import numpy as np
from adet.config import get_cfg
from demo.predictor import VisualizationDemo
from adet.utils.visualizer import TextVisualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
# from detectron2.data.detection_utils import read_image
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy

img_path = "./sample_images/documents.jpg"
output_path = "output_images"
config_file = 'configs/BAText/CTW1500/attn_R_50.yaml'
opts = ['MODEL.WEIGHTS', 'ctw1500_attn_R_50.pth']
confidence_threshold = 0.5

def decode_recognition(vis, rec):
    s = ''
    for c in rec:
        c = int(c)
        if c < vis.voc_size - 1:
            if vis.voc_size == 96:
                s += vis.CTLABELS[c]
            else:
                s += str(chr(vis.CTLABELS[c]))
        elif c == vis.voc_size -1:
            s += u'口'
    return s

def setup_cfg(config_file, opts, confidence_threshold=0.5):

    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    return cfg

def read_img(img_upload, format="RGB"):
    image = Image.open(img_upload)
    image = _apply_exif_orientation(image)
    return convert_PIL_to_numpy(image, format)

def dowload_img(url, format="RGB"):
    import requests
    from io import BytesIO

    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = _apply_exif_orientation(image)
    return convert_PIL_to_numpy(image, format)
    

def dectection(img, img_name):
    cfg = setup_cfg(config_file, opts, confidence_threshold)
    demo = VisualizationDemo(cfg)
    # img = read_image(img_path, format="BGR")
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img)
    inference_time = time.time() - start_time
    print(f"{img_path}: detected {len(predictions['instances'])} instances in {inference_time:.2f}s")
    out_filename = os.path.join(output_path, os.path.basename(img_name))

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
    instance_mode = ColorMode.IMAGE
    visualizer = TextVisualizer(img, metadata, instance_mode=instance_mode, cfg=cfg)
    
    json_str = {
        'image name': img_name,
        'number instances': len(predictions['instances']),
        'inference time': str(round(inference_time, 2)) + "s",
        'output path': out_filename,
        'results':[]
    }

    if "instances" in predictions:
        instances = predictions["instances"].to(torch.device("cpu"))
        pred_texts = [decode_recognition(visualizer, rec) for rec in instances.recs]
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
        img_output = vis_output.get_image()[:, :, ::-1]
        # vis_output.save(out_filename)

        print("Text prediction:", pred_texts)
        print("Scores: ", predictions["instances"].scores)
        
        for pred_text, score in zip(pred_texts, predictions["instances"].scores.cpu().detach().numpy()):
            i_json = {'text': pred_text, 'score': score}
            json_str['results'].append(i_json)

    return out_filename, img_output, json_str

def load_image(img_path):
    img = Image.open(img_path)
    return img

def main():
    st.set_page_config(layout="wide")
    st.title("Scene Text Spotting with ABCNet")
    st.caption("Nhận dạng văn bản trong hình ảnh sử dụng Adaptive Bezier-Curve Network")

    select_str = '<h3 style="text-align: left;">Select images</h3>'
    st.markdown(select_str, unsafe_allow_html=True)

    choose_img = st.radio(
        "Bạn muốn chọn hình ảnh như thế nào?",
        ('Upload image', 'From link')
    )

    if choose_img == 'From link':
        url_input = st.text_input(
            "Enter image link here:",
            key="placeholder",
        )

        if url_input:
            url_img = url_input.strip()
            st.write("You entered: ", url_img)
            try:
                input_img = dowload_img(url_img)

                with st.spinner('Detecting...'):
                    output_file, img_output, json_str = dectection(input_img, url_input)

                # output_img = load_image(output_file)
                print("Ouput image path:", output_file)

                results_str = '<h3 style="text-align: left;">Result</h3>'
                st.markdown(results_str, unsafe_allow_html=True)

                col1, mid, col2 = st.columns([20,1,20])
                style_ = """<h6 style='text-align: center;'>{}</h6>"""
                with col1:
                    st.image(input_img, use_column_width=True)
                    st.markdown(style_.format("Ogirinal Image"),unsafe_allow_html=True)
                with col2:
                    st.image(img_output, use_column_width=True)
                    st.markdown(style_.format("Prediction Image"),unsafe_allow_html=True)
                
                st.json(json_str)
            except:
                st.info('Image URL is not valid!', icon="ℹ️")            

    else:
        # Uploading the image to the page
        uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])

        if uploadFile is not None:
            input_img = read_img(uploadFile)
            st.write("Image Uploaded Successfully")

            with st.spinner('Detecting...'):
                output_file, img_output, json_str = dectection(input_img, uploadFile.name)

            # output_img = load_image(output_file)
            print("Ouput image path:", output_file)

            results_str = '<h3 style="text-align: left;">Result</h3>'
            st.markdown(results_str, unsafe_allow_html=True)

            col1, mid, col2 = st.columns([20,1,20])
            style_ = """<h6 style='text-align: center;'>{}</h6>"""
            with col1:
                st.image(input_img, use_column_width=True)
                st.markdown(style_.format("Ogirinal Image"),unsafe_allow_html=True)
            with col2:
                st.image(img_output, use_column_width=True)
                st.markdown(style_.format("Prediction Image"),unsafe_allow_html=True)
            
            st.json(json_str)
        else:
            st.write("Make sure you image is in JPG/PNG Format.")

if __name__ == '__main__':
    main()