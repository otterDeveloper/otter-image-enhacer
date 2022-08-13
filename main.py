#!/usr/bin/env python3
from asyncio.log import logger
from pathlib import Path
from cmath import e, log
import datetime
from os.path import isfile, join, isdir
from os import listdir
import os
import string
import sys
import mimetypes
import csv
import cv2
import logging
from tqdm import tqdm
import urllib.request
# max numbers of pixels of the image to avoid out of memory exception
MAX_MATRIX_SIZE = 3500000


def make_path(path):
    try:
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    except e:
        logging.error(f"Error creating folder: {e}")


def file_list(path: string, max_level=4):
    files = []
    for file in listdir(path):
        listing = join(path, file)
        if isfile(listing):
            files.append(listing)
        if isdir(listing) and max_level >= 0:
            files += file_list(listing, max_level - 1)

    return files


def get_file_mime_type(file):
    mime = mimetypes.guess_type(file)[0]
    return mime if mime else 'application/octet-stream'


def get_files_data(file_list, original_path):
    files_data = []
    for file in tqdm(file_list, desc="Getting files data", unit="files"):
        mime = get_file_mime_type(file)
        w, h, c = get_image_size(file)
        # only append if it's an image
        if mime.startswith('image') and w and h and c:
            files_data.append({
                "path": file,
                "filename": file.replace(original_path, ''),
                "type": mime,
                "size": sys.getsizeof(file),
                "width": w,
                "height": h
            })
    return files_data


def get_image_size(file):
    img = cv2.imread(file)
    try:
        return img.shape
    except:
        return [None, None, None]


def export_files_data(files_data, output_path):
    with open(os.path.abspath(f'{output_path}files.csv'), 'w') as csvfile:
        fieldnames = ['path', 'filename', 'type', 'size', 'width', 'height']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(files_data)


def upscale_image(img: cv2.Mat):
    try:

        w, h, c = img.shape
        if w * h > MAX_MATRIX_SIZE:
            logging.warning(f"Skipping, left as is: {file_data['filename']}")
            return img
        model_path = os.path.abspath("models/EDSR_x4.pb")
        sr = cv2.dnn_superres.DnnSuperResImpl_create()

        sr.readModel(model_path)
        sr.setModel("edsr", 4)
        # sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        result = sr.upsample(img)

        return result
    except Exception as e:
        logging.error(f"Error: {e}")
        return None


def shrink(img: cv2.Mat, factor=.5):
    try:
        resized = cv2.resize(img, dsize=None, fx=factor, fy=factor,
                             interpolation=cv2.INTER_AREA)
        return resized
    except Exception as e:
        logging.error(f"Error: {e}")
        return None


def denoise_image(img: cv2.Mat):
    try:
        result = cv2.fastNlMeansDenoisingColored(src=img, h=5, hColor=5,
                                                 templateWindowSize=7, searchWindowSize=21)
        logging.debug(
            f"Denoise parameters: h={5}, hColor={5}, templateWindowSize={7}, searchWindowSize={21}")
        return result
    except Exception as e:
        logging.error(f"Error: {e}")
        return None


def file_exists(path):
    return os.path.isfile(path)


def download_models():
    try:
        make_path("models/")
        if not file_exists("models/EDSR_x4.pb"):
            logging.info("Downloading models")
            urllib.request.urlretrieve(
                "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb", "models/EDSR_x4.pb")

    except Exception as e:
        logging.critical(f"Could not download models, {e}")
        print(f"Could not download models, {e}")


def opencv_wrapper(file_data, output_path, operation):
    try:
        logging.info(f"Processing: {file_data['filename']} with {operation}")
        start_time = datetime.datetime.now()
        img = cv2.imread(file_data['path'])  # original image
        result = None  # result image
        if mode == "shrink":
            if result := shrink(img) is not None:
                result = upscale_image(result)
        elif mode == "upscale":
            result = upscale_image(img)
        elif mode == "denoise":
            result = denoise_image(img)
        end_time = datetime.datetime.now()
        logging.info(
            f"Processed: {file_data['filename']}, time: {end_time - start_time}")
        new_path = f"{output_path}{file_data['filename']}"
        make_path(new_path)
        if result is not None:
            cv2.imwrite(new_path, result)
        else:
            logging.warning(
                f"Received Empty result for: {file_data['filename']}")
    except Exception as e:
        logging.error(f"Error processing: {file_data['filename']}, {e}")


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    path = os.path.abspath(sys.argv[1])
    modes = ["shrink", "upscale", "denoise"]
    mode = ""
    if sys.argv[2] in modes:
        mode = sys.argv[2]

    # create output directory with timestamp
    output_path = f"output/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}_{mode}/"
    os.makedirs(output_path)

    # set logging
    # format with timestamp and log level
    FORMAT = '%(asctime)s %(levelname)s %(message)s'
    # output to file and console
    logging.basicConfig(level=logging.DEBUG, format=FORMAT, handlers=[
                        logging.FileHandler(f'{output_path}/output.log')])
    logging.info(f"Started: {start_time}")
    print(f"Started: {start_time}")

    file_list = file_list(path)
    logging.info(f"Found {len(file_list)} files")
    print(f"Found {len(file_list)} files")
    files_data = get_files_data(file_list, path)

    download_models()

    # export file_data to csv
    # completely unnecesary but why not
    export_files_data(files_data, output_path)

    # apply correct mode
    for file_data in (progress_bar := tqdm(files_data, desc="Processing images", unit="files")):
        progress_bar.set_description(
            f"Processing image: {file_data['filename']}")
        opencv_wrapper(file_data, output_path, mode)

    end_time = datetime.datetime.now()

    logging.info(f"End time: {end_time}")
    logging.info(f"Total time: {end_time - start_time}")
    logging.info(f"Total files: {len(files_data)}")
    logging.info(
        f"Average time per file: {(end_time - start_time) / len(files_data)}")

    print(f"Done, total time: {end_time - start_time}")
    print(f"Total files: {len(files_data)}")
    print(
        f"Average time per file: {(end_time - start_time) / len(files_data)}")
    print(f"Output path: {output_path}")
