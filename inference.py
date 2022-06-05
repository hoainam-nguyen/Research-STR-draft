from src.tools.predictor import Predictor
from src.tools.config import Cfg
import cv2
import os
from PIL import Image
import argparse
from tqdm import tqdm

def main():

    config = Cfg.load_config_from_file('./config/vgg_transformer.yml')
    config['vocab'] = config['vocab'] + ' ' + '̉'+ '̀' + '̃'+ '́'+ '̣'
    config['weights'] = './weights/checkpoints.pth'
    detector = Predictor(config)

    path = './data/VinText/test'
    file_label = os.path.join(path, 'labels.txt')
    with open(file_label, 'r', encoding='utf8') as f:
        cases = f.readlines()
    
    cnt = 0
    total = 0
    for ins in tqdm(cases):
        im, labels = ins.split('\t')
        file_img = os.path.join(path, im)
        img = Image.open(file_img)
        label = labels[:-1]
        if label == '###':
            continue
        s, prob= detector.predict(img, return_prob=True)
        if s == label:
            cnt += 1
        total += 1

    print(cnt/total)
if __name__ == '__main__':
    main()

