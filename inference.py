from src.tools.predictor import Predictor
from src.tools.config import Cfg
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import json
def main():

    config = Cfg.load_config_from_file('./config/vgg_transformer.yml')
    config['vocab'] = config['vocab'] + ' ' + '̉'+ '̀' + '̃'+ '́'+ '̣'
    config['weights'] = './weights/checkpoint_oov_fullvalid.pth'
    config['device'] = 'cuda:1'
    detector = Predictor(config)

    root = '/mlcv/Databases/OOV/WordCropping/test'

    imgs_id = os.listdir(root)

    results = []
    img_faild = []
    for img_id in tqdm(imgs_id):
        try:
            img = Image.open(os.path.join(root, img_id))
            res = detector.predict(img)
        except:
            res = 'null'
            img_faild.append(img_id)

        results.append({   
                        "text_id": img,
                        "transcription": res
                    })
        
    with open("./submission/submit_ver1.json", "w") as outfile:
        outfile.write(json.dumps(results, indent = 4))

    print(img_faild)
    # batch_size = 256
    # imgs_id_split = np.array_split(imgs_id, int(len(imgs_id)/batch_size)+1)
    # imgs_id_batch = [list(img_id) for img_id in imgs_id_split]


    # results = []
    # for imgs_id in tqdm(imgs_id_batch):

    #     imgs = [Image.open(os.path.join(root, fn)) for fn in imgs_id]
    #     res = detector.predict_batch(imgs)

    #     for i in range(len(imgs_id)):
    #         results.append({   
    #                         "text_id": imgs_id[i].split('.')[0], 
    #                         "transcription": res[i]
    #                        })
    
    with open("./submission/submit_ver1.json", "w") as outfile:
        outfile.write(json.dumps(results, indent = 4))

if __name__ == '__main__':
    main()

