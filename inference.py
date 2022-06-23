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

    # results = []
    # img_faild = []
    # for img_id in tqdm(imgs_id[:100]):
    #     try:
    #         img = Image.open(os.path.join(root, img_id))
    #         res = detector.predict(img)
    #     except:
    #         res = 'null'
    #         img_faild.append(img_id)

    #     results.append({   
    #                     "text_id": int(img_id.split('.')[0]),
    #                     "transcription": res
    #                 })
        
    # with open("./submission/submit_ver1.json", "w") as outfile:
    #     outfile.write(json.dumps(results, indent = 4))

    # print(img_faild)

    batch_size = 512
    imgs_id_split = np.array_split(imgs_id, int(len(imgs_id)/batch_size)+1)
    imgs_id_batch = [list(img_id) for img_id in imgs_id_split]

    falid = []
    results = []

    start = time.time()
    try:
        for imgs_id in tqdm(imgs_id_batch):
            imgs = []
            for fn in imgs_id:
                try:   
                    img = Image.open(os.path.join(root, fn))
                    imgs.append(img)
                except:
                    falid.append(fn)
                    continue  

            res = detector.predict_batch(imgs)

            for i in range(len(imgs_id)):
                try:
                    results.append({   
                                    "text_id": imgs_id[i].split('.')[0], 
                                    "transcription": res[i]
                                })
                except:
                    falid.append(imgs_id[i])

    

            with open("./submission/submit_ver2.json", "w") as outfile:
                outfile.write(json.dumps(results, indent = 4))
    except:

        print(f'Done time {time.time() - start}')

        with open("img_faild.txt", 'w', encoding='utf8') as f:
            for img in falid:
                img.write(img + '\n')


if __name__ == '__main__':
    main()

