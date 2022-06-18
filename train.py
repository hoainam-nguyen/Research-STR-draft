from src.tools.config import Cfg
from src.runner.trainer import Trainer
import sys


def main():
    config = Cfg.load_config_from_file('./config/vgg_transformer.yml')

    if len(sys.argv) > 2:
        config['device'] = f'cuda:{sys.argv[1]}'

    config['vocab'] = config['vocab'] + ' ' + '̉'+ '̀' + '̃'+ '́'+ '̣'
    params = {
        'log': './log/train.log',
        'print_every':20,
        'valid_every':500,
        'iters':10000,
        'batch_size': 128,
        'checkpoint': None,    
        'export':'./weights/vintext.pth',
        'metrics': 1000,
    }

    config['trainer'].update(params)  
    trainer = Trainer(config, pretrained=False)
    print('Start training')
    trainer.train()
    print(trainer.precision())
 
if __name__ == '__main__':
    main()

