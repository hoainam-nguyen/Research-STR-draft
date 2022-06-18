from src.model.backbone.cnn import CNN
from src.model.seqmodel.transformer import LanguageTransformer
from torch import nn

class TransSTR(nn.Module):
    def __init__(self, vocab_size,
                 cnn_args, 
                 transformer_args):
        
        super(TransSTR, self).__init__()
        
        self.cnn = CNN(**cnn_args)
        self.transformer = LanguageTransformer(vocab_size, **transformer_args)

    def forward(self, img, tgt_input, tgt_key_padding_mask):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        src = self.cnn(img)
        outputs = self.transformer(src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)
        outputs = outputs.view(-1, outputs.size(2))

        return outputs

