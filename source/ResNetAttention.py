import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from source.blocks import EncoderBlock, PositionalEmbedding
    from source.resnet18 import ResNet, BasicBlock
except:
    from blocks import EncoderBlock, PositionalEmbedding
    from resnet18 import ResNet, BasicBlock
    

class ResNetAttention(nn.Module):
    """
    BERT-based [Devlin et al. NAACL 2019] language model to predict the next word given a context.

    This is just a stack of encoder blocks followed by a pooling layer for classification.

    Notes
    -----
    Instead of a <CLS> token, we use a pooling by multi-head attention (PMA) block for final layer.
    """
    def __init__(self, device, num_class, num_layers, dim, hidden_dim,
                 num_heads=8, dropout_prob=0.1, max_length=30, key_point=48):
        super().__init__()
        self.device = device
        self.dim = dim
        #self.embedding = nn.Embedding(max_length, key_point, padding_idx=1)
        self.positional_encoding = PositionalEmbedding(dim=key_point, max_length=max_length)
        #self.cls_token = nn.Parameter(torch.zeros((1, dim)))
        
        self.layers = nn.ModuleList()
        self.feature_extract = ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_class, nseq = max_length, out_dim=94)
        self.positional_encoding_feature = PositionalEmbedding(dim=94, max_length=max_length)
        
        self.fc = nn.Linear(dim, num_class)
        
        for _ in range(num_layers):
            self.layers.append(EncoderBlock(dim, hidden_dim, num_heads, dropout_prob))
        self.initialize_weights()
        self.set_device()

    def initialize_weights(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        #nn.init.xavier_normal_(self.cls_token)

    def set_device(self):
        for m in self.modules():
            m = m.to(self.device)

    def forward(self, x):
        x_box = x[0]
        x_key = x[1]
        #bsz = x.shape[0]
        #x_key = self.embedding(x_key)
        x_key = self.positional_encoding(x_key)
        
        x_box = self.feature_extract(x_box)
        x_box = self.positional_encoding_feature(x_box)
        
        x = torch.cat((x_key, x_box), dim=2)
        for layer in self.layers:
            x = layer(x)
        x = x[:, -1]
        x = self.fc(x)
        return x
    
class ResNetAttentionVisual(nn.Module):
    """
    BERT-based [Devlin et al. NAACL 2019] language model to predict the next word given a context.

    This is just a stack of encoder blocks followed by a pooling layer for classification.

    Notes
    -----
    Instead of a <CLS> token, we use a pooling by multi-head attention (PMA) block for final layer.
    """
    def __init__(self, device, num_class, num_layers, dim, hidden_dim,
                 num_heads=8, dropout_prob=0.1, max_length=30):
        super().__init__()
        self.device = device
        self.dim = dim
        self.layers = nn.ModuleList()
        self.feature_extract = ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_class, nseq = max_length, out_dim = self.dim)
        self.positional_encoding_feature = PositionalEmbedding(self.dim, max_length)
        
        for _ in range(num_layers):
            self.layers.append(EncoderBlock(dim, hidden_dim, num_heads, dropout_prob))
            
        self.fc = nn.Linear(dim, num_class, bias=True)
        
        self.initialize_weights()
        self.set_device()

    def initialize_weights(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        #nn.init.xavier_normal_(self.cls_token)

    def set_device(self):
        for m in self.modules():
            m = m.to(self.device)

    def forward(self, x):
        x_box = x
        #x_key = x[1]
        #bsz = x.shape[0]
        #x_key = self.embedding(x_key)
        #x_key = self.positional_encoding(x_key)
        
        x_box = self.feature_extract(x_box)
        x_box = self.positional_encoding_feature(x_box)
        
        #x = torch.cat((x_key, x_box), dim=2)
        x = x_box
        for layer in self.layers:
            x = layer(x)
        x = x[:, -1]
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    # model = ResNetAttention(device = 'cuda:0', num_class = 6, num_layers = 2, dim = 128, hidden_dim = 128, num_heads=8, dropout_prob=0.1, max_length=10)
    # x_box = torch.randn(20, 10, 3, 224, 224).to('cuda:0')
    # x_key = torch.randn(20, 10, 48).to('cuda:0')
    # y = model([x_box, x_key])
    # print('\n output shape:', y.shape)
    
    
    model = ResNetAttentionVisual(device = 'cuda:0', num_class = 6, num_layers = 2, dim = 128, hidden_dim = 128, num_heads=8, dropout_prob=0.1, max_length=10)
    x_box = torch.randn(20, 10, 3, 224, 224).to('cuda:0')
    y = model(x_box)
    print('\n output shape:', y.shape)