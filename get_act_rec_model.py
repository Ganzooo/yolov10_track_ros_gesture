import torch 
import torch.nn as nn
import math 
import pytorch_model_summary
from source.resnet18LSTM import ResNetLSTM, BasicBlock
from source.ResNetAttention import ResNetAttention, ResNetAttentionVisual
from source.losses import LabelSmoothingLoss

class ActionBasicModule(nn.Module):
    def __init__(self, device="cpu", net=None, classes=7):
        super().__init__()
        self.classes = classes
        self.device = device
        self.model = net
        #self.model.blocks[6].proj = nn.Linear(self.model.blocks[6].proj.in_features, self.classes, bias=True)
        self.model = self.model.to(self.device)
        

    def forward(self, x, label=None, loss_mode="smoothin", smoothing=0.0):
        x = self.model(x)
        if label is not None:
            if loss_mode == "smoothing":
                lossFunc = LabelSmoothingLoss(self.classes, smoothing=smoothing).to(self.device)
            else:
                lossFunc = nn.CrossEntropyLoss().to(self.device)
            label = label.to(self.device)    
            loss = lossFunc(x, label)
            return x, loss
        return x
    
def get_action_rec_model(model_name='ResNetLSTM',action_classes=6, num_action_frame=64, device='cuda', weightPath=''):
    classes = action_classes
    num_frames = num_action_frame
    
    if model_name == 'ResNetLSTM':    
        
        ###ResnetLSTM
        net = ResNetLSTM(BasicBlock, [2, 2, 2, 2], num_classes = classes, lstm_hidden_layer = 512, lstm_sequence_number = num_frames)
        
        x = torch.zeros(1, num_frames,3,224,224)
        print(pytorch_model_summary.summary(net,(x), show_input=True))
        
        model = ActionBasicModule('cuda', net=net, classes = classes)
        state_dict = torch.load('./data/weight/model_weights/ResNetLSTM/modeltype_ResNetLSTM_image_best.pth')
        model.load_state_dict(state_dict)
        model.eval()
    elif model_name == 'ResNetAttention':
        #validDataset = ActionDatasetAttention(validVideo, interval=cfg.sampling_rate, max_len=cfg.num_frames, train=False, transform=data_transformation, mode=mode, img_size = cfg.cropped_box_size)
        #validLoader = DataLoader(validDataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
        
        ###ResnetLSTM
        net = ResNetAttention(device = device, num_class = classes, num_layers = 1, dim = 128, hidden_dim = 128, num_heads=8, dropout_prob=0.1, max_length=num_frames, key_point=34)
        
        model = ActionBasicModule(device, net=net, classes = classes)
        model.load_state_dict(torch.load(weightPath))
        model = model.to(device)
        model.eval()  
    elif model_name == 'ResNetAttentionVisual':
        ###ResnetLSTM
        net = ResNetAttentionVisual(device = device, num_class = classes, num_layers = 1, dim = 128, hidden_dim = 128, num_heads=8, dropout_prob=0.1, max_length=num_frames)
        model = ActionBasicModule(device, net=net, classes = classes)
        
        model.load_state_dict(torch.load(weightPath))
        model = model.to(device)
        model.eval()
        #print(pytorch_model_summary.summary(net,(x), show_input=True))
    return model