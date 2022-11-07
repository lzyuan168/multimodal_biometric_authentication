import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, inchannel):
        super().__init__()
        # if fingerprint, in_chans=1,  if resnet18, then need num_classes=65
        self.model = timm.create_model("vgg16", pretrained=True, num_classes=65, in_chans=inchannel)
        self.pre_trained = self.model.features[:24]
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm1d(4096)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(25088,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,65)


        # # Freeze all layers
        # for params in self.model.parameters():
        #     params.requires_grad = False

        # # Unfreeze selected layers for training
        # for params in self.model.features[24].parameters():
        #     params.requires_grad = True

        # for params in self.model.features[26].parameters():
        #     params.requires_grad = True

        # for params in self.model.features[28].parameters():
        #     params.requires_grad = True

        # for params in self.model.pre_logits.fc1.parameters():
        #     params.requires_grad = True

        # for params in self.model.pre_logits.fc2.parameters():
        #     params.requires_grad = True

        # for params in self.model.head.fc.parameters():
        #     params.requires_grad = True        

        # for name, param in self.model.named_parameters():
        #     print(name, param.requires_grad)

    
    def forward(self, x):
        out = self.pre_trained(x)
        out = F.relu(self.conv1(out))
        out = self.bn(out)
        out = F.relu(self.conv2(out))
        out = self.bn(out)
        out = F.relu(self.conv3(out))
        out = self.pool(self.bn(out))
        out = out.reshape(out.shape[0],-1)
        out = F.relu(self.fc1(out))
        #out = F.relu(out)
        out = F.relu(self.fc2(out))
        # out = F.relu(out)
        out = F.dropout(out, 0.3)
        out = self.fc3(out)
        #print(out.shape)
        return out


# test = VGG16()
# ins = torch.rand(1, 3, 224, 224)
# output = test(ins)


class SiameseVGG16(nn.Module):
    def __init__(self, fg_inchannel, ecg_inchannel):
        super().__init__()
        # if fingerprint, in_chans=1,  if resnet18, then need num_classes=65
        self.fg_model = timm.create_model("vgg16", pretrained=True, num_classes=65, in_chans=fg_inchannel)
        self.ecg_model = timm.create_model("vgg16", pretrained=True, num_classes=65, in_chans=ecg_inchannel)
        self.fg_pre_trained = self.fg_model.features[:24]
        self.ecg_pre_trained = self.ecg_model.features[:24]
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm1d(4096)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(25088,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,65)
    
    def forward(self, fg, ecg):
        fg_out = self.fg_pre_trained(fg)
        ecg_out = self.ecg_pre_trained(ecg)
        fg_out = F.relu(self.conv1(fg_out))
        ecg_out = F.relu(self.conv1(ecg_out))
        fg_out = self.bn(fg_out)
        ecg_out = self.bn(ecg_out)
        fg_out = F.relu(self.conv2(fg_out))
        ecg_out = F.relu(self.conv2(ecg_out))
        fg_out = self.bn(fg_out)
        ecg_out = self.bn(ecg_out)
        fg_out = F.relu(self.conv3(fg_out))
        ecg_out = F.relu(self.conv3(ecg_out))
        fg_out = self.pool(self.bn(fg_out))
        ecg_out = self.pool(self.bn(ecg_out))
        fg_out = fg_out.reshape(fg_out.shape[0],-1)
        ecg_out = ecg_out.reshape(ecg_out.shape[0],-1)
        fg_out = F.relu(self.fc1(fg_out))
        ecg_out = F.relu(self.fc1(ecg_out))
        combine_out = fg_out + ecg_out
        out = F.relu(self.fc2(combine_out))
        out = F.dropout(out, 0.3)
        out = self.fc3(out)
        #print(out.shape)
        return out


