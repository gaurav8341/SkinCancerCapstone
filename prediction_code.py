#!/usr/bin/env python
# coding: utf-8

# In[22]:


import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torch.autograd import Variable


# In[23]:


class ConvNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(64*13*13, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
#         self.classifier =  nn.Sequential(
#             nn.Dropout(0.001),
#             nn.Linear(64 * 13 * 13, 4096),
#             nn.ReLU(),
#             nn.Dropout(0.001),
#             nn.Linear(4096, 512),
#             nn.ReLU())
#         self.final_l = nn.Linear(512, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
#         out = self.layer5(out)
#         print(out.shape)
        out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         out = self.final_l(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# In[24]:


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
input_size = (224, 224)
test_transform = transforms.Compose([transforms.Resize(input_size), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])


# In[44]:


lesion_type_dict = {
    4 : 'Melanocytic nevi',
    6 : 'Melanoma',
    2 : 'Benign keratosis-like lesions ',
    1 : 'Basal cell carcinoma',
    0 : 'Actinic keratoses',
    5 : 'Vascular lesions',
    3 : 'Dermatofibroma'
}


# In[59]:


from PIL import Image

def predict(path, modelpath = "D:\Github\SkinCancerCapstone\models\custommodel.pth"):
    model = torch.load(modelpath)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    img = Image.open(path)
    img = test_transform(img).float()
    img = Variable(img, requires_grad=False)
    img = img.unsqueeze(0).to(device)
    output = model(img)
    print(output)
    prediction = output.max(1, keepdim=True)[1].tolist()
    print(prediction[0][0])
    return lesion_type_dict[prediction[0][0]]


# In[60]:


predict("D:/Github/SkinCancerCapstone/report/img/melanoma.jpg")


# In[ ]:




