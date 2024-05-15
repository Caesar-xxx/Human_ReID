import torch
import torch.nn as nn
import timm
from torchvision import transforms as Transforms
import torch.nn.functional as F
from PIL import Image


# 定义模型
class SwinTransformer(nn.Module):
    def __init__(self, num_features=512):
        super(SwinTransformer, self).__init__()
        self.model = timm.create_model("swin_base_patch4_window7_224")
        self.num_features = num_features
        self.feat = (
            nn.Linear(1024, num_features) if num_features > 0 else None
        )  # 1024是swin_base_patch4_window7_224的输出维度

    def forward(self, x):
        x = self.model.forward_features(x)
        if not self.feat is None:
            x = self.feat(x)
        return x


# 数据预处理
class Data_Processor(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.transformer = Transforms.Compose(
            [
                Transforms.Resize((self.height, self.width)),
                Transforms.ToTensor(),
                Transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, img):
        return self.transformer(img).unsqueeze(0)


data_processor = Data_Processor(height=224, width=224)
model = SwinTransformer(num_features=512).cuda()
model.eval()

# 加载权重
weight_path = "./weights/swin_base_patch4_window7_224.pth"
weight = torch.load(weight_path)
model.load_state_dict(weight["state_dict"], strict=True)


def getImgFeat(img_file):
    with torch.no_grad():
        # PIL read image
        img = Image.open(img_file).convert("RGB")  # 读取图片，转换为RGB
        img = data_processor(img).cuda()  # 数据预处理
        feat = F.normalize(model(img), dim=1).cpu()  # 使用F.normalize对特征进行L2归一化
        return feat
