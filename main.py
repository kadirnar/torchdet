import torch
import torchvision
from PIL import Image

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.load_state_dict(torch.load('models/maskrcnn_resnet50_fpn_coco.pth'))
model.eval()

img = "images/1.jpg"
test_image = Image.open(img).convert('RGB')
test_image = torchvision.transforms.functional.to_tensor(test_image).unsqueeze(0)
predictions = model(test_image)
