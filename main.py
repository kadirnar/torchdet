import torch
import torchvision
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.load_state_dict(torch.load('models/maskrcnn_resnet50_fpn_coco.pth'))
model.eval()

img = "images/1.jpg"
test_image = Image.open(img).convert('RGB')
test_image = torchvision.transforms.functional.to_tensor(test_image).unsqueeze(0)
predictions = model(test_image)

# boxes
boxes = predictions[0]['boxes'].detach().numpy()
# labels
labels = predictions[0]['labels'].detach().numpy()
# scores
scores = predictions[0]['scores'].detach().numpy()
# masks
masks = predictions[0]['masks'].detach().numpy()
# threshold
threshold = 0.5

predictions_scores = [scores[thd] for thd in range(0, len(scores)) if scores[thd] > threshold]
predictions_boxes = [boxes[thd] for thd in range(0, len(scores)) if scores[thd] > threshold]
predictions_labels = [labels[thd] for thd in range(0, len(scores)) if scores[thd] > threshold]
predictions_masks = [masks[thd] for thd in range(0, len(scores)) if scores[thd] > threshold]

# imshow kodu yazılacak.
