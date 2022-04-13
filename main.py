import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
import numpy as np


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# model.load_state_dict(torch.load('models/maskrcnn_resnet50_fpn_coco.pth'))
model.eval()

img = "1.jpg"

# tensor img
tensor_img = torch.Tensor(cv2.imread(img))
tensor_img = tensor_img.view(1, 3, tensor_img.shape[0], tensor_img.shape[1])


# numpy img
numpy_img = tensor_img.view(tensor_img.shape[2], tensor_img.shape[3], tensor_img.shape[1]).numpy()
numpy_img = np.array(numpy_img, np.int32)

# test_image = Image.open(img).convert('RGB')
# test_image = torchvision.transforms.functional.to_tensor(test_image).unsqueeze(0)
predictions = model(tensor_img)

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


prediction_list = []
for thd in range(0, len(scores)):
    if scores[thd] > threshold:
        prediction_list.append({'score': scores[thd], 'box': boxes[thd], 'label': labels[thd], 'mask': masks[thd]})
  

for i, box in enumerate(prediction_list[0]['box']):
    score = prediction_list[0]["score"][i]
    labels = '%{} {}'.format(float(int(score * 100)), prediction_list[0]["label"][i])
    x, y, w, h = box[0], box[1], box[2], box[3]
