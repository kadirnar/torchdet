import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
import numpy as np


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.load_state_dict(torch.load('models/maskrcnn_resnet50_fpn_coco.pth'))
model.eval()

img = "images/1.jpg"

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

predictions_scores, predictions_boxes, predictions_labels, predictions_masks = [], [], [], []

for thd in range(0, len(scores)):
    if scores[thd] > threshold:
        predictions_scores.append(scores[thd])
        predictions_boxes.append(boxes[thd])
        predictions_labels.append(labels[thd])
        predictions_masks.append(masks[thd])

for i, box in enumerate(predictions_boxes):
    score = predictions_scores[i]
    labels = '%{} {}'.format(float(int(score * 100)), predictions_labels[i])
    x, y, w, h = box[0], box[1], box[2], box[3]
    # cv2 sürüm hatası çözülmesi gerekiyor.
    cv2.rectangle(numpy_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(numpy_img, labels, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('img', numpy_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
