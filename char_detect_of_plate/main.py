from PIL import Image
import torch
from numpy import random, asarray

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, strip_optimizer, set_logging
from utils.torch_utils import select_device, load_classifier, time_synchronized
import cv2 as cv


def detect(img_path, conf_th):
    char_dict = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
                 'الف': '10', 'ب': '11', 'پ': '12', 'ت': '13', 'ث': '14', 'ج': '15', 'چ': '16', 'ح': '17', 'خ': '18',
                 'د': '19', 'ذ': '20', 'ر': '21', 'ز': '22', 'ژ': '23', 'س': '24', 'ش': '25', 'ص': '26', 'ض': '27',
                 'ط': '28', 'ظ': '29', 'ع': '30', 'غ': '31', 'ف': '32', 'ق': '33', 'ک': '34', 'گ': '35', 'ل': '36',
                 'م': '37', 'ن': '38', 'ه‍': '39', 'و': '40', 'ی': '41', 'ژ (معلولین و جانبازان)': '42'}

    char_id_dict = {v: k for k, v in char_dict.items()}

    weights = 'best.pt'
    imgsz = 640

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    dataset = LoadImages(img_path, img_size=imgsz, stride=stride)

    # im0 = asarray(im_crop)
    for path, img, im0s, vid_cap in dataset:
        height, width = im0s.shape[0], im0s.shape[1]
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        plate_img = cv.imread(img_path)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            sorted_det = sorted(det.tolist(), key=lambda x: (x[0]))
            plate_char = ''
            for k in sorted_det:
                conf = k[4]
                if conf > conf_th:
                    plate_char += char_id_dict[str(int(k[5]))]
                    x1, y1, x2, y2 = (int(k[0]), int(k[1]), int(k[2]), int(k[3]))

                    label = char_id_dict[str(int(k[5]))]
                    cv.rectangle(plate_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    labelSize = cv.getTextSize(label, cv.FONT_ITALIC, 0.5, 2)

                    _x1 = x1
                    _y1 = y1
                    _x2 = _x1 + labelSize[0][0]
                    _y2 = y1 - int(labelSize[0][1])
                    cv.rectangle(plate_img, (_x1, _y1), (_x2, _y2), (0, 255, 0), cv.FILLED)
                    cv.putText(plate_img, label, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

            return plate_char, sorted_det, plate_img


input_path='img_test/t (9).jpg'
plate_string, char_detected, out_img = detect(input_path, 0.6)
print("\nlicense_plate_number: {}".format(plate_string))
print(char_detected)

cv.imwrite('out/out-{}.png'.format(input_path.split('.')[0].split('/')[-1]), out_img)
