import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import CasUNeXt


def create_fine_img(prediction, original_img, original_mask, img, size=224):
    # 获取前景位置，得到粗分割图片的尺寸大小
    img_height, img_width = img.shape[-2:]
    min_row, min_column, max_row, max_column = 0, 0, 0, 0
    for row in range(0, img_height):
        for col in range(0, img_width):
            if prediction[row][col] == 1:
                if min_row == 0:
                    min_row = row
                if min_column == 0 or col < min_column:
                    min_column = col
                if row > max_row:
                    max_row = row
                if col > max_column:
                    max_column = col

    # 生成粗分割结果图及其对应mask，图片尺寸默认为224*224
    coarse_height = size
    coarse_width = size
    coarse_img = torch.zeros((coarse_height, coarse_width))
    coarse_mask = torch.zeros((coarse_height, coarse_width))

    # 根据前景部分的高度和宽度计算扩充尺寸,默认上下左右均等扩充
    seg_width = max_column - min_column
    seg_height = max_row - min_row
    left_broad = int((coarse_width - seg_width) / 2) if coarse_width - seg_width >= 0 else 0
    top_broad = int((coarse_height - seg_height) / 2) if coarse_height - seg_height >= 0 else 0

    if min_row - top_broad < 0:
        top_broad = min_row
    if min_column - left_broad < 0:
        left_broad = min_column

    for row in range(0, coarse_height):
        for col in range(0, coarse_width):
            coarse_img[row][col] = np.array(original_img)[min_row - top_broad + row][min_column - left_broad + col]
            coarse_mask[row][col] = np.array(original_mask)[min_row - top_broad + row][min_column - left_broad + col]

    image = Image.fromarray(np.array(coarse_img).astype(np.uint8))
    image.save("./result/coarse_seg_img.png")
    mask = Image.fromarray(np.array(coarse_mask).astype(np.uint8))
    mask.save("./result/fine_ground_truth.png")


def segment(img_path, mask_path, weights_path, mean, std, coarse: bool, classes=1):
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = CasUNeXt(in_channels=1, num_classes=classes+1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load image
    original_img = Image.open(img_path).convert('L')

    #load mask
    original_mask = Image.open(mask_path).convert('L')

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 1, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        if coarse:
            create_fine_img(prediction, original_img, original_mask, img)
            # 将前景对应的像素值改成255(白色)
            prediction[prediction == 1] = 255
            mask = Image.fromarray(prediction)
            mask.save("./result/coarse_seg_result.png")
        else:
            # 将前景对应的像素值改成255(白色)
            prediction[prediction == 1] = 255
            mask = Image.fromarray(prediction)
            mask.save("./result/fine_seg_result.png")


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # coarse_predict
    coarse_weights_path = "./save_weights/best_model_Fetal_coarse.pth"

    img_path = ""
    mask_path = ""

    # load and save image
    original_img = Image.open(img_path).convert('L')
    original_img.save("./result/Original image.png")

    # load and save mask
    mask = Image.open(mask_path).convert('L')
    mask.save("./result/coarse_ground truth.png")

    segment(img_path=img_path, mask_path=mask_path, weights_path=coarse_weights_path, mean=0.157, std=0.182, coarse=True)

    # fine_predict
    fine_weights_path = "./save_weights/best_model_Fetal_fine.pth"
    segment(img_path="./result/coarse_seg_img.png", mask_path=mask_path, weights_path=fine_weights_path, mean=0.329, std=0.161, coarse=False)


if __name__ == '__main__':
    main()