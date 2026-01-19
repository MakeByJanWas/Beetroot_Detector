import cv2
import numpy as np
import os
from glob import glob

output_img_folder = "dataset/images/train"
output_lbl_folder = "dataset/labels/train"

os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_lbl_folder, exist_ok=True)

lower_green = np.array([27, 30, 120])
upper_green = np.array([90, 255, 255])

categories = {
    "chwast": 0,
    "burak": 1
}

for category, class_id in categories.items():
    image_files = glob(os.path.join(".", category, "*"))
    for image_path in image_files:
        filename = os.path.splitext(os.path.basename(image_path))[0]

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        labels = []
        for cnt in contours:
           x, y, bw, bh = cv2.boundingRect(cnt)
           x_c = (x + bw / 2) / w
           y_c = (y + bh / 2) / h
           bw_n = bw / w
           bh_n = bh / h
           labels.append(f"{class_id} {x_c:.6f} {y_c:.6f} {bw_n:.6f} {bh_n:.6f}")

        if labels:
            label_path = os.path.join(output_lbl_folder, f"{filename}.txt")
            with open(label_path, "w") as f:
                f.write("\n".join(labels))

            img_out_path = os.path.join(output_img_folder, f"{filename}.jpg")
            cv2.imwrite(img_out_path, img)
            print(f"Zapisano: {filename} (klasa {class_id})")
        else:
            print(f"Brak konturów: {filename}")
