from matplotlib import pyplot as plt

import numpy as np

import cv2
import os
import csv

date = '22-9-2023'
dataset_directory = f'Dataset/Image/{date}'
header = ['No.', 'Roundness']
data = []

for file in os.listdir(dataset_directory):
    if not file.endswith("JPG") or file == ".DS_Store":
        continue

    i = int(file.split('.')[0].split('JPG')[0])

    image = cv2.imread(dataset_directory + "/" + file)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    _, threshold_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY_INV)

    # plt.figure(figsize=(15, 15))
    # plt.imshow(threshold_image, cmap="gray")
    # plt.show()

    kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel)

    contour, hierarchy = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_by_max_area = sorted(contour, key=cv2.contourArea, reverse=True)[0]

    max_area = cv2.contourArea(contour_by_max_area)

    result_image = image.copy()

    if contour_by_max_area is not None:
        cv2.drawContours(result_image, contour_by_max_area, -1, (0, 255, 0), 15)

        (x, y), radius = cv2.minEnclosingCircle(contour_by_max_area)
        radius = int(radius)
        center = (int(x), int(y))
        circle_area = 3.141592653589 * radius * radius
        cv2.circle(result_image, center, radius, (0, 0, 255), 15)
        cv2.putText(result_image, f"Circle Area: {circle_area:.2f}px", (center[0], center[1] - radius),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)

        M = cv2.moments(contour_by_max_area)
        if M["m00"] != 0:
            X = int(M["m10"] / M["m00"])
            Y = int(M["m01"] / M["m00"])
        else:
            X, Y = 0, 0

        roundness = round(max_area / circle_area, 3)

        cv2.putText(result_image, f"Area: {max_area:.2f}px", (X - 100, Y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)
        cv2.putText(result_image, f"Roundness: {roundness}", (200, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)
        cv2.putText(result_image, f"File name: {file}", (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)

    cv2.imwrite(f"roundness_{i}.jpg", result_image)

    data.append([i, roundness])

    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # plt.figure(figsize=(15, 15))
    # plt.imshow(result_image)
    # plt.show()

data.sort()
print(data)

with open(f'roundness_{date}.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(header)
    writer.writerows(data)
