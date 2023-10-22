from matplotlib import pyplot as plt

import numpy as np

import cv2


class Roundness:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = None
        self.image = cv2.imread(file_path)
        self.roundness = 0
        self.result_image = None

    def analyze(self):
        self.file_name = self.file_path.split("/")[-1]

        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        _, threshold_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        cleaned_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel)

        contour, hierarchy = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_by_max_area = sorted(contour, key=cv2.contourArea, reverse=True)[0]

        max_area = cv2.contourArea(contour_by_max_area)

        self.result_image = self.image.copy()

        if contour_by_max_area is not None:
            cv2.drawContours(self.result_image, contour_by_max_area, -1, (0, 255, 0), 15)

            (x, y), radius = cv2.minEnclosingCircle(contour_by_max_area)
            radius = int(radius)
            center = (int(x), int(y))
            circle_area = 3.141592653589 * radius * radius

            cv2.circle(self.result_image, center, radius, (0, 0, 255), 15)
            cv2.putText(self.result_image, f"Circle Area: {circle_area:.2f}px", (center[0], center[1] - radius),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)

            M = cv2.moments(contour_by_max_area)
            if M["m00"] != 0:
                X = int(M["m10"] / M["m00"])
                Y = int(M["m01"] / M["m00"])
            else:
                X, Y = 0, 0

            self.roundness = round(max_area / circle_area, 3)

            cv2.putText(self.result_image, f"Area: {max_area:.2f}px", (X - 100, Y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)
            cv2.putText(self.result_image, f"Roundness: {self.roundness}", (200, 380),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)
            cv2.putText(self.result_image, f"File: {self.file_name}", (200, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)

        self.result_image = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)

    def get_roundness(self):
        return self.roundness

    def show_result(self):
        plt.figure(figsize=(15, 15))
        plt.imshow(self.result_image)
        plt.show()

    def save_result(self):
        bgr_image = cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"roundness_{self.file_name.split('.')[0]}.jpg", bgr_image)
