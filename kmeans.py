import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def kmeans(image, centers, iterations):
    centroidsReal = []
    for c in centers:
        centroidsReal = np.array([image[c[1]][c[0]] for c in centers])

    # Reshape the image into a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Initialize the centers
    c = centroidsReal.astype(np.float32)

    for i in range(iterations):
        # Calculate distances between pixels and each center
        distances = np.sum(np.abs(pixels - c[:, np.newaxis]), axis=2)
        
        # Set each pixel to the nearest center
        labels = np.argmin(distances, axis=0)
        
        # Update centers based on assigned pixels
        for j in range(len(c)):
            c[j] = np.mean(pixels[labels == j], axis=0)

    # Reshape the image back to its original shape
    labels = labels.reshape(image.shape[:2])

    # Convert the image back to values ranging from 0 to 255 (uint8)
    colors = np.round(c).astype(np.uint8)
    segmented_image = colors[labels]

    return segmented_image


def calculate_centers(image, mode, dimension, num_centers):
    if mode == "manual":
        cv.namedWindow("Select {} centers using the left mouse button. Press 'c' to continue. Press 'z' to undo. Press 'q' to close.".format(3))
        cv.imshow("Select {} centers using the left mouse button. Press 'c' to continue. Press 'z' to undo. Press 'q' to close.".format(3), image)
        centers = []

        def mouse_callback(event, x, y, flags, params):
            if event == cv.EVENT_LBUTTONUP:
                centers.append((x, y))
                image_copy = image.copy()
                for center in centers:
                    cv.circle(image_copy, center, 5, (0, 0, 255), -1)
                cv.imshow("Select {} centers using the left mouse button. Press 'c' to continue. Press 'z' to undo. Press 'q' to close.".format(3), image_copy)

        cv.setMouseCallback("Select {} centers using the left mouse button. Press 'c' to continue. Press 'z' to undo. Press 'q' to close.".format(3), mouse_callback)

        while True:
            key = cv.waitKey(0)
            if key == ord('c'):
                break
            elif key == ord('q'):
                cv.destroyAllWindows()
                return None
            elif key == ord('z'):
                if len(centers) > 0:
                    centers.pop()
                    image_copy = image.copy()
                    for center in centers:
                        cv.circle(image_copy, center, 5, (0, 0, 255), -1)
                    cv.imshow("Select {} centers using the left mouse button. Press 'c' to continue. Press 'z' to undo. Press 'q' to close.".format(3), image_copy)

        centers = np.array(centers)

        return centers

    elif mode == "random":
        if dimension == 3:
            # Random mode - centers are randomly selected
            color_diff = 150  # minimum color difference between pixels
            h, w, c = image.shape
            # Choose the first center randomly
            centers = [[np.random.randint(0, w), np.random.randint(0, h)]]

            # Choose the next centers until the desired number is reached
            while len(centers) < num_centers:
                # Randomly choose a point
                x, y = np.random.randint(0, w), np.random.randint(0, h)
                # Check if it is far enough from all existing centers and if the pixel color is sufficiently different from the others
                if all(np.sum(np.abs(image[y, x] - image[cy, cx])) > color_diff for cx, cy in centers):
                    centers.append([x, y])

            return centers
        
        elif dimension == 5:
            # Random mode - centers are randomly selected
            T = 300  # threshold T
            color_diff = 150  # minimum color difference between pixels
            h, w, c = image.shape
            # Choose the first center randomly
            centers = [[np.random.randint(0, w), np.random.randint(0, h)]]

            # Choose the next centers until the desired number is reached
            while len(centers) < num_centers:
                # Randomly choose a point
                x, y = np.random.randint(0, w), np.random.randint(0, h)
                # Check if it is far enough from all existing centers and if the pixel color is sufficiently different from the others
                if all((abs(x - cx) + abs(y - cy)) > T and np.sum(np.abs(image[y, x] - image[cy, cx])) > color_diff for cx, cy in centers):
                    centers.append([x, y])

            return centers

# Load the image
img = cv.imread('peppers.jpg')

slika1 = kmeans(img, calculate_centers(img, "manual", 3, 3), 2)
slika2 = kmeans(img, calculate_centers(img, "random", 3, 3), 6)
slika3 = kmeans(img, calculate_centers(img, "random", 3, 3), 10)
slika4 = kmeans(img, calculate_centers(img, "random", 5, 3), 2)
slika5 = kmeans(img, calculate_centers(img, "random", 5, 3), 6)
slika6 = kmeans(img, calculate_centers(img, "random", 5, 3), 10)
slika7 = kmeans(img, calculate_centers(img, "random", 5, 5), 5)

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
slika1 = cv.cvtColor(slika1, cv.COLOR_BGR2RGB)
slika2 = cv.cvtColor(slika2, cv.COLOR_BGR2RGB)
slika3 = cv.cvtColor(slika3, cv.COLOR_BGR2RGB)
slika4 = cv.cvtColor(slika4, cv.COLOR_BGR2RGB)
slika5 = cv.cvtColor(slika5, cv.COLOR_BGR2RGB)
slika6 = cv.cvtColor(slika6, cv.COLOR_BGR2RGB)
slika7 = cv.cvtColor(slika7, cv.COLOR_BGR2RGB)

plt.subplot(2, 4, 1), plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 2), plt.imshow(slika1)
plt.title('K-Means 1'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 3), plt.imshow(slika2)
plt.title('K-Means 2'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 4), plt.imshow(slika3)
plt.title('K-Means 3'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 5), plt.imshow(slika4)
plt.title('K-Means 4'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 6), plt.imshow(slika5)
plt.title('K-Means 5'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 7), plt.imshow(slika6)
plt.title('K-Means 6'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 8), plt.imshow(slika7)
plt.title('K-Means 7'), plt.xticks([]), plt.yticks([])

plt.show()
