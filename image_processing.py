"""
Image Processing Project
Kate Feng
"""

import numpy as np
import os
from PIL import Image

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates RGBImage object from image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        if not isinstance(pixels, list) or (not len(pixels) >= 1):
            raise TypeError()

        if not all([(isinstance(row, list)) and (len(row) >= 1) \
    for row in pixels]):
            raise TypeError()

        row_length = len(pixels[0])
        if not all([len(row) == row_length for row in pixels]):
            raise TypeError()

        if not all([(isinstance(pixel, list)) \
for row in pixels for pixel in row]):
            raise TypeError()

        if not all([len(pixel) == 3 for row in pixels for pixel in row]):
            raise TypeError()

        if not all([(isinstance(value, int)) and \
((value >= 0) and (value <= 255)) for row in pixels \
for pixel in row for value in pixel]):
            raise ValueError()


        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        copy = [[[pixel for pixel in col] for col in row] \
for row in self.pixels]
        return copy


    def copy(self):
        """
        Returns a copy of this RGBImage object

        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        if (not isinstance(row, int)) or (not isinstance(col, int)):
            raise TypeError()

        if (row >= self.num_rows) or (col >= self.num_cols):
            raise ValueError()

        if (row < 0) or (col < 0):
            raise ValueError()

        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if (not isinstance(row, int)) or (not isinstance(col, int)):
            raise TypeError()

        if (row >= self.num_rows) or (col >= self.num_cols):
            raise ValueError()

        if (row < 0) or (col < 0):
            raise ValueError()

        if (not isinstance(new_color, tuple)) or (not len(new_color) == 3) or \
(not all([isinstance(value, int) for value in new_color])):
            raise TypeError()

        if (not all([value <= 255 for value in new_color])):
            raise ValueError()

        for value in range(len(new_color)):
            try:
                if new_color[value] >= 0:
                    self.pixels[row][col][value] = new_color[value]
            except ValueError as error:
                continue



# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True
        """
        copy = [[[255 - pixel for pixel in col] for col in row] \
for row in image.get_pixels()]
        return RGBImage(copy)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        gray = [[[(col[0] + col[1] + col[2])//3 for pixel in col] \
for col in row] for row in image.get_pixels()]
        return RGBImage(gray)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        rotate = [[[pixel for pixel in col] for col in row[::-1]] \
for row in image.get_pixels()[::-1]]
        return RGBImage(rotate)

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        brightness = [(col[0] + col[1] + col[2])//3 for row in \
image.get_pixels() for col in row for pixel in col]
        return sum(brightness)//len(brightness)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 75)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """
        if not isinstance(intensity, int):
            raise TypeError()
        if (intensity > 255) or (intensity < -255):
            raise ValueError()

        intense = [[[pixel + intensity for pixel in col] for col in row] \
for row in image.get_pixels()]

        intense_adjusted = [[[0 if pixel < 0 else 255 if pixel > 255 \
else pixel for pixel in col] for col in row] for row in intense]
        
        return RGBImage(intense_adjusted)


    def blur(self, image):
        """
        Returns a new image with the pixels blurred

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_blur.png')
        >>> img_blur = img_proc.blur(img)
        >>> img_blur.pixels == img_exp.pixels # Check blur
        True
        >>> img_save_helper('img/out/test_image_32x32_blur.png', img_blur)
        """
        pixels = image.get_pixels()
        rows = len(pixels)
        cols = len(pixels[0])
        blurred_pixels = []

        for row in range(rows):
            blurred_row = []
            for col in range(cols):
                total_pixels = 0
                total_r = 0
                total_g = 0
                total_b = 0

                for neighbor_row in range(-1, 2):
                    for neighbor_col in range(-1, 2):
                        row_index = row + neighbor_row
                        col_index = col + neighbor_col

                        if 0 <= row_index < rows and 0 <= col_index < cols:
                            total_pixels += 1
                            r, g, b = pixels[row_index][col_index]
                            total_r += r
                            total_g += g
                            total_b += b

                avg_r = total_r // total_pixels
                avg_g = total_g // total_pixels
                avg_b = total_b // total_pixels

                blurred_row.append([avg_r, avg_g, avg_b])

            blurred_pixels.append(blurred_row)

        return RGBImage(blurred_pixels)


# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0
        self.coupon_count = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        if self.coupon_count == 0:
            self.cost += 5
        else:
            self.coupon_count -= 1

        return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        if self.coupon_count == 0:
            self.cost += 6
        else:
            self.coupon_count -= 1

        return super().grayscale(image)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        if self.coupon_count == 0:
            self.cost += 10
        else:
            self.coupon_count -= 1

        return super().rotate_180(image)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        if self.coupon_count == 0:
            self.cost += 1
        else:
            self.coupon_count -= 1

        return super().adjust_brightness(image, intensity)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred
        """
        if self.coupon_count == 0:
            self.cost += 5
        else:
            self.coupon_count -= 1

        return super().blur(image)

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0

        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(3)
        >>> img = img_proc.rotate_180(img)
        >>> img = img_proc.rotate_180(img)
        >>> img = img_proc.rotate_180(img)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        10
        """
        if not isinstance(amount, int):
            raise TypeError()
        if amount <= 0:
            raise ValueError()
        self.coupon_count = amount



# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        Returns a copy of the chroma image where all pixels with the given
        color are replaced with the background image.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> img_in_back = img_read_helper('img/test_image_32x32.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_32x32_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_32x32_chroma.png', img_chroma)
        """
        if (not isinstance(chroma_image, RGBImage)) or \
(not isinstance(background_image, RGBImage)):
            raise TypeError()
        if chroma_image.size() != background_image.size():
            raise ValueError()

        reference_lst = chroma_image.get_pixels()
        chroma_image = chroma_image.copy()

        for row in range(len(reference_lst)):
            for col in range(len(reference_lst[0])):
                pixel = tuple(reference_lst[row][col])
                if pixel == color:
                    new_color = background_image.get_pixel(row, col)
                    chroma_image.set_pixel(row, col, new_color)
        return chroma_image



    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Returns a copy of the background image where the sticker image is
        placed at the given x and y position.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (31, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/test_image_32x32_sticker.png', img_combined)
        """
        if (not isinstance(sticker_image, RGBImage)) or \
(not isinstance(background_image, RGBImage)):
            raise TypeError()
        if (sticker_image.num_rows > background_image.num_rows) or \
(sticker_image.num_cols > background_image.num_cols):
            raise ValueError()
        if (not isinstance(x_pos, int)) or \
(not isinstance(y_pos, int)):
            raise TypeError()
        if x_pos < 0 or y_pos < 0:
            raise ValueError()
        if (x_pos + sticker_image.num_cols) > background_image.num_cols:
            raise ValueError()
        if (y_pos + sticker_image.num_rows) > background_image.num_rows:
            raise ValueError()

        reference_lst = sticker_image.get_pixels()
        sticker_added = background_image.copy()

        for row in range(len(reference_lst)):
            for col in range(len(reference_lst[0])):
                new_pixel = sticker_image.get_pixel(row, col)
                sticker_added.set_pixel(y_pos + row, x_pos + col, new_pixel)
        return sticker_added


    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        pixels = [[(col[0] + col[1] + col[2])//3 for col in row] for row in \
image.get_pixels()]

        rows = len(pixels)
        cols = len(pixels[0])
        highlighted = []

        kernel = [[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]]

        for row in range(rows):
            highlighted_row = []
            for col in range(cols):
                new_pixel = 0
                center_row = row
                center_col = col
                
                if (center_row < 1) or (center_row > 1):
                    center_row = 1
                if (center_col < 1) or (center_col > 1):
                    center_col = 1

                for neighbor_row in range(-1, 2):
                    for neighbor_col in range(-1, 2):
                        row_index = row + neighbor_row
                        col_index = col + neighbor_col
                        
                        if 0 <= row_index < rows and 0 <= col_index < cols:
                            og_val = pixels[row_index][col_index]
                            kernel_val = kernel[center_row + neighbor_row]\
[center_col + neighbor_col]
                            new_val = og_val * kernel_val
                            new_pixel += new_val
                            
                highlighted_row.append(new_pixel)

            highlighted.append(highlighted_row)

        for row in range(rows):
            for col in range(cols):
                if highlighted[row][col] > 255:
                    highlighted[row][col] = 255
                elif highlighted[row][col] < 0:
                    highlighted[row][col] = 0

        for row in range(rows):
            for col in range(cols):
                highlighted[row][col] = \
[highlighted[row][col], highlighted[row][col], highlighted[row][col]]

        return RGBImage(highlighted)


# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """

        self.k_neighbors = k_neighbors
        self.data = []

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """

        if len(data) < self.k_neighbors:
            raise ValueError()

        self.data = data

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """

        if (not isinstance(image1, RGBImage)):
            raise TypeError()

        if (not isinstance(image2, RGBImage)):
            raise TypeError()

        if image1.size() != image2.size():
            raise ValueError()

        img1 = [pixel for row in image1.get_pixels() for pixel in row]
        img2 = [pixel for row in image2.get_pixels() for pixel in row]
        both_img = zip(img1, img2)

        differences = [(img1[i] - img2[i])**2 for (img1, img2) in both_img \
for i in range(3)]
        distance = sum(differences) ** (1/2)

        return distance

    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        """
        
        label_dict = {label: 0 for label in candidates}
        for label in candidates:
            label_dict[label] += 1

        return max(label_dict)


    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        if len(self.data) == 0:
            raise ValueError()

        distances = [(self.distance(image, image2[0]), image2) for \
image2 in self.data]
        distances = sorted(distances)[:self.k_neighbors]
        labels = [compare[1][1] for compare in distances]
        return self.vote(labels)


def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label
