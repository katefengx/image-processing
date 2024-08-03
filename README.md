# image-processing
Using a KNN Classifier to process and predict image labels.

## Description
This project implements an image processing toolkit using Python. The main functionalities include reading, saving, and manipulating images. The toolkit provides various image processing methods, such as negating, grayscaling, rotating, adjusting brightness, blurring, and more advanced techniques like chroma key, adding stickers, and edge highlighting.

## Installation
1. Ensure you have Python installed (preferably version 3.6 or above).
2. Install the required libraries using pip

## Usage
### Reading and Saving Images
- img_read_helper(path): Reads an image from the specified path and returns an RGBImage object.
- img_save_helper(path, image): Saves an RGBImage object to the specified path.
### Basic Image Operations
- RGBImage: Represents an image in RGB format.
  - __init__(pixels): Initializes a new RGBImage object.
  - size(): Returns the size of the image in (rows, cols) format.
  - get_pixels(): Returns a copy of the image pixel array.
  - copy(): Returns a copy of the RGBImage object.
  - get_pixel(row, col): Returns the (R, G, B) value at the specified position.
  - set_pixel(row, col, new_color): Sets the (R, G, B) value at the specified position.
### Image Processing Methods
- ImageProcessingTemplate: Contains basic image processing methods.
  - negate(image): Returns a negated copy of the given image.
  - grayscale(image): Returns a grayscale copy of the given image.
  - rotate_180(image): Returns a 180-degree rotated copy of the given image.
  - adjust_brightness(image, intensity): Adjusts the brightness of the given image.
  - blur(image): Returns a blurred copy of the given image.
### Standard Image Processing
- StandardImageProcessing: Extends ImageProcessingTemplate with cost management.
  - Inherits all methods from ImageProcessingTemplate with additional cost tracking.
### Premium Image Processing
- PremiumImageProcessing: Extends ImageProcessingTemplate with advanced features.
  - chroma_key(chroma_image, background_image, color): Replaces specific color in the chroma image with the background image.
  - sticker(sticker_image, background_image, x_pos, y_pos): Adds a sticker image to the background image at specified position.
  - edge_highlight(image): Returns an image with edges highlighted.
## KNN Classifier
- ImageKNNClassifier: Implements a K-Nearest Neighbors classifier for images.
- fit(data): Stores the given set of data and labels.
- distance(image1, image2): Calculates the distance between two images.
- predict(image): Predicts the label for a given image based on the stored data.
