"""
Last Edited: 2026-10-01
Author: Youngsu Choi

Image manipulation implementation

"""

import numpy as np
import PIL
import matplotlib.pyplot as plt
import subprocess
import io


def visualize_image(img):   

    if isinstance(img, np.ndarray):
        img = PIL.Image.fromarray(img.astype(np.uint8))

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    subprocess.run(["kitty", "+kitten", "icat"], input=buffer.getvalue())


class Imager():

    def __init__(self):
        self.img = None
        self.orig_img = None
        self.color_order = "RGB"
        self.max_pixel_val = None
        self.min_pixel_val = None

    def _clip(self):
        self.img = np.clip(self.img, self.min_pixel_val, self.max_pixel_val).astype(np.uint8)
    
    def load_image(self, path, verbose=False):
        self.orig_img = self.img = np.array(PIL.Image.open(path))
        self.max_pixel_val = np.max(self.img)
        self.min_pixel_val = np.min(self.img) 

        print(f"Successfully Loaded {path}")

        if verbose:
            print("----- IMAGE STATISTICS -----")
            print(f"Shape: {self.img.shape}")
            print(f"dtype: {self.img.dtype}")
            print(f"min: {self.min_pixel_val}")
            print(f"max: {self.max_pixel_val}")
            print("----- IMAGE STATISTICS -----")
    
    def get_image_dim(self):
        return self.img.shape if self.img is not None else None
    
    def show_image(self):

        if self.img.size == 0:
            print("Image EMPTY!")
            return 

        visualize_image(self.img)

    def crop(self, anchor: tuple[int, int], row, col, origin="bottom-right"):
        """Crops the currently loaded image

        Parameters
        ----------
        anchor: Where you'd like to start the crop (row, col) 
        row, col: How much to crop (height, width) 
        origin: In which direction to crop

        If given height and width exceed the image limit no error is thrown 
        as it will just crop to the edge of image

        Error is only thrown if given anchor is out of bounds

        """

        if not ((0, 0) <= tuple(anchor[:2]) < self.img.shape[:2]):
            raise IndexError(f"Anchor is out of bounds. \n Current image dimensions are {self.img.shape}")

        if origin == "bottom-right":
            clipped = (min(anchor[0] + row, self.img.shape[0]), min(anchor[1] + col, self.img.shape[1]))
            self.img = self.img[anchor[0]:clipped[0], anchor[1]:clipped[1], :]

        elif origin == "bottom-left":
            clipped = (min(anchor[0] + row, self.img.shape[0]), max(anchor[1] - col, 0))
            self.img = self.img[anchor[0]:clipped[0], clipped[1]:anchor[1], :]
        elif origin == "top-right":
            clipped = (max(0, anchor[0] - row), min(self.img.shape[1], anchor[1] + col))
            self.img = self.img[clipped[0]:anchor[0], anchor[1]:clipped[1], :]
        else:
            clipped = (max(0, anchor[0] - row), max(0, anchor[1] - col))
            self.img = self.img[clipped[0]:anchor[0], clipped[1]:anchor[1], :]

    def horizontalFlip(self):
        """Flips the image over the horizontal axis"""
        self.img = self.img[::-1, :, :]

    def verticalFlip(self):
        """Flips the image over the vertical axis"""
        self.img = self.img[:, ::-1, :]

    def rotate(self, clockwise=True):
        """Rotates image 90 degrees once"""

        if clockwise:
            self.img = np.transpose(self.img, (1, 0, 2))[:, ::-1, :]
        else: 
            self.img = np.transpose(self.img, (1, 0, 2))[::-1, :, :]

    def get_channel(self, color='R'):
        """Returns a copy of a single RGB Channel from current image

        WARNING
        -------
        The image when loaded first time is always assumed to be in R -> G -> B format. 
        """

        return self.img[:,:, self.color_order.index(color.upper())].copy()

    def swap_channels(self, order="RGB"):
        """Re-orders the colours channels to the order given.

        WARNING
        -------
        Changing the channel order changes the actual current image that is being held.
        Refer to get_channel() or reset() for information
        
        """
        order = order.upper()
        order_idx = [self.color_order.index(i) for i in order]

        self.img = self.img[:, :, order_idx]
        self.color_order = order

    def change_brightness(self, value):
        """
        
        """

        self.img = self.img.astype(np.float16)
        self._clip()

    def change_constrast(self, factor):

        self.img = self.img.astype(np.float16)
        mean = self.img.mean()
        self.img = (self.img - mean)*factor + mean

        self._clip()

    def invert(self):
        self.img = self.max_pixel_val - self.img 

    def grayscale(self):

        coeff = np.array([0.299, 0.587, 0.114])

        # self.img = np.dot(self.img, coeff).astype(np.uint8)
        # self.img = np.tensordot(self.img, coeff, axes=([2], [0])).astype(np.uint8)
        self.img = (self.img * coeff).sum(axis=2).astype(np.uint8)

    def threshold(self):

        self.grayscale()

        self.img = ((self.img > 128) * 255).astype(np.uint8)
        # self.img = np.where((self.img > 128), 255, 0).astype(np.uint8) 


    def reset(self):
        self.img = self.orig_img




if __name__ == "__main__":
    path = "./test2.jpg"
    imager = Imager()
    imager.load_image(path, verbose=True)
    imager.show_image()
    imager.threshold()
    imager.show_image()
    