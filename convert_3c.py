import numpy as np


class ConvertTo3C:
    def __init__(self, inputImage):
        self.inputImage = inputImage

    def convert(self):
        output = np.zeros((np.array(self.inputImage).shape[0], np.array(self.inputImage).shape[1], 3))
        output[:, :, 0] = self.inputImage  # same value in each channel
        output[:, :, 1] = self.inputImage
        output[:, :, 2] = self.inputImage
        return output
