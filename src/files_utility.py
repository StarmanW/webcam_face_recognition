import os
import re

"""
    Files Utility class
    for listing or manipulating files
"""

# Image Extension Regex
IMG_EXT_REGEX = "^.*\.(jpg|png)$"

class FilesUtility:
    """
        Get image paths
    """
    @staticmethod
    def getImages(path):
        img_paths = []
        img_dir =  os.path.abspath(path)
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if re.match(IMG_EXT_REGEX, file):
                    img_paths.append(os.path.join(root, file))
        return img_paths
