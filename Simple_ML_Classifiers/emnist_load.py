import zipfile
import numpy as np
import cv2
import binascii
from os import listdir
from os.path import isfile, join
import os
import codecs
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from thinning import *
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.signal import savgol_filter
from scipy import signal
from scipy.signal import argrelextrema
from mnist import MNIST


mndata = MNIST('./emnist-mnist')
images, labels = mndata.load_training()


print(images.shape)