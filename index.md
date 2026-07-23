# Optical Character Recognition

OCR is prevalent whenever scan a file, translate documents or extract text from a screenshot. But how does it work? This project combines feature detection along with a CNN to create a basic OCR system capable of detecting text.

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| William M. | Basis Independent Silicon Valley | CS/AI | Incoming Sophmore

**Replace the BlueStamp logo below with an image of yourself and your completed project. Follow the guide [here](https://tomcam.github.io/least-github-pages/adding-images-github-pages-site.html) if you need help.**

![Headstone Image](logo.svg)
  
# Final Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/zD_gSm0l_ls?si=4D8yFeMzz-VK62Y3" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

My final milestone is to link up both my model and my text detection algorithm to a camera to identify text form the enrivornment. This was built using Raspberry Pi's picamera2 module, allowing me to use the given Raspberry Pi camera to take images. The biggest issue with this milestone was all of the libraries that had to be imported-some libraries took up way too much space, and due to how Linux works, you cannot even install most python libraries without first creating a virtual environment.

(2nd part is TO BE COMPLETED)

# Second Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/aNhNrOw-PBY?si=VwLUmwP46oCStIqq" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

My second milestone is to train a model to identify individual characters of text. For this I decided at first on a linear model, but then I switched to a 2-layer CNN to learn features. This final model(trained on 15 iterations of the Extended MNIST balanced dataset) achieved 87% accuracy on the testset.

The big issue with this model is that there are a few very similar characters, which account for almost 10% of the dataset, those being 1, l, I and 0/O. To tell the difference between these I would need more complicated models(and more training time).

My final milestone will be to connect this model sequence to a Raspberry Pi to be able to capture images from the environment.

# First Milestone


<iframe width="560" height="315" src="https://www.youtube.com/embed/6XHCfbCdElQ?si=llJrcRIMrLr4BYJH" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


My project is to self-build a basic OCR system from scratch, and connect it to a Raspberry Pi camera to be able to extract text from the environment. This can be split into 3 parts: using a camera to take a picture of a page, identifying where text is on said page and recognizing individual characters of text for output.

My first milestone is to write an algorithm to detect where text is in a file. For this, after binarizing the image via a filter, I used a kernel to iterate over subsets of pixels and then applied a threshold to identify groups of text. The main challenge with this algorithm is that due to uneven text spacing, it is not possible to identify all text equally well, as it could bunch text up on accident.

The next milestone is to train a model to identify individual characters of English text, uppercase and lowercase.

# Schematics 

![CNN Schematic](network-2.png)

Text detection Schematic: TO BE ADDED

# Code

Colab for CNN: https://colab.research.google.com/drive/1JPxSS0inNdVZi3DgPQ9Ln58SS83VIB4u

Identifying where text is:

```python3
# bounding box sum of values
# take any connected section with high enough score
# remove lines
# continuously rotate image until you get a vertical enough line and split
# pass those as matrices to trained model

import cv2
from PIL import Image
import numpy as np
import scipy
import matplotlib.pyplot as plt

with Image.open("/Users/williammao/Desktop/screenshot5.png") as im:
    im_cpy = np.array(im.convert("L"))
    im_cpy = (im_cpy < 200)
    filter_s = 25
    sk = np.array(np.lib.stride_tricks.sliding_window_view(im_cpy, (15, 15)))
    sk = (np.sum(np.sum(sk, axis = 2), axis = 2) > 50)
    lbl, _ = scipy.ndimage.label(sk, structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    slices = scipy.ndimage.find_objects(lbl)
    for idx, slc in enumerate(slices, start=1):
        if slc != None:
            cropped_component = im_cpy[slc]
            plt.imshow(cropped_component)
            plt.show()
```

Camera capturing:

```python3
from picamera2 import Picamera2, Preview
from pynput import mouse, keyboard

cam = Picamera2()
cnfg = cam.create_preview_configuration()
cam.configure(cnfg)
cam.start_preview(Preview.QTGL)
cam.start()
with keyboard.Events() as events:
    for event in events:
        if event.key == keyboard.Key.enter:
            break
cam.capture_file("test.jpg")
cam.stop_preview()
```
# Parts Used
Here's where you'll list the parts in your project. To add more rows, just copy and paste the example rows below.
Don't forget to place the link of where to buy each component inside the quotation marks in the corresponding row after href =. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize this to your project needs. 

| **Part** | **Note** | **Price** | **Link** |
|:--:|:--:|:--:|:--:|
| CanaKit Raspberry Pi 4 Starter Kit(32 GB EVO+, Premium Black Case)/4 GB Ram | Raspberry Pi | $119.00 | <a href="https://www.canakit.com/raspberry-pi-4-starter-kit.html?srsltid=AfmBOoqactFRiLuNfSdSD2wUCHARI6ijl9eRnVXGIvdhOYxsMbABwKK7"> Link </a> |
| Generic Keyboard/Mouse | Raspberry Pi Monitor connection | $16 |  |

Note on the keyboard/mouse: Once a VNC is set up, there is no more need for a keyboard/mouse, but due to higher resolution of the monitor ist is still recommended.

# Other Resources/Examples
One of the best parts about Github is that you can view how other people set up their own work. Here are some past BSE portfolios that are awesome examples. You can view how they set up their portfolio, and you can view their index.md files to understand how they implemented different portfolio components.
- [Example 1](https://trashytuber.github.io/YimingJiaBlueStamp/)
- [Example 2](https://sviatil0.github.io/Sviatoslav_BSE/)
- [Example 3](https://arneshkumar.github.io/arneshbluestamp/)

To watch the BSE tutorial on how to create a portfolio, click here.

Something's fishy...
