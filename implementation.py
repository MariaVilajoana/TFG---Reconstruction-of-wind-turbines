
from align import align_images_fast, exhaustive_search_torch,align_images_lightglue, randomized_search,align_translation_powell, exhaustive_search_torch,differential_ev, align_affine_powell, align_homography_powell,exhaustive_search, align_images_superglue ,align_images_sift, exhaustive_search
import numpy as np
import argparse
import imutils
import cv2
import torch


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True,
    help="path to input image that we'll align to template")
ap.add_argument("-t", "--template", required=True,
    help="path to input template image")
ap.add_argument("-m", "--metode", required=True)
ap.add_argument("-n", "--name", required=True)
args = vars(ap.parse_args())


# load the input image and template from disk
print("[INFO] loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])
image = cv2.resize(image, None, fx=0.5, fy=0.5)
template = cv2.resize(template, None, fx=0.5, fy=0.5)
metode = args["metode"]
name = args["name"]

# image = cv2.resize(image,None, fx=0.5, fy=0.5)
# template = cv2.resize(template,None, fx=0.5, fy=0.5)

# align the images
print("[INFO] aligning images...")

if metode == "fast":
    print("method: fast")
    aligned_image, aligned_template,_ = align_images_fast(image, template, debug=True)

if metode == "superglue":
    print("method: superglue")

    base_name_1 = name      
    base_name_2 = name       

    aligned_image, aligned_template, H = align_images_superglue(
        image,
        template,
        base_name_1,
        base_name_2,
        debug=True
    )

if metode == "sift":
    print("method: sift")
    aligned_image, aligned_template,_ = align_images_sift(image, template, debug=False)

if metode == "exhaustive":
    print("method: exhaustive")
    aligned_image, aligned_template,_,__ = exhaustive_search(image, template)

if metode == "lightglue":
    print("method: lightglue")
    aligned_image, aligned_template, H = align_images_lightglue(image, template, debug=False)

if metode == "exhaustive_torch":
    print("method: exhaustive torch")
    aligned_image, aligned_template,_,__ = exhaustive_search_torch(image, template)

if metode == "randomized":
    print("method: randomized")
    aligned_image, aligned_template,_,__ = randomized_search(image, template)

if metode == "diff_ev":
    print("method: differential evolution")
    aligned_image, aligned_template,_,__ = differential_ev(image, template)

if metode == "affine_powell":
    print("method: affine powell")
    aligned_image, aligned_template, _ = align_affine_powell(image, template)

if metode == "homography_powell":
    print("method: homography powell")
    aligned_image, aligned_template, _ = align_homography_powell(image, template)
    
if metode == "translation_powell":
    print("method: translation powell")
    aligned_image, aligned_template,_,__ = align_translation_powell(image, template)
    

# resize both the aligned and template images so we can easily
# visualize them on our screen
aligned_image = imutils.resize(aligned_image, width=500)
aligned_template = imutils.resize(aligned_template, width=500)


# our first output visualization of the image alignment will be a
# side-by-side comparison of the output aligned image and the
# template
stacked = np.hstack([aligned_image, aligned_template])

# our second image alignment visualization will be *overlaying* the
# aligned image on the template, that way we can obtain an idea of
# how good our image alignment is
overlay = aligned_template.copy()
output = aligned_image.copy()
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

# show the two output image alignment visualizations
cv2.imwrite(f"a{name}_alineacion_s_{metode}.png", stacked)
cv2.imwrite(f"a{name}_alineacion_o_{metode}.png", output)
