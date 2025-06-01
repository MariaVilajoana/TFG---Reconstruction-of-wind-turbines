# import the necessary packages
import numpy as np
import imutils
import cv2
from sklearn.metrics import mean_squared_error
import random
from scipy.optimize import differential_evolution, minimize
import torch
import torch.nn.functional as F
from sift_manual import sift
import os
import matplotlib.pyplot as plt
import tempfile 
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image, rbd
from superglue_wrapper import initialize_matcher, match_two_images



def orb_fast(image, template):

    # FAST keypoints
    fast = cv2.FastFeatureDetector_create()
    kp1 = fast.detect(image, None)
    kp2 = fast.detect(template, None)

    # BRIEF descriptors
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp1, des1 = brief.compute(image, kp1)
    kp2, des2 = brief.compute(template, kp2)

    return kp1, des1, kp2, des2

def align_images_superglue(
    image: np.ndarray,
    template: np.ndarray,
    base_name_1: str = "",
    base_name_2: str = "",
    debug: bool = True
):
    """

    Args:
      - image:    np.ndarray (BGR) of the image to align.
      - template: np.ndarray (BGR) of the template.
      - base_name_1, base_name_2: strings for naming the debug file.
      - debug:    if True, draws inliers in red and saves the visualization.

    Returns:
      - aligned_image:    the “image” re-projected onto the template.
      - aligned_template: the canvas with the template placed in the same canvas.
      - H:                3×3 homography (float32) mapping image → template.
    """
    # 1) Initialize (or reuse) the static matcher
    if not hasattr(align_images_superglue, "_matcher") or align_images_superglue._matcher is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        align_images_superglue._matcher = initialize_matcher(
            superglue_weights='indoor',
            keypoint_thresh=0.005,
            nms_radius=4,
            max_keypoints=1024,
            sinkhorn_iterations=20,
            match_thresh=0.2,
            device=device
        )
    matcher = align_images_superglue._matcher
    device  = next(matcher.parameters()).device

    # 2) Temporarily save both images to disk to pass them to the wrapper
    tmp1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        cv2.imwrite(tmp1.name, image)
        cv2.imwrite(tmp2.name, template)

        # 3) Directly obtain the matched keypoints at 640×480
        mkpts0_resized, mkpts1_resized = match_two_images(
            tmp1.name,
            tmp2.name,
            matcher=matcher,
            device=device,
            resize=(640, 480),
            resize_float=False
        )
    finally:
        tmp1.close()
        tmp2.close()
        os.unlink(tmp1.name)
        os.unlink(tmp2.name)

    # 4) Scale coordinates from 640×480 → actual resolution
    h_i, w_i = image.shape[:2]
    h_t, w_t = template.shape[:2]
    scale_x_img = w_i / 640.0
    scale_y_img = h_i / 480.0
    scale_x_tmp = w_t / 640.0
    scale_y_tmp = h_t / 480.0

    matched_kpts_img = mkpts0_resized.copy()
    matched_kpts_tmp = mkpts1_resized.copy()
    matched_kpts_img[:, 0] *= scale_x_img
    matched_kpts_img[:, 1] *= scale_y_img
    matched_kpts_tmp[:, 0] *= scale_x_tmp
    matched_kpts_tmp[:, 1] *= scale_y_tmp

    # 5) Verify that there are at least 4 valid matches
    if matched_kpts_img.shape[0] < 4:
        raise ValueError(f"Not enough matches ({matched_kpts_img.shape[0]}) to estimate homography.")

    # 6) Estimate homography with RANSAC
    H, mask = cv2.findHomography(matched_kpts_img, matched_kpts_tmp, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Could not estimate homography (cv2.findHomography returned None).")

    # 7) Build a canvas containing the warped image and the template
    corners = np.float32([[0, 0], [w_i, 0], [w_i, h_i], [0, h_i]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    all_points = np.concatenate(
        (warped_corners.reshape(-1, 2),
         np.array([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]])),
        axis=0
    )
    x_min, y_min = np.floor(all_points.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_points.max(axis=0)).astype(int)

    tx = -x_min if x_min < 0 else 0
    ty = -y_min if y_min < 0 else 0
    new_w = x_max - x_min
    new_h = y_max - y_min

    translation = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    H_adj = translation @ H

    aligned_image = cv2.warpPerspective(image, H_adj, (new_w, new_h))
    aligned_template = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    aligned_template[ty:ty + h_t, tx:tx + w_t] = template

    # 8) If debug=True, draw inliers in red and save the visualization
    if debug:
        inliers_img = matched_kpts_img[mask.ravel() == 1]
        inliers_tmp = matched_kpts_tmp[mask.ravel() == 1]

        vis_dir = 'vis_superglue'
        os.makedirs(vis_dir, exist_ok=True)
        h2 = max(h_i, h_t) * 2
        w2 = (w_i + w_t) * 2
        vis = np.zeros((h2, w2, 3), dtype=np.uint8)

        # Place original images (double size) side by side
        vis[: h_i * 2, : w_i * 2] = cv2.resize(image, (w_i * 2, h_i * 2))
        vis[: h_t * 2, w_i * 2 : w_i * 2 + (w_t * 2)] = cv2.resize(template, (w_t * 2, h_t * 2))

        offset = w_i * 2
        for p1, p2 in zip(inliers_img, inliers_tmp):
            pt1 = (int(p1[0] * 2), int(p1[1] * 2))
            pt2 = (int(p2[0] * 2 + offset), int(p2[1] * 2))
            cv2.line(vis, pt1, pt2, (0, 0, 255), thickness=2)
            cv2.circle(vis, pt1, 4, (0, 0, 255), -1)
            cv2.circle(vis, pt2, 4, (0, 0, 255), -1)

        filename = f"matches_inliers_{base_name_1}_{base_name_2}.png"
        vis_path = os.path.join(vis_dir, filename)
        cv2.imwrite(vis_path, vis)
        print(f"✅ Inlier matches saved at: {vis_path}")

    return aligned_image, aligned_template, H



def align_images_fast(image, template, maxFeatures=500, keepPercent=0.2, debug=False):

    # Convert to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # ORB to detect keypoints and descriptors
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    # Brute-force matching
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    matches = sorted(matches, key=lambda x: x.distance)
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    for (i, m) in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    # ===== SAVE VISUALIZATION OF FILTERED INLIERS WITH THICK LINES =====
    if debug:
        matches_mask = mask.ravel().tolist()
        inlier_ptsA = ptsA[np.array(matches_mask, dtype=bool)]
        inlier_ptsB = ptsB[np.array(matches_mask, dtype=bool)]

        # Create composite image (side by side)
        h1, w1 = image.shape[:2]
        h2, w2 = template.shape[:2]
        height = max(h1, h2)
        vis = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = image
        vis[:h2, w1:w1 + w2] = template

        # Draw matches with matplotlib
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(vis[..., ::-1])  # BGR -> RGB
        for ptA, ptB in zip(inlier_ptsA, inlier_ptsB):
            ptA = tuple(ptA.astype(int))
            ptB = (int(ptB[0] + w1), int(ptB[1]))  # adjust x coordinate
            ax.plot([ptA[0], ptB[0]], [ptA[1], ptB[1]], 'lime', linewidth=2)
            ax.plot(ptA[0], ptA[1], 'ro', markersize=5)
            ax.plot(ptB[0], ptB[1], 'ro', markersize=5)
        ax.axis("off")
        plt.tight_layout()

        # Save image
        os.makedirs("metode_sift_fast", exist_ok=True)
        plt.savefig("metode_sift_fast/matches_inliers_fast.png")
        plt.close()
        print("✅ Inlier matches saved to metode_sift_fast/matches_inliers_fast.png")

    # Align as usual
    h_img, w_img = image.shape[:2]
    corners = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)

    x_coords = transformed_corners[:, 0, 0]
    y_coords = transformed_corners[:, 0, 1]
    min_x, min_y = np.floor([x_coords.min(), y_coords.min()]).astype(int)
    max_x, max_y = np.ceil([x_coords.max(), y_coords.max()]).astype(int)

    min_x = min(min_x, 0)
    min_y = min(min_y, 0)
    max_x = max(max_x, template.shape[1])
    max_y = max(max_y, template.shape[0])

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    T = np.array([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0, 1]], dtype=np.float32)

    H_adjusted = T @ H
    transformed_image = cv2.warpPerspective(image, H_adjusted, (canvas_w, canvas_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    aligned_image = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    aligned_template = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    aligned_template[-min_y:template.shape[0]-min_y, -min_x:template.shape[1]-min_x] = template
    aligned_image[:transformed_image.shape[0], :transformed_image.shape[1]] = transformed_image

    print(H)

    return aligned_image, aligned_template, H


def align_images_sift(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    # 1) Grayscale
    grayA = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 2) SIFT
    sift = cv2.SIFT_create(nfeatures=maxFeatures)
    kpsA, descsA = sift.detectAndCompute(grayA, None)
    kpsB, descsB = sift.detectAndCompute(grayB, None)

    # 3) L2 matcher + crossCheck
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(descsA, descsB)

    # 4) Sort and trim
    matches = sorted(matches, key=lambda m: m.distance)
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    #  → Check minimum matches
    if len(matches) < 4:
        raise ValueError(f"Not enough SIFT matches ({len(matches)}) to estimate homography (minimum 4)")

    # 5) Extract points
    ptsA = np.array([kpsA[m.queryIdx].pt for m in matches], dtype="float32")
    ptsB = np.array([kpsB[m.trainIdx].pt for m in matches], dtype="float32")

    # 6) Homography
    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        raise ValueError("cv2.findHomography returned None")

    # 1) Compute warped corners
    h_img, w_img = image.shape[:2]
    corners = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)

    # 2) Combine with template points
    all_pts = np.vstack([
        warped_corners.reshape(-1, 2),
        [[0, 0], [template.shape[1], template.shape[0]]]
    ])

    x_min, y_min = np.floor(all_pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_pts.max(axis=0)).astype(int)

    canvas_w, canvas_h = x_max - x_min, y_max - y_min

    # ➡️ Avoid huge canvas
    MAX_CANVAS = 5000
    if canvas_w > MAX_CANVAS or canvas_h > MAX_CANVAS:
        raise ValueError(f"Canvas too large ({canvas_w}×{canvas_h}), skipping this pair")

    # 3) Minimal translation to fit everything
    tx, ty = -x_min if x_min < 0 else 0, -y_min if y_min < 0 else 0
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    H_adj = T @ H

    # 4) Safe warp
    warped = cv2.warpPerspective(image, H_adj, (canvas_w, canvas_h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    aligned_image = warped
    aligned_template = np.zeros_like(warped)
    aligned_template[ty:ty + template.shape[0], tx:tx + template.shape[1]] = template

    return aligned_image, aligned_template, H



def mse_shifted(image, template, shift_x, shift_y, min_overlap_size, d):

    # Convert to grayscale (same as before)
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if template.ndim == 3 and template.shape[2] == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    hI, wI = image.shape
    hT, wT = template.shape

    # Cropping indices identical to your version
    y1T = max(0, shift_y)
    y2T = min(hT, hI + shift_y)
    x1T = max(0, shift_x)
    x2T = min(wT, wI + shift_x)

    y1I = max(0, -shift_y)
    y2I = min(hI, hT - shift_y)
    x1I = max(0, -shift_x)
    x2I = min(wI, wT - shift_x)

    # ADD THESE PRINTS
    if d == 1:
        print("---- mse_shifted debug ----")
        print(f" shift = ({shift_x},{shift_y})")
        print(f" image.shape = {image.shape}, template.shape = {template.shape}")
        print(f" template[{y1T}:{y2T}, {x1T}:{x2T}] → shape = ({y2T - y1T}, {x2T - x1T})")
        print(f" image   [{y1I}:{y2I}, {x1I}:{x2I}] → shape = ({y2I - y1I}, {x2I - x1I})")

    patchT = template[y1T:y2T, x1T:x2T]
    patchI = image[y1I:y2I, x1I:x2I]

    area = patchT.size
    if area == 0 or patchI.size == 0 or area < min_overlap_size:
        return np.inf

    mse = mean_squared_error(patchT.ravel(), patchI.ravel())
    if d == 1:
        print(f" → Calculated MSE = {mse}")
        print("----------------------------\n")
    return mse




def align(image, template, shiftx, shifty):
    max_shape_y = max(image.shape[0], template.shape[0])
    max_shape_x = max(image.shape[1], template.shape[1])

    aligned_image = np.zeros((max_shape_y+ abs(shifty), max_shape_x+abs(shiftx),3), dtype=np.uint8)
    aligned_image [ max(0, shifty): max(0, shifty) + image.shape[0], max(0, shiftx): max(0, shiftx) + image.shape[1]] = image
    aligned_template =  np.zeros((max_shape_y+ abs(shifty), max_shape_x+abs(shiftx),3), dtype=np.uint8)
    aligned_template [ max(0, -shifty): max(0, -shifty) + template.shape[0], max(0, -shiftx): max(0, -shiftx) + template.shape[1]] = template
    
    return aligned_image, aligned_template


def exhaustive_search(image, template):
    shift_x_increment = 1
    shift_y_increment = 1

    shift_x_range = range(-105,-95,shift_x_increment)
    shift_y_range = range(-1120,-1070,shift_y_increment)

    min_overlap_size = (template.shape[0]*template.shape[1])*0.3

    mse=np.inf
    shiftx = 0
    shifty = 0

    for shift_x in shift_x_range:
        for shift_y in shift_y_range:

            new_mse = mse_shifted(image, template, shift_x, shift_y, min_overlap_size,0)
            
            if new_mse < mse:
                    
                mse = new_mse
                shiftx = shift_x
                shifty = shift_y
           
    print(shiftx, shifty)
    print(mse_shifted(image, template, shiftx, shifty, min_overlap_size,0))
    return align(image, template, shiftx, shifty)


def mse_shifted_torch(image, template, shift_x, shift_y, min_overlap_size):
    # image and template: tensors [1, 3, H, W] on GPU (RGB)

    # Convert tensors to numpy and move to CPU
    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    tmp_np = template.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Convert to grayscale with OpenCV
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    tmp_gray = cv2.cvtColor(tmp_np, cv2.COLOR_RGB2GRAY)

    # Convert back to float32 tensors on GPU with shape [1, 1, H, W]
    img_tensor = torch.tensor(img_gray, dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
    tmp_tensor = torch.tensor(tmp_gray, dtype=torch.float32, device=template.device).unsqueeze(0).unsqueeze(0)

    _, _, H_img, W_img = img_tensor.shape
    _, _, H_tmp, W_tmp = tmp_tensor.shape

    # Cropping coordinates
    y1_tmp = max(0, shift_y)
    y2_tmp = min(H_tmp, H_img + shift_y)
    x1_tmp = max(0, shift_x)
    x2_tmp = min(W_tmp, W_img + shift_x)

    y1_img = max(0, -shift_y)
    y2_img = min(H_img, H_tmp - shift_y)
    x1_img = max(0, -shift_x)
    x2_img = min(W_img, W_tmp - shift_x)

    # Crop overlapping regions
    overlap_template = tmp_tensor[:, :, y1_tmp:y2_tmp, x1_tmp:x2_tmp]
    overlap_image = img_tensor[:, :, y1_img:y2_img, x1_img:x2_img]

    # Validate minimum size
    if overlap_template.numel() < min_overlap_size:
        return float('inf')

    # Compute MSE
    diff = (overlap_template - overlap_image) ** 2
    mse = diff.mean().item()

    return mse

def align_torch(image_np, template_np, shiftx, shifty):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H_img, W_img = image_np.shape[:2]
    H_tmp, W_tmp = template_np.shape[:2]

    max_shape_y = max(H_img, H_tmp) + abs(shifty)
    max_shape_x = max(W_img, W_tmp) + abs(shiftx)

    # Create empty tensors
    aligned_image = torch.zeros((3, max_shape_y, max_shape_x), dtype=torch.uint8, device=device)
    aligned_template = torch.zeros((3, max_shape_y, max_shape_x), dtype=torch.uint8, device=device)

    # Convert images to tensors and normalize
    image = torch.tensor(image_np, dtype=torch.uint8).permute(2, 0, 1).to(device)
    template = torch.tensor(template_np, dtype=torch.uint8).permute(2, 0, 1).to(device)

    # Calculate insertion positions
    sy, sx = max(0, shifty), max(0, shiftx)
    tsy, tsx = max(0, -shifty), max(0, -shiftx)

    # Insert images at shifted position
    aligned_image[:, sy:sy + H_img, sx:sx + W_img] = image
    aligned_template[:, tsy:tsy + H_tmp, tsx:tsx + W_tmp] = template

    # Convert back to numpy
    aligned_image_np = aligned_image.permute(1, 2, 0).cpu().numpy()
    aligned_template_np = aligned_template.permute(1, 2, 0).cpu().numpy()

    return aligned_image_np, aligned_template_np


def exhaustive_search_torch(image_np, template_np):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # Convert images to normalized tensors on GPU
    image = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    template = torch.tensor(template_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255

    shift_x_increment = 10
    shift_y_increment = 10

    shift_x_range = range(-600, 300, shift_x_increment)
    shift_y_range = range(-2000, -200, shift_y_increment)

    min_overlap_size = (template.shape[-2] * template.shape[-1]) * 0.15

    mse = float('inf')
    shiftx = 0
    shifty = 0

    for shift_x in shift_x_range:
        for shift_y in shift_y_range:
            new_mse = mse_shifted_torch(image, template, shift_x, shift_y, min_overlap_size)

            if new_mse < mse:
                mse = new_mse
                shiftx = shift_x
                shifty = shift_y

    print("Best shift found:", shiftx, shifty)
    aligned_image, aligned_template = align_torch(image_np, template_np, shiftx, shifty)
    return aligned_image, aligned_template, shiftx, shifty




def randomized_search(image, template):

    # Initialize variables for the optimization process
    shiftx, shifty, min_mse = 0, -1500, 1e6
    previous_min_z = min_mse
    iteration = 0
    convergence_threshold = 1e-5

    # Iterative process to find the shift that minimizes MSE
    while True:
        if iteration % 2 == 0:
            # In even iterations, vary y while keeping x fixed
            y_values = [shifty + random.randint(-50, 50) for _ in range(10)]
            new_data = [(shiftx, y, mse_shifted(image, template, shiftx, y, 0)) for y in y_values]
        else:
            # In odd iterations, vary x while keeping y fixed
            x_values = [shiftx + random.randint(-10, 10) for _ in range(10)]
            new_data = [(x, shifty, mse_shifted(image, template, x, shifty, 0)) for x in x_values]

        # Find the minimum MSE in the new set of data
        new_shiftx, new_shifty, new_min_z = min(new_data, key=lambda item: item[2])

        # Check if the change in MSE is below the convergence threshold
        # print("iteration:", iteration)
        # print(abs(new_min_z - previous_min_z))
        if abs(new_min_z - previous_min_z) < convergence_threshold:
            break

        # Update minimum values for the next iteration
        shiftx, shifty, min_mse = new_shiftx, new_shifty, new_min_z
        previous_min_z = new_min_z
        iteration += 1

    print(shiftx, shifty)
    aligned_image, aligned_template = align(image, template, shiftx, shifty)
    return aligned_image, aligned_template, shiftx, shifty



def differential_ev(image, template):

    # Objective function for optimization
    def objective(params):
        shift_x = int(params[0])
        shift_y = int(params[1])
        return mse_shifted(image, template, shift_x, shift_y, template.shape[0] * template.shape[1] * 0.3, 0)

    # Define bounds for the shift values
    bounds = [(-1000, 1001), (-2000, 2000)]

    # Perform differential evolution optimization
    result = differential_evolution(objective, bounds, workers=1)
    shift_x, shift_y = result.x

    shift_x = int(shift_x)
    shift_y = int(shift_y)

    min_mse = result.fun
    x_min = shift_x
    y_min = shift_y

    # Search in a small range around the found minimum
    for x in range(shift_x - 5, shift_x + 5):
        for y in range(shift_y - 5, shift_y + 5):

            new_mse = mse_shifted(image, template, x, y, 0, 0)

            # Update if a lower MSE is found
            if new_mse < min_mse:
                min_mse = new_mse
                x_min = x
                y_min = y

    print(shift_x, shift_y)
    aligned_image, aligned_template = align(image, template, x_min, y_min)
    return aligned_image, aligned_template, x_min, y_min




def mse_affine(params, image, template, min_overlap_size):
    a, b, c, d, e, f = params

    # Convert to grayscale if they are not already
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Affine transformation matrix (scaling, rotation, and translation)
    M = np.array([
        [a, b, c],
        [d, e, f]
    ], dtype=np.float32)

    # Apply the transformation to image using OpenCV
    transformed_image = cv2.warpAffine(
        image, M, (template.shape[1], template.shape[0]), borderMode=cv2.BORDER_REPLICATE
    )

    # Create a mask to define valid areas after transformation
    mask = np.ones_like(image, dtype=np.uint8) * 255
    transformed_mask = cv2.warpAffine(
        mask, M, (template.shape[1], template.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # Verify overlap area
    nonzero_pixels = np.count_nonzero(transformed_mask)
    if nonzero_pixels < min_overlap_size:
        return float('inf')  # Insufficient overlap, discard

    # Apply the mask to both images
    template_masked = cv2.bitwise_and(template, template, mask=transformed_mask)
    transformed_image_masked = cv2.bitwise_and(transformed_image, transformed_image, mask=transformed_mask)

    # Compute MSE only in the overlapping region
    mse = np.sum((template_masked - transformed_image_masked) ** 2) / nonzero_pixels
    return mse




def optimize_translation_powell(image, template):
    # Función objetivo para minimizar el MSE según traslación
    def objective(params):
        dx = int(params[0])
        dy = int(params[1])
        return mse_shifted(image, template, dx, dy, min_overlap_size)

    # Definir el tamaño mínimo de solapamiento (30% del template)
    min_overlap_size = (template.shape[0] * template.shape[1]) * 0.3

    # Parámetros iniciales
    initial_params = [0, -1000]  # dx, dy
    bounds = [(-500, 500), (-2000, 1700)]  # puedes ajustar

    # Optimización
    result = minimize(objective, initial_params, method='Powell', bounds=bounds)

    dx_opt = int(result.x[0])
    dy_opt = int(result.x[1])
    print(objective(result.x))
    print(f"🧭 Mejor desplazamiento encontrado: dx={dx_opt}, dy={dy_opt}")

    # Alinear usando esos desplazamientos
    aligned_image, aligned_template = align(image, template, dx_opt, dy_opt)
    return aligned_image, aligned_template, dx_opt, dy_opt



def mse_affine(params, image, template, min_overlap_size):
    """
    Compute the mean squared error (MSE) between the template and the image
    after applying a full 2D affine transformation with 6 degrees of freedom.

    params: [a11, a12, a21, a22, tx, ty]
        where the affine matrix is:
            [ a11  a12  tx ]
            [ a21  a22  ty ]
    image: input image (BGR)
    template: target template (BGR)
    min_overlap_size: minimum number of overlapping pixels required
    """
    a11, a12, a21, a22, tx, ty = params

    # Convert both images to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_tmp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Build the 2×3 affine transformation matrix
    M = np.array([
        [a11, a12, tx],
        [a21, a22, ty]
    ], dtype=np.float32)

    # Apply the affine transform to the image
    transformed_img = cv2.warpAffine(
        gray_img,
        M,
        (template.shape[1], template.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Build a mask (255 where valid, 0 outside after warp)
    mask = np.ones_like(gray_img, dtype=np.uint8) * 255
    transformed_mask = cv2.warpAffine(
        mask,
        M,
        (template.shape[1], template.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Count the number of valid (nonzero) pixels in the transformed mask
    nonzero_pixels = np.count_nonzero(transformed_mask)
    if nonzero_pixels < min_overlap_size:
        return float('inf')  # insufficient overlap

    # Mask the transformed image and the template to compute MSE on overlap only
    tmp_masked = cv2.bitwise_and(gray_tmp, gray_tmp, mask=transformed_mask)
    img_masked = cv2.bitwise_and(transformed_img, transformed_img, mask=transformed_mask)

    # Compute MSE over the overlapping region
    diff = (tmp_masked.astype(np.float32) - img_masked.astype(np.float32)) ** 2
    mse = diff.sum() / nonzero_pixels
    return mse


def align_affine_powell(image, template):
    """
    Find and apply a full 6-DOF affine alignment between image and template.
    Returns the aligned image, aligned template (padded on the same canvas),
    and the final 2×3 affine matrix.
    """
    # Bounds for [a11, a12, a21, a22, tx, ty]
    # a11, a22: scale/stretch factors (allowing slight deviations around 1)
    # a12, a21: shear/rotation components (bounded between -1 and 1)
    # tx, ty: translation (in pixels)
    bounds = [
        (0.5, 2.0),    # a11
        (-1.0, 1.0),   # a12
        (-1.0, 1.0),   # a21
        (0.5, 2.0),    # a22
        (-500, 500),   # tx
        (-2000, 2000)  # ty
    ]
    # Initial guess: identity with no shear or translation
    initial_params = [1.0, 0.0, 0.0, 1.0, 0.0, -1000.0]
    min_overlap_size = (template.shape[0] * template.shape[1]) * 0.3

    print(f"Initial parameters: {initial_params}")

    # Minimize the MSE over the affine parameters
    result = minimize(
        mse_affine,
        initial_params,
        args=(image, template, min_overlap_size),
        bounds=bounds,
        method='Powell',
        options={'xtol': 1e-8, 'ftol': 1e-8, 'disp': True}
    )

    print(f"Optimization result: {result}")
    print(f"Optimized parameters: {result.x}")
    print(f"Minimum MSE found: {result.fun}")

    a11_opt, a12_opt, a21_opt, a22_opt, tx_opt, ty_opt = result.x

    # Build the final affine matrix (2×3)
    M = np.array([
        [a11_opt, a12_opt, tx_opt],
        [a21_opt, a22_opt, ty_opt]
    ], dtype=np.float32)

    # Determine the canvas size to fit both the transformed image and the template
    h_img, w_img = image.shape[:2]
    h_tmp, w_tmp = template.shape[:2]

    # Compute the four corners of the image, then transform them
    corners_img = np.array([
        [0, 0],
        [w_img, 0],
        [w_img, h_img],
        [0, h_img]
    ], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.transform(corners_img, M)

    # Also include the template corners (untouched)
    corners_tmp = np.array([
        [0, 0],
        [w_tmp, 0],
        [w_tmp, h_tmp],
        [0, h_tmp]
    ], dtype=np.float32).reshape(-1, 1, 2)

    all_corners = np.vstack([transformed_corners.reshape(-1, 2), corners_tmp.reshape(-1, 2)])
    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    # Build a translation that ensures both fit fully into the canvas
    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y]
    ], dtype=np.float32)

    # Convert the original 2×3 affine matrix M into 3×3 for concatenation
    M_hom = np.vstack([M, [0, 0, 1]])
    M_final_hom = T @ M_hom   # 3×3
    M_final = M_final_hom[:2, :]  # back to 2×3

    # Warp the image onto the new canvas
    aligned_image = cv2.warpAffine(
        image,
        M_final,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Create a blank canvas for the template and place it
    aligned_template = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    # The template should be positioned at (-min_x, -min_y) on the canvas
    x_offset = -min_x
    y_offset = -min_y
    aligned_template[y_offset:y_offset + h_tmp, x_offset:x_offset + w_tmp] = template

    return aligned_image, aligned_template, M



def mse_homography(params, image, template, min_overlap_size):
    tx, ty, a, b, c, d, e, f = params

    # Build the homography matrix (3×3)
    H = np.array([
        [1 + a, b,     tx],
        [c,     1 + d, ty],
        [e,     f,     1]
    ], dtype=np.float32)

    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Apply the homography to the image
    warped = cv2.warpPerspective(
        image_gray,
        H,
        (template.shape[1], template.shape[0]),
        borderMode=cv2.BORDER_CONSTANT
    )

    # Create a validity mask
    mask = np.ones_like(image_gray, dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(
        mask,
        H,
        (template.shape[1], template.shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Check the overlap area
    nonzero = np.count_nonzero(warped_mask)
    if nonzero < min_overlap_size:
        return float('inf')  # Discard if overlap is insufficient

    # Apply the mask to both images
    template_masked = cv2.bitwise_and(template_gray, template_gray, mask=warped_mask)
    warped_masked = cv2.bitwise_and(warped, warped, mask=warped_mask)

    # Compute MSE only in the valid region
    mse = np.sum((template_masked - warped_masked) ** 2) / nonzero
    return mse


def align_homography_powell(image, template):
    bounds = [
        (-500, 500),   # tx
        (-2000, 2000), # ty
        (-0.3, 0.3),   # a
        (-0.3, 0.3),   # b
        (-0.3, 0.3),   # c
        (-0.3, 0.3),   # d
        (-1e-4, 1e-4), # e
        (-1e-4, 1e-4)  # f
    ]
    initial_params = [0, -1000, 0, 0, 0, 0, 0, 0]

    print(f"Parámetros iniciales: {initial_params}")
    min_overlap_size = template.shape[0] * template.shape[1] * 0.3
    result = minimize(mse_homography, initial_params, args=(image, template, min_overlap_size), bounds=bounds, method='Powell')

    print(f"Optimísation result: {result}")
    print(f"Optimízed parameters: {result.x}")
    print(f"Minimum MSE: {result.fun}")

    tx, ty, a, b, c, d, e, f = result.x

    H = np.array([
        [1 + a, b,     tx],
        [c,     1 + d, ty],
        [e,     f,     1]
    ], dtype=np.float32)

    # Get transformed size using corner coordinates
    h_img, w_img = image.shape[:2]
    h_tmp, w_tmp = template.shape[:2]

    corners = np.array([
        [0, 0],
        [w_img, 0],
        [w_img, h_img],
        [0, h_img]
    ], dtype=np.float32).reshape(-1, 1, 2)

    transformed_corners = cv2.perspectiveTransform(corners, H)

    # Also include the template size
    all_corners = np.vstack([
        transformed_corners.reshape(-1, 2),
        np.array([[0, 0], [w_tmp, 0], [w_tmp, h_tmp], [0, h_tmp]])
    ])

    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    # Adjust homography to fit into canvas
    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float32)

    H_adjusted = T @ H

    # Warp image
    aligned_image = cv2.warpPerspective(image, H_adjusted, (canvas_w, canvas_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Canvas for template
    aligned_template = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    aligned_template[-min_y:h_tmp - min_y, -min_x:w_tmp - min_x] = template

    return aligned_image, aligned_template, H



def align_images_lightglue(image, template, debug=True):
    """
    Aligns `image` onto `template` using LightGlue (SuperPoint + LightGlue).
    Saves a visualization of the matches if debug=True.
    Returns (aligned_image, aligned_template, H).
    """

    # Create folder for visualization
    vis_dir = "vis_lightblue"
    os.makedirs(vis_dir, exist_ok=True)

    # Helper function to convert to tensor
    def img_to_tensor(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_norm = img_gray.astype(np.float32) / 255.0
        return torch.from_numpy(img_norm)[None, None].to("cpu")

    # Prepare tensors
    img0 = img_to_tensor(template)  # template
    img1 = img_to_tensor(image)     # image to align

    # Initialize models on CPU
    extractor = SuperPoint(max_num_keypoints=2048).eval().to("cpu")
    matcher   = LightGlue(features='superpoint').eval().to("cpu")

    # Extract features and perform matching
    feats0 = extractor.extract(img0)
    feats1 = extractor.extract(img1)
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]

    # Get the matched keypoints
    matches = matches01['matches']
    valid = (matches[:, 0] >= 0) & (matches[:, 1] >= 0)
    kpts0 = feats0['keypoints'][matches[valid][:, 0]].cpu().numpy()
    kpts1 = feats1['keypoints'][matches[valid][:, 1]].cpu().numpy()

    if len(kpts0) < 4:
        raise ValueError("❌ Not enough matches to estimate homography.")

    # Estimate homography (image → template)
    H, mask = cv2.findHomography(kpts1, kpts0, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("❌ Could not compute homography.")

    # Compute new adjusted canvas
    h_t, w_t = template.shape[:2]
    corners = np.array([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    min_x = int(min(warped_corners[:,0,0].min(), 0))
    min_y = int(min(warped_corners[:,0,1].min(), 0))
    max_x = int(max(warped_corners[:,0,0].max(), w_t))
    max_y = int(max(warped_corners[:,0,1].max(), h_t))
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    T = np.array([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0, 1]], dtype=np.float32)
    H_adj = T @ H

    # Warp the image and create aligned_image / aligned_template
    warped = cv2.warpPerspective(image, H_adj, (canvas_w, canvas_h), borderValue=(0,0,0))
    aligned_image = warped
    aligned_template = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    aligned_template[-min_y: h_t-min_y, -min_x: w_t-min_x] = template

    # Optional visualization of inliers
    if debug:
        inliers_img = kpts1[mask.ravel() == 1]
        inliers_tmp = kpts0[mask.ravel() == 1]
        vis = np.zeros((canvas_h, canvas_w*2, 3), dtype=np.uint8)
        # place template and image side by side (scaled to canvas_h height)
        tmp_vis = cv2.resize(template, (canvas_w, canvas_h))
        img_vis = cv2.resize(image,   (canvas_w, canvas_h))
        vis[:, :canvas_w] = tmp_vis
        vis[:, canvas_w:] = img_vis
        offset = canvas_w
        for pt0, pt1 in zip(inliers_tmp, inliers_img):
            p0 = (int(pt0[0] - min_x), int(pt0[1] - min_y))
            p1 = (int(pt1[0]),             int(pt1[1]))
            p1 = (p1[0] + offset, p1[1])
            cv2.line(vis, p0, p1, (255,255,0), 2)
            cv2.circle(vis, p0, 4, (255,255,0), -1)
            cv2.circle(vis, p1, 4, (255,255,0), -1)
        cv2.imwrite(os.path.join(vis_dir, f"matches_lightblue.png"), vis)
        print(f"✅ Visualization saved at: {vis_dir}/matches_lightblue.png")

    return aligned_image, aligned_template, H



