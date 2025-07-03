import os
import random
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

def get_random_position(image_shape, obj_shape, existing_objects, min_distance=20):
    """Get random position for object placement that avoids significant overlap."""
    img_h, img_w = image_shape[:2]
    obj_h, obj_w = obj_shape[:2]
    
    # Maximum number of attempts to find a non-overlapping position
    max_attempts = 50
    
    for _ in range(max_attempts):
        # Random position within image boundaries
        x = random.randint(0, img_w - obj_w)
        y = random.randint(0, img_h - obj_h)
        
        # Check overlap with existing objects
        overlap = False
        for existing_obj in existing_objects:
            ex, ey, ew, eh = existing_obj
            # Calculate distance between centers
            center_dist = np.sqrt(((x + obj_w/2) - (ex + ew/2))**2 + 
                                ((y + obj_h/2) - (ey + eh/2))**2)
            if center_dist < min_distance:
                overlap = True
                break
        
        if not overlap:
            return x, y
    
    # If no non-overlapping position found, return a random position
    return random.randint(0, img_w - obj_w), random.randint(0, img_h - obj_h)

def get_scaled_object(image, mask, bbox, target_size=(100, 100)):
    """Extract and scale object from image using mask and bbox."""
    x, y, w, h = [int(v) for v in bbox]
    
    # Extract object using mask
    object_img = image.copy()
    object_img[mask == 0] = 0  # Zero out non-mask areas
    
    # Crop to bbox
    object_img = object_img[y:y+h, x:x+w]
    
    # Scale to target size while maintaining aspect ratio
    h_scale = target_size[0] / h
    w_scale = target_size[1] / w
    scale = min(h_scale, w_scale)  # Use smaller scale to maintain aspect ratio
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    object_img = cv2.resize(object_img, (new_w, new_h))
    
    return object_img, (new_h, new_w)

def overlay_object(image, object_img, mask, position, alpha=0.7):
    """Overlay scaled object onto image at specified position."""
    x, y = position
    obj_h, obj_w = object_img.shape[:2]
    
    # Create a region of interest (ROI) in the original image
    roi = image[y:y+obj_h, x:x+obj_w]
    
    # Resize mask to match object size
    mask_resized = cv2.resize(mask, (obj_w, obj_h))
    mask_resized = mask_resized > 0
    
    # Create a copy of the ROI
    roi_copy = roi.copy()
    
    # Overlay object onto ROI using mask
    roi_copy[mask_resized] = object_img[mask_resized]
    
    # Blend with original image
    image[y:y+obj_h, x:x+obj_w] = cv2.addWeighted(roi, 1-alpha, roi_copy, alpha, 0)
    
    return image

def main():
    # COCO dataset paths
    data_dir = "/ps/project/datasets/COCO"
    ann_file = os.path.join(data_dir, "annotations/instances_train2014.json")
    img_dir = os.path.join(data_dir, "train2014")
    
    # Initialize COCO api
    coco = COCO(ann_file)
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    
    # Randomly select an image
    img_id = random.choice(img_ids)
    img_info = coco.loadImgs(img_id)[0]
    
    # Load the image
    img_path = os.path.join(img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not load image: {img_path}")
        return
    
    # Get annotations for this image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    # Randomly select 3-5 objects
    num_objects = random.randint(3, 5)
    selected_anns = random.sample(anns, min(num_objects, len(anns)))
    
    # Create a copy of the image for overlay
    overlay_image = image.copy()
    
    # Keep track of placed objects for overlap checking
    placed_objects = []
    
    # Process each object
    for ann in selected_anns:
        if 'segmentation' in ann and 'bbox' in ann:
            # Convert polygon to mask
            if isinstance(ann['segmentation'], list):
                # Polygon format
                mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                for seg in ann['segmentation']:
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [points], 1)
            else:
                # RLE format
                mask = coco_mask.decode(ann['segmentation'])
            
            # Get scaled object
            object_img, obj_shape = get_scaled_object(image, mask, ann['bbox'])
            
            # Get random position
            position = get_random_position(image.shape, obj_shape, placed_objects)
            
            # Overlay object
            overlay_image = overlay_object(overlay_image, object_img, mask, position)
            
            # Add to placed objects list
            x, y = position
            placed_objects.append((x, y, obj_shape[1], obj_shape[0]))
    
    # Save the results
    output_dir = "overlay_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overlay
    output_path = os.path.join(output_dir, f"object_overlay_{img_info['file_name']}")
    cv2.imwrite(output_path, overlay_image)
    print(f"Saved overlay to: {output_path}")

if __name__ == "__main__":
    main() 