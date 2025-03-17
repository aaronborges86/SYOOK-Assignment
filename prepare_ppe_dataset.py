import os
import cv2
from shutil import copyfile

def prepare_ppe_data(images_dir, labels_dir, output_dir):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    for img_file in os.listdir(images_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        if not os.path.exists(label_path):
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            cls_id = int(parts[0])
            if cls_id != 0:  # Skip non-person classes
                continue
            
            # Get bounding box coordinates
            x_center, y_center, width, height = map(float, parts[1:5])
            img_h, img_w = image.shape[:2]
            
            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - width / 2) * img_w)
            y1 = int((y_center - height / 2) * img_h)
            x2 = int((x_center + width / 2) * img_w)
            y2 = int((y_center + height / 2) * img_h)
            
            # Crop the person
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # Save cropped image
            crop_name = f"{base_name}_person{line_idx}.jpg"
            cv2.imwrite(os.path.join(output_dir, 'images', crop_name), crop)
            
            # Save corresponding labels (only PPE classes)
            ppe_labels = []
            for ppe_line in lines:
                ppe_parts = ppe_line.strip().split()
                ppe_cls_id = int(ppe_parts[0])
                if ppe_cls_id == 0:  # Skip person class
                    continue
                
                # Convert to relative coordinates in the cropped image
                ppe_x_center, ppe_y_center, ppe_width, ppe_height = map(float, ppe_parts[1:5])
                ppe_x1 = (ppe_x_center - ppe_width / 2) * img_w
                ppe_y1 = (ppe_y_center - ppe_height / 2) * img_h
                ppe_x2 = (ppe_x_center + ppe_width / 2) * img_w
                ppe_y2 = (ppe_y_center + ppe_height / 2) * img_h
                
                # Check if the PPE item is within the person's bounding box
                if ppe_x1 >= x1 and ppe_x2 <= x2 and ppe_y1 >= y1 and ppe_y2 <= y2:
                    # Convert to relative coordinates in the cropped image
                    rel_x_center = (ppe_x_center - x1 / img_w) / (x2 - x1) * img_w
                    rel_y_center = (ppe_y_center - y1 / img_h) / (y2 - y1) * img_h
                    rel_width = ppe_width / (x2 - x1) * img_w
                    rel_height = ppe_height / (y2 - y1) * img_h
                    
                    ppe_labels.append(f"{ppe_cls_id} {rel_x_center} {rel_y_center} {rel_width} {rel_height}")
            
            if ppe_labels:
                with open(os.path.join(output_dir, 'labels', f"{crop_name[:-4]}.txt"), 'w') as f:
                    f.write('\n'.join(ppe_labels))

# Usage
prepare_ppe_data(
    images_dir=r'C:\Users\91797\Desktop\Aaron Borges 21btrca003 syook assignment\datasets\images',
    labels_dir=r'C:\Users\91797\Desktop\Aaron Borges 21btrca003 syook assignment\datasets\labels',
    output_dir='C:/Users/91797/Downloads/ppe_dataset'
)