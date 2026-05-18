import os
import cv2
import numpy as np
import glob
import shutil
from .base import AIVisionComponent

# ==========================================
# Shared XML Templates
# ==========================================
XML_BODY_1 = """<annotation>
        <folder>{FOLDER}</folder>
        <filename>{FILENAME}</filename>
        <path>{PATH}</path>
        <source>
                <database>Unknown</database>
        </source>
        <size>
                <width>{WIDTH}</width>
                <height>{HEIGHT}</height>
                <depth>3</depth>
        </size>
"""
XML_OBJECT = """ <object>
                <name>{CLASS}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>{XMIN}</xmin>
                        <ymin>{YMIN}</ymin>
                        <xmax>{XMAX}</xmax>
                        <ymax>{YMAX}</ymax>
                </bndbox>
        </object>
"""
XML_BODY_2 = """</annotation>        
"""

# ==========================================
# 1. Darknet (YOLO) Auto Labeler
# ==========================================
class DarknetAutoLabeler(AIVisionComponent):
    """Auto-labels images using a pre-trained Darknet/YOLO model."""
    
    def setup(self, config):
        import sys
        
        # Extract configurations
        self.darknet_path = config.get('darknet_path', 'C:\\darknet\\build\\darknet\\x64')
        self.weights = config.get('weights', 'yolov4.weights')
        self.cfg = config.get('cfg', 'yolov4.cfg')
        self.meta = config.get('meta', 'cfg/coco.data')
        self.min_thresh = config.get('min_thresh', 0.5)
        self.iou_thresh = config.get('iou_thresh', 0.5)
        
        self.folder_name = config.get('folder_name', 'images')
        self.labeled_dir = os.path.join(self.folder_name, config.get('labeled_dir', 'labeled'))
        self.unlabeled_dir = os.path.join(self.folder_name, config.get('unlabeled_dir', 'unlabeled'))
        
        # Ensure directories exist
        os.makedirs(self.labeled_dir, exist_ok=True)
        os.makedirs(self.unlabeled_dir, exist_ok=True)

        # Import darknet and load network
        sys.path.append(self.darknet_path)
        global darknet
        import darknet
        
        self.network, self.class_names, self.class_colors = darknet.load_network(
            self.cfg, self.meta, self.weights, batch_size=1
        )
        self.darknet_width = darknet.network_width(self.network)
        self.darknet_height = darknet.network_height(self.network)
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.is_initialized = True
        print(f"[{self.__class__.__name__}] Darknet Model Loaded Successfully.")

    def _execute(self, data, config):
        """Processes a SINGLE image path (Pipeline handles batching)."""
        image_path = data
        if not os.path.exists(image_path):
            return {"file": image_path, "status": "error - file not found"}

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.darknet_width, self.darknet_height), interpolation=cv2.INTER_LINEAR)

        # Reformat image for Darknet
        darknet_image = darknet.make_image(self.darknet_width, self.darknet_height, 3)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

        # Perform Inference
        detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.min_thresh)
        darknet.free_image(darknet_image)

        # Post-process Coordinates
        frame_h, frame_w = image.shape[0:2]
        detections_true = []
        for label, confidence, bbox in detections:
            x, y, w, h = bbox
            x_true = int((x / self.darknet_width) * frame_w)
            y_true = int((y / self.darknet_height) * frame_h)
            w_true = int((w / self.darknet_width) * frame_w)
            h_true = int((h / self.darknet_height) * frame_h)
            detections_true.append([str(label), confidence, (x_true, y_true, w_true, h_true)])

        # Filter overlapping boxes
        detections_true = self._filter_iou(detections_true, self.iou_thresh)

        # Draw UI
        image_results = np.copy(image)
        for detection in detections_true:
            x, y, w, h = detection[2]
            image_results = self._draw_pred(image_results, detection[0], float(detection[1]), x, y, w, h)

        cv2.putText(image_results, 'Label good? (y/n)', (30, 50), self.font, 1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image_results, 'Label good? (y/n)', (30, 50), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Label attempt', image_results)

        # Handle User Input
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration("User triggered pipeline exit.")
        
        save_labels = (key == ord('y'))
        
        # Save or Reject
        if save_labels:
            bboxdata = []
            for detection in detections_true:
                x, y, w, h = detection[2]
                coords = self._get_min_max([x, y, w, h])
                bboxdata.append([detection[0], coords])
            
            self._create_xml(image_path, bboxdata, frame_h, frame_w)
            dest_dir = self.labeled_dir
        else:
            dest_dir = self.unlabeled_dir

        new_image_path = os.path.join(dest_dir, os.path.basename(image_path))
        shutil.move(image_path, new_image_path)

        return {"file": new_image_path, "status": "labeled" if save_labels else "rejected"}

    # --- Internal Darknet Helper Methods ---
    def _draw_pred(self, draw_frame, classId, conf, x, y, w, h):
        left, top, right, bottom = self._get_min_max([x, y, w, h])
        cv2.rectangle(draw_frame, (left, top), (right, bottom), (10, 255, 0), 3)
        label = f'{classId}: {int(conf)}%'
        labelSize, baseLine = cv2.getTextSize(label, self.font, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(draw_frame, (left, top - labelSize[1] - 12), (left + labelSize[0] + 40, top + baseLine - 8), (255, 255, 255), cv2.FILLED)
        cv2.putText(draw_frame, label, (left, top - 7), self.font, .7, (0, 0, 0), 2)
        return draw_frame

    def _get_min_max(self, coords_wh):
        x, y, w, h = coords_wh
        return [int(round(x - (w / 2))), int(round(y - (h / 2))), int(round(x + (w / 2))), int(round(y + (h / 2)))]

    def _filter_iou(self, detections, iou_threshold):
        # Implementation of IOU filter mapping directly to original logic
        detections = sorted(detections, key=lambda obj: float(obj[1]), reverse=True)
        for i in range(len(detections)):
            if detections[i][1] == 0: continue
            for j in range(i + 1, len(detections)):
                if self._calculate_iou(detections[i], detections[j]) > iou_threshold:
                    detections[j][1] = 0
        return [det for det in detections if float(det[1]) > 0]

    def _calculate_iou(self, box_1, box_2):
        b1_xmin, b1_ymin, b1_xmax, b1_ymax = self._get_min_max(box_1[2])
        b2_xmin, b2_ymin, b2_xmax, b2_ymax = self._get_min_max(box_2[2])
        w_overlap = min(b1_xmax, b2_xmax) - max(b1_xmin, b2_xmin)
        h_overlap = min(b1_ymax, b2_ymax) - max(b1_ymin, b2_ymin)
        if w_overlap < 0 or h_overlap < 0: return 0
        overlap_area = w_overlap * h_overlap
        union_area = ((b1_ymax - b1_ymin) * (b1_xmax - b1_xmin)) + ((b2_ymax - b2_ymin) * (b2_xmax - b2_xmin)) - overlap_area
        return overlap_area / union_area if union_area > 0 else 0

    def _create_xml(self, im_path, im_bbs, imH, imW):
        imFn = os.path.basename(im_path)
        xmlFn = os.path.splitext(imFn)[0] + '.xml'
        xmlPath = os.path.join(self.labeled_dir, xmlFn)
        
        with open(xmlPath, 'w') as f:
            f.write(XML_BODY_1.format(FOLDER=self.folder_name, FILENAME=imFn, PATH=im_path, WIDTH=imW, HEIGHT=imH))
            for bbox in im_bbs:
                f.write(XML_OBJECT.format(CLASS=bbox[0], XMIN=bbox[1][0], YMIN=bbox[1][1], XMAX=bbox[1][2], YMAX=bbox[1][3]))
            f.write(XML_BODY_2)