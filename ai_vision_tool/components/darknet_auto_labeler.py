import os
import cv2
import numpy as np
import glob
import shutil
from .base import AIVisionComponent

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


class DarknetAutoLabeler(AIVisionComponent):
    """Auto-labels images using a pre-trained Darknet/YOLO model."""

    def setup(self, config):
        """Loads the Darknet network and prepares output directories.

        Appends the Darknet build directory to ``sys.path`` and imports
        the ``darknet`` module, then loads the specified network weights.

        Args:
            config (dict): Setup parameters. Supports:
                - 'darknet_path' (str): Path to the Darknet x64 build directory.
                - 'weights' (str): Model weights filename. Default is 'yolov4.weights'.
                - 'cfg' (str): Network config filename. Default is 'yolov4.cfg'.
                - 'meta' (str): Metadata file path. Default is 'cfg/coco.data'.
                - 'min_thresh' (float): Detection confidence threshold. Default is 0.5.
                - 'iou_thresh' (float): IoU threshold for NMS. Default is 0.5.
                - 'folder_name' (str): Root image folder name. Default is 'images'.
                - 'labeled_dir' (str): Subdirectory for accepted labels. Default is 'labeled'.
                - 'unlabeled_dir' (str): Subdirectory for rejected images. Default is 'unlabeled'.
        """
        import sys

        self.darknet_path = config.get('darknet_path', 'C:\\darknet\\build\\darknet\\x64')
        self.weights = config.get('weights', 'yolov4.weights')
        self.cfg = config.get('cfg', 'yolov4.cfg')
        self.meta = config.get('meta', 'cfg/coco.data')
        self.min_thresh = config.get('min_thresh', 0.5)
        self.iou_thresh = config.get('iou_thresh', 0.5)

        self.folder_name = config.get('folder_name', 'images')
        self.labeled_dir = os.path.join(self.folder_name, config.get('labeled_dir', 'labeled'))
        self.unlabeled_dir = os.path.join(self.folder_name, config.get('unlabeled_dir', 'unlabeled'))

        os.makedirs(self.labeled_dir, exist_ok=True)
        os.makedirs(self.unlabeled_dir, exist_ok=True)

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
        """Runs inference on a single image and prompts the user to accept or reject labels.

        Displays the image with predicted bounding boxes. Press ``y`` to save
        Pascal VOC XML labels and move the image to the labeled directory, or
        any other key to move it to the unlabeled directory. Press ``q`` to
        abort the pipeline.

        Args:
            data (str): Absolute or relative path to the image file.
            config (dict): Unused at execute time; parameters are read during setup.

        Returns:
            dict: Result payload with:
                - 'file' (str): Destination path of the moved image.
                - 'status' (str): 'labeled', 'rejected', or 'error - file not found'.

        Raises:
            StopIteration: When the user presses ``q`` to exit the pipeline.
        """
        image_path = data
        if not os.path.exists(image_path):
            return {"file": image_path, "status": "error - file not found"}

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.darknet_width, self.darknet_height), interpolation=cv2.INTER_LINEAR)

        darknet_image = darknet.make_image(self.darknet_width, self.darknet_height, 3)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

        detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.min_thresh)
        darknet.free_image(darknet_image)

        frame_h, frame_w = image.shape[0:2]
        detections_true = []
        for label, confidence, bbox in detections:
            x, y, w, h = bbox
            x_true = int((x / self.darknet_width) * frame_w)
            y_true = int((y / self.darknet_height) * frame_h)
            w_true = int((w / self.darknet_width) * frame_w)
            h_true = int((h / self.darknet_height) * frame_h)
            detections_true.append([str(label), confidence, (x_true, y_true, w_true, h_true)])

        detections_true = self._filter_iou(detections_true, self.iou_thresh)

        image_results = np.copy(image)
        for detection in detections_true:
            x, y, w, h = detection[2]
            image_results = self._draw_pred(image_results, detection[0], float(detection[1]), x, y, w, h)

        cv2.putText(image_results, 'Label good? (y/n)', (30, 50), self.font, 1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image_results, 'Label good? (y/n)', (30, 50), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Label attempt', image_results)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration("User triggered pipeline exit.")

        save_labels = (key == ord('y'))

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

    def _draw_pred(self, draw_frame, classId, conf, x, y, w, h):
        """Draws a labeled bounding box on the frame.

        Args:
            draw_frame (numpy.ndarray): Image to annotate (modified in place).
            classId (str): Detected class name.
            conf (float): Detection confidence score (0–100).
            x (int): Bounding box center x in original image coordinates.
            y (int): Bounding box center y in original image coordinates.
            w (int): Bounding box width in original image coordinates.
            h (int): Bounding box height in original image coordinates.

        Returns:
            numpy.ndarray: Annotated frame.
        """
        left, top, right, bottom = self._get_min_max([x, y, w, h])
        cv2.rectangle(draw_frame, (left, top), (right, bottom), (10, 255, 0), 3)
        label = f'{classId}: {int(conf)}%'
        labelSize, baseLine = cv2.getTextSize(label, self.font, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(draw_frame, (left, top - labelSize[1] - 12), (left + labelSize[0] + 40, top + baseLine - 8), (255, 255, 255), cv2.FILLED)
        cv2.putText(draw_frame, label, (left, top - 7), self.font, .7, (0, 0, 0), 2)
        return draw_frame

    def _get_min_max(self, coords_wh):
        """Converts center-format (x, y, w, h) to corner-format (xmin, ymin, xmax, ymax).

        Args:
            coords_wh (list[int]): Bounding box as [x_center, y_center, width, height].

        Returns:
            list[int]: Bounding box as [xmin, ymin, xmax, ymax].
        """
        x, y, w, h = coords_wh
        return [int(round(x - (w / 2))), int(round(y - (h / 2))), int(round(x + (w / 2))), int(round(y + (h / 2)))]

    def _filter_iou(self, detections, iou_threshold):
        """Removes overlapping detections using non-maximum suppression.

        Detections are sorted by confidence descending. Any detection whose IoU
        with a higher-confidence detection exceeds ``iou_threshold`` is suppressed.

        Args:
            detections (list): List of [label, confidence, (x, y, w, h)] entries.
            iou_threshold (float): IoU overlap threshold above which a box is suppressed.

        Returns:
            list: Filtered detections with suppressed entries removed.
        """
        detections = sorted(detections, key=lambda obj: float(obj[1]), reverse=True)
        for i in range(len(detections)):
            if detections[i][1] == 0:
                continue
            for j in range(i + 1, len(detections)):
                if self._calculate_iou(detections[i], detections[j]) > iou_threshold:
                    detections[j][1] = 0
        return [det for det in detections if float(det[1]) > 0]

    def _calculate_iou(self, box_1, box_2):
        """Computes the Intersection over Union (IoU) of two bounding boxes.

        Args:
            box_1 (list): Detection entry [label, confidence, (x, y, w, h)].
            box_2 (list): Detection entry [label, confidence, (x, y, w, h)].

        Returns:
            float: IoU score in [0, 1]. Returns 0 if there is no overlap or
                if the union area is zero.
        """
        b1_xmin, b1_ymin, b1_xmax, b1_ymax = self._get_min_max(box_1[2])
        b2_xmin, b2_ymin, b2_xmax, b2_ymax = self._get_min_max(box_2[2])
        w_overlap = min(b1_xmax, b2_xmax) - max(b1_xmin, b2_xmin)
        h_overlap = min(b1_ymax, b2_ymax) - max(b1_ymin, b2_ymin)
        if w_overlap < 0 or h_overlap < 0:
            return 0
        overlap_area = w_overlap * h_overlap
        union_area = ((b1_ymax - b1_ymin) * (b1_xmax - b1_xmin)) + ((b2_ymax - b2_ymin) * (b2_xmax - b2_xmin)) - overlap_area
        return overlap_area / union_area if union_area > 0 else 0

    def _create_xml(self, im_path, im_bbs, imH, imW):
        """Writes a Pascal VOC XML annotation file for the given image.

        Args:
            im_path (str): Path to the source image file.
            im_bbs (list): List of [class_name, [xmin, ymin, xmax, ymax]] entries.
            imH (int): Image height in pixels.
            imW (int): Image width in pixels.
        """
        imFn = os.path.basename(im_path)
        xmlFn = os.path.splitext(imFn)[0] + '.xml'
        xmlPath = os.path.join(self.labeled_dir, xmlFn)

        with open(xmlPath, 'w') as f:
            f.write(XML_BODY_1.format(FOLDER=self.folder_name, FILENAME=imFn, PATH=im_path, WIDTH=imW, HEIGHT=imH))
            for bbox in im_bbs:
                f.write(XML_OBJECT.format(CLASS=bbox[0], XMIN=bbox[1][0], YMIN=bbox[1][1], XMAX=bbox[1][2], YMAX=bbox[1][3]))
            f.write(XML_BODY_2)
