from .base import AIVisionComponent
import os
import cv2
import numpy as np
import shutil
from .darknet_auto_labeler import XML_BODY_1, XML_OBJECT, XML_BODY_2

class TensorFlowAutoLabeler(AIVisionComponent):
    """Auto-labels images using a TensorFlow Object Detection API frozen graph."""
    
    def setup(self, config):
        import tensorflow as tf
        import sys
        
        # Ensure Object Detection API is accessible
        sys.path.append("..") 
        from utils import label_map_util
        
        self.model_name = config.get('model_name', 'fine_tuned_model')
        self.num_classes = config.get('num_classes', 1)
        self.thresh = config.get('thresh', 0.2)
        
        self.folder_name = config.get('folder_name', 'images')
        self.labeled_dir = os.path.join(self.folder_name, config.get('labeled_dir', 'labeled'))
        self.unlabeled_dir = os.path.join(self.folder_name, config.get('unlabeled_dir', 'unlabeled'))
        
        os.makedirs(self.labeled_dir, exist_ok=True)
        os.makedirs(self.unlabeled_dir, exist_ok=True)

        cwd_path = os.getcwd()
        path_to_ckpt = os.path.join(cwd_path, self.model_name, 'frozen_inference_graph.pb')
        path_to_labels = os.path.join(cwd_path, self.model_name, 'labelmap.pbtxt')

        # Load Label Map
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        # Load TF Graph
        # Suppress deprecation warnings for TF 1.x logic
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # Tensors
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.is_initialized = True
        print(f"[{self.__class__.__name__}] TensorFlow Model Loaded Successfully.")

    def _execute(self, data, config):
        """Processes a SINGLE image path (Pipeline handles batching)."""
        from utils import visualization_utils as vis_util

        image_path = data
        if not os.path.exists(image_path):
            return {"file": image_path, "status": "error"}

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)
        imH, imW, _ = image_rgb.shape

        # Run Inference
        (boxes, scores, classes) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes],
            feed_dict={self.image_tensor: image_expanded})

        # Visualization
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=self.thresh)

        cv2.putText(image, 'Label good? (y/n)', (30, 50), self.font, 1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, 'Label good? (y/n)', (30, 50), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Label attempt', image)

        # Handle User Input
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration("User triggered pipeline exit.")
            
        save_labels = (key == ord('y'))

        if save_labels:
            bboxdata = []
            for i in range(len(scores[0])):
                if scores[0][i] > self.thresh:
                    ymin = int(boxes[0][i][0] * imH)
                    xmin = int(boxes[0][i][1] * imW)
                    ymax = int(boxes[0][i][2] * imH)
                    xmax = int(boxes[0][i][3] * imW)
                    
                    # Look up class name from category index
                    class_id = int(classes[0][i])
                    label_name = self.category_index[class_id]['name'] if class_id in self.category_index else 'unknown'
                    
                    coords = [xmin, ymin, xmax, ymax] # TF format corrected to Pascal VOC standard
                    bboxdata.append([label_name, coords])

            self._create_xml(image_path, bboxdata, imH, imW)
            dest_dir = self.labeled_dir
        else:
            dest_dir = self.unlabeled_dir

        new_image_path = os.path.join(dest_dir, os.path.basename(image_path))
        shutil.move(image_path, new_image_path)

        return {"file": new_image_path, "status": "labeled" if save_labels else "rejected"}

    def _create_xml(self, im_path, im_bbs, imH, imW):
        imFn = os.path.basename(im_path)
        xmlFn = os.path.splitext(imFn)[0] + '.xml'
        xmlPath = os.path.join(self.labeled_dir, xmlFn)
        
        with open(xmlPath, 'w') as f:
            f.write(XML_BODY_1.format(FOLDER=self.folder_name, FILENAME=imFn, PATH=im_path, WIDTH=imW, HEIGHT=imH))
            for bbox in im_bbs:
                f.write(XML_OBJECT.format(CLASS=bbox[0], XMIN=bbox[1][0], YMIN=bbox[1][1], XMAX=bbox[1][2], YMAX=bbox[1][3]))
            f.write(XML_BODY_2)

    def cleanup(self):
        """Closes the TensorFlow session when done."""
        if hasattr(self, 'sess'):
            self.sess.close()
