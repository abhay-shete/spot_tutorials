import tensorflow as tf
import numpy as np
import cv2


class TensorflowModel:
    """ Wraps a tensorflow model in a way that allows online switching between models."""

    def __init__(self, path, labels_path=None):
        self.path = path

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Make sure we tell tensor flow that this is a different model.
        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        if labels_path is None:
            self.labels = None
        else:
            # Load the class label mappings
            self.labels = open(labels_path).read().strip().split('\n')
            self.labels = {int(L.split(',')[1]): L.split(',')[0] for L in self.labels}

    def predict(self, image):
        """ Predict with this model. """
        with self.detection_graph.as_default():
            # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = self.sess.run([
                self.detection_boxes, self.detection_scores, self.detection_classes,
                self.num_detections
            ], feed_dict={self.image_tensor: image_np_expanded})

            im_height, im_width, _ = image.shape
            boxes_list = [None for i in range(boxes.shape[1])]
            for i in range(boxes.shape[1]):
                boxes_list[i] = (int(boxes[0, i, 0] * im_height), int(boxes[0, i, 1] * im_width),
                                 int(boxes[0, i, 2] * im_height), int(boxes[0, i, 3] * im_width))

            if self.labels is not None:
                labels_out = [self.labels[int(x)] for x in classes[0].tolist()]
            else:
                labels_out = classes[0].tolist()

            return boxes_list, scores[0].tolist(), labels_out, int(num[0])
