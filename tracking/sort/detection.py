import numpy as np


class Detection:
    """Bounding box detection in a single image
    Parameters
    ----------
    tlwh        : (ndarray) Bounding box in format `(x, y, w, h)`.
    confidence  : (float) Detector confidence score.
    feature     : (ndarray) A feature vector that describes the object contained in this image.
    
    Attributes
    ----------
    tlwh        : (ndarray) Bounding box in format `(top left x, top left y, width, height)`.
    confidence  : (ndarray) Detector confidence score.
    class_num   : (ndarray) Detector class.
    feature     : (ndarray) | (None) A feature vector that describes the object contained in this image.
    """

    def __init__(self, tlwh, confidence, class_num, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.class_num = class_num
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e., `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio, height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret