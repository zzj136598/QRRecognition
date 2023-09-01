import os
import cv2
import numpy as np
import threading
from pond import Pond, PooledObjectFactory, PooledObject


class wechat_qrcode_WeChatQRCode_wrapper:
    validate_result: bool = True

    def __init__(self, model_dir="./models"):
        try:
            self.detector = cv2.wechat_qrcode_WeChatQRCode(
                os.path.join(model_dir, "detect.prototxt"),
                os.path.join(model_dir, "detect.caffemodel"),
                os.path.join(model_dir, "sr.prototxt"),
                os.path.join(model_dir, "sr.caffemodel"),
            )
        except Exception as e:
            raise Exception("load model failed: %s" % str(e))


class ModelFactory(PooledObjectFactory):
    _model_dir = None

    def set_model_dir(self, model_dir):
        self._model_dir = model_dir

    def createInstance(self) -> PooledObject:
        detector_wrapper = wechat_qrcode_WeChatQRCode_wrapper(self._model_dir)
        return PooledObject(detector_wrapper)

    def reset(self, pooled_object: PooledObject) -> PooledObject:
        return pooled_object

    def validate(self, pooled_object: PooledObject) -> bool:
        return pooled_object.keeped_object.validate_result

    def destroy(self, pooled_object: PooledObject):
        del pooled_object


class QRRecognition:
    '''
    基于WeChatCV的二维码检测识别解码
    https://github.com/WeChatCV
    单例模式，内部支持pool,可以自动创建和回收识别器
    '''
    __instance = None
    __lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        if cls.__instance: return cls.__instance
        with cls.__lock:
            if not cls.__instance: cls.__instance = super().__new__(cls)
            return cls.__instance

    def __init__(self, model_dir="./models", pool=True, pooled_maxsize=5):
        assert os.path.exists(os.path.join(model_dir, "detect.prototxt"))
        assert os.path.exists(os.path.join(model_dir, "detect.caffemodel"))
        assert os.path.exists(os.path.join(model_dir, "sr.prototxt"))
        assert os.path.exists(os.path.join(model_dir, "sr.caffemodel"))
        self.pool = pool
        if pool:
            self.pond = Pond(borrowed_timeout=2,
                             time_between_eviction_runs=-1,
                             thread_daemon=True,
                             eviction_weight=0.8)
            self.factory = ModelFactory(pooled_maxsize=pooled_maxsize, least_one=False)
            self.factory.set_model_dir(model_dir)
            self.pond.register(self.factory)
        try:
            self.detector = wechat_qrcode_WeChatQRCode_wrapper(model_dir).detector
        except Exception as e:
            raise Exception("load model failed: %s" % str(e))

    def detect_from_np(self, image):
        assert (type(image) == np.ndarray or image is not None)
        if self.pool:
            pool_obj: PooledObject = self.pond.borrow(self.factory)
            detector_wrapper = pool_obj.use()
            res, points = detector_wrapper.detector.detectAndDecode(image)
            self.pond.recycle(pool_obj, self.factory)
        else:
            res, points = self.detector.detectAndDecode(image)
        if res is None or len(res) < 1 or (len(res) == 1 and not res[0]): return ((), ())
        return (res, points)

    def detect_from_file(self, image):
        assert (type(image) == str or image is not None)
        image = cv2.imread(image)
        return self.detect_from_np(image)

    def url_to_uniscid(self, s: str):
        return s.split("=")[1]

    def get_unisc_id_from_image_file(self, image_file):
        assert os.path.exists(image_file)
        return [self.url_to_uniscid(s) for s in self.detect_from_file(image_file)[0]]


if __name__ == "__main__":
    qr_detect = QRRecognition(model_dir="models")
    r = qr_detect.get_unisc_id_from_image_file("test.jpg")
    print(r)
