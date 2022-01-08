import argparse
from PIL import Image
from tracking import load_feature_extractor
from tracking.sort.tracker import DeepSORTTracker
from tracking.utils import *
import sys
sys.path.insert(0, 'yolov5')
from yolov5.models.experimental import attempt_load
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import app
import visualization





objectClass = []



class Tracking:
    def __init__(self, 
        yolo_model, \
        reid_model,
        img_size=640,
        filter_class=None,
        conf_thres=0.25,
        iou_thres=0.45,
        max_cosine_dist=0.4,    # the higher the value, the easier it is to assume it is the same person
        max_iou_dist=0.7,       # how much bboxes should overlap to determine the identity of the unassigned track
        nn_budget=None,         # indicates how many previous frames of features vectors should be retained for distance calc for ecah track
        max_age=60,             # specifies after how many frames unallocated tracks will be deleted
        n_init=3                # specifies after how many frames newly allocated tracks will be activated
    ) -> None:
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.filter_class = filter_class

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = attempt_load(yolo_model, map_location=self.device)
        self.model = self.model.to(self.device)
        self.names = self.model.names

        self.patch_model, self.patch_transform = load_feature_extractor(reid_model, self.device)
        self.tracker = DeepSORTTracker('cosine', max_cosine_dist, nn_budget, max_iou_dist, max_age, n_init)


    def preprocess(self, image):
        img = letterbox(image, new_shape=self.img_size)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img[None]
        return img


    def extract_features(self, boxes, img):
        image_patches = []
        for xyxy in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            img_patch = Image.fromarray(img[y1:y2, x1:x2])
            img_patch = self.patch_transform(img_patch)
            image_patches.append(img_patch)

        image_patches = torch.stack(image_patches).to(self.device)
        features = self.patch_model.encode_image(image_patches).cpu().numpy()
        return features

    def postprocess(self, pred, img1, img0):

        global objectClass
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.filter_class)
        for det in pred:
            if len(det):
                boxes = scale_boxes(det[:, :4], img0.shape[:2], img1.shape[-2:]).cpu()
                features = self.extract_features(boxes, img0)
                self.tracker.predict()
                self.tracker.update(boxes, det[:, 5], features)

                for track in self.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1: continue
                    label = f"{self.names[int(track.class_id)]} #{track.track_id}"
                    objectClass.append(self.names[int(track.class_id)]+str(track.track_id))
                    plot_one_box(track.to_tlbr(), img0, color=colors(int(track.class_id)), label=label)

            else:
                self.tracker.increment_ages()

   

        
    @torch.no_grad()
    def predict(self, image):
        img = self.preprocess(image)
        pred = self.model(img)[0]  
        self.postprocess(pred, img, image)
        return image

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--yolo-model', type=str, default='checkpoints/yolov5s.pt')
    parser.add_argument('--reid-model', type=str, default='CLIP-RN50')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--filter-class', nargs='+', type=int, default=None)
    parser.add_argument('--conf-thres', type=float, default=0.4)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--max-cosine-dist', type=float, default=0.2)
    parser.add_argument('--max-iou-dist', type=int, default=0.7)
    parser.add_argument('--nn-budget', type=int, default=100)
    parser.add_argument('--max-age', type=int, default=70)
    parser.add_argument('--n-init', type=int, default=3)
    return parser.parse_args()



def objectPerSecond(path,className):
    seconds = videoDuration(path)
    object = countClasses()
    nbObject= object[className]
    if seconds == 0:
        return 0
    else:
        objectPerSecond = nbObject / seconds
        return objectPerSecond





def personPerMinute(path):
    minutes = int(videoDuration(path)/60)
    person = countClasses()
    nbPerson = person['person']
    if minutes == 0:
        return 0
    else:
        personPerMinute = int(nbPerson / minutes)
        return personPerMinute


def personPerHours(path):
    minutes = int(videoDuration(path)/60)
    hours = minutes/60
    person = countClasses()
    nbPerson = person['person']
    if hours == 0:
        return 0
    else:
        personPerHours = int(nbPerson / hours)
        return personPerHours





def videoDuration(path):
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT);
    seconds = float(frames) / float(fps)
    return seconds








def countClasses():
    global objectClass
    dic = {}
    allClass = list(set(objectClass))
    data2 = []
    for x in allClass:
        result = ''.join([i for i in x if not i.isdigit()])
        data2.append(result)
    for i in data2:
        dic.update({i:data2.count(i)})
    return dic








def numberOfObjects():
    global objectClass
    result = {}
    nb = []
    allClass = list(set(objectClass))
    data2 = []
    for x in allClass:
        result = ''.join([i for i in x if not i.isdigit()])
        data2.append(result)
    data3 = data2
    data2 = list(dict.fromkeys(data2))
    for i in data2:
        nb.append(data3.count(i))

    result = {"OBJECTS": data2, "NUMBERS": nb}

    return result




def getUniqueObject():
    global objectClass
    allClass = list(set(objectClass))
    data2 = []
    for x in allClass:
        result = ''.join([i for i in x if not i.isdigit()])
        data2.append(result)
    data2 = list(dict.fromkeys(data2))
    return data2






def numberOfObjectsPerSecond(path):
    result = {}
    nbObject = []
    names = getUniqueObject()
    for i in names:
        nbObject.append(objectPerSecond(path, i))
    result.update({"OBJECTS": names})
    result.update({"NUMBERS": nbObject})
    return result




def totalNumberExceptClassName(className):
    object = countClasses()
    total = 0
    for i in object:
        if i != className:
            total += object[i]
    return total






def percentageOfObjects():
    result = {}
    object = countClasses()
    percentage = []
    names = getUniqueObject()
    for i in names:
        percentage.append(object[i]/totalNumberExceptClassName(i))
    result.update({"OBJECTS": names})
    result.update({"PERCENTAGE": percentage})
    return result














if __name__ == '__main__':

    path, var = app.main()
    args = argument_parser()
    tracking = Tracking(
        "checkpoints/yolov5s.pt", 
        "CLIP-RN50",
        args.img_size, 
        var,
        args.conf_thres, 
        args.iou_thres, 
        args.max_cosine_dist,  
        args.max_iou_dist,
        args.nn_budget,
        args.max_age,
        args.n_init
    )

    reader = VideoReader(path)
    writer = VideoWriter(f"{args.source.rsplit('.', maxsplit=1)[0]}_out.mp4", reader.fps)
    fps = FPS(len(reader.frames))
    for frame in tqdm(reader):
        fps.start()
        output = tracking.predict(frame.numpy())
        fps.stop(False)
        writer.update(output)

    print(f"FPS: {fps.fps}")
    writer.write()

    count = countClasses()
    nb = 0
    for i in count:
        nb += count[i]

    duration = "Video Duration: " + str(videoDuration(path)) + "s"
    totalNumber = "Total Number Of Objects: "+str(nb)

    visualization.count = numberOfObjects()
    visualization.perSecond = numberOfObjectsPerSecond(path)
    visualization.percentage = percentageOfObjects()
    visualization.duration = duration
    visualization.totalNumber = totalNumber
    visualization.runViz()











