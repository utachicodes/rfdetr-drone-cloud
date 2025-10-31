import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

MODEL_PATH = "models/jawji-finetune.pt"
DATA_ROOT = Path("data")
IMG_DIR = DATA_ROOT / "images"
VID_DIR = DATA_ROOT / "videos"
OUT_DIR = Path("results")
CONF_THRESH = 0.15
IOU_THRESH = 0.35
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANNOT_JSON = None

OUT_DIR.mkdir(exist_ok=True)

model = YOLO(MODEL_PATH)
model.to(DEVICE)
print(f"Model loaded on {DEVICE}")


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - inter
    return inter / denom if denom > 0 else 0


class SimpleTracker:
    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1

    def update(self, dets):
        assigned = {}
        unmatched_dets = set(range(len(dets)))
        for tid, tr in list(self.tracks.items()):
            best_iou = 0
            best_j = -1
            for j in unmatched_dets:
                val = iou(tr['bbox'], dets[j][:4])
                if val > best_iou:
                    best_iou = val
                    best_j = j
            if best_iou >= self.iou_threshold:
                det = dets[best_j]
                tr['bbox'] = det[:4]
                tr['score'] = det[4]
                tr['age'] = 0
                tr['hits'] += 1
                assigned[best_j] = tid
                unmatched_dets.remove(best_j)
            else:
                tr['age'] += 1
        for j in list(unmatched_dets):
            det = dets[j]
            self.tracks[self.next_id] = {'bbox': det[:4], 'score': det[4], 'age': 0, 'hits': 1}
            assigned[j] = self.next_id
            self.next_id += 1
        to_delete = [tid for tid, tr in self.tracks.items() if tr['age'] > self.max_age]
        for tid in to_delete:
            del self.tracks[tid]
        outputs = []
        for j, tid in assigned.items():
            tr = self.tracks[tid]
            if tr['hits'] >= self.min_hits:
                x1, y1, x2, y2 = map(int, tr['bbox'])
                outputs.append([x1, y1, x2, y2, tr['score'], tid])
        return outputs


def draw_boxes(img, items, class_names):
    img_out = img.copy()
    for it in items:
        x1, y1, x2, y2 = map(int, it['xyxy'])
        conf = it['conf']
        cls_id = it['cls']
        track_id = it.get('id')
        label = f"{class_names[cls_id]} {conf:.2f}"
        if track_id is not None:
            label = f"{class_names[cls_id]} #{track_id} {conf:.2f}"
        color = (0, 255, 0) if "drone" in class_names[cls_id].lower() else (255, 0, 0)
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_out, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img_out


def results_to_items(results):
    items = []
    for r in results:
        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            items.append({'xyxy': xyxy, 'conf': float(box.conf.item()), 'cls': int(box.cls.item())})
    return items


def test_images():
    img_paths = sorted(IMG_DIR.glob("*.*"))[:100]
    class_names = model.names
    for img_path in tqdm(img_paths, desc="Images"):
        img = cv2.imread(str(img_path))
        results = model(img, conf=CONF_THRESH, iou=IOU_THRESH, device=DEVICE)
        items = results_to_items(results)
        img_vis = draw_boxes(img, items, class_names)
        out_path = OUT_DIR / f"vis_{img_path.name}"
        cv2.imwrite(str(out_path), img_vis)


def test_video(track=False):
    vid_paths = list(VID_DIR.glob("*.mp4")) + list(VID_DIR.glob("*.avi"))
    if not vid_paths:
        print("No video files found - skipping video test")
        return
    class_names = model.names
    tracker = SimpleTracker() if track else None
    for vid_path in vid_paths:
        cap = cv2.VideoCapture(str(vid_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_vid = cv2.VideoWriter(str(OUT_DIR / f"out_{vid_path.name}"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Video {vid_path.name}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=CONF_THRESH, iou=IOU_THRESH, device=DEVICE, verbose=False)
            items = results_to_items(results)
            if track and items:
                dets = [item['xyxy'][:4] + [item['conf']] for item in items]
                tracked = tracker.update(dets)
                id_map = {}
                for t in tracked:
                    x1, y1, x2, y2, score, tid = t
                    for item in items:
                        ixy = item['xyxy']
                        if int(ixy[0]) == x1 and int(ixy[1]) == y1 and int(ixy[2]) == x2 and int(ixy[3]) == y2:
                            item['id'] = tid
                            break
            frame_vis = draw_boxes(frame, items, class_names)
            out_vid.write(frame_vis)
            pbar.update(1)
        pbar.close()
        cap.release()
        out_vid.release()


def live_tracking(source=0, track=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open source {source}")
        return
    class_names = model.names
    tracker = SimpleTracker() if track else None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=CONF_THRESH, iou=IOU_THRESH, device=DEVICE, verbose=False)
        items = results_to_items(results)
        if track and items:
            dets = [item['xyxy'][:4] + [item['conf']] for item in items]
            tracked = tracker.update(dets)
            for t in tracked:
                x1, y1, x2, y2, score, tid = t
                for item in items:
                    ixy = item['xyxy']
                    if int(ixy[0]) == x1 and int(ixy[1]) == y1 and int(ixy[2]) == x2 and int(ixy[3]) == y2:
                        item['id'] = tid
                        break
        frame_vis = draw_boxes(frame, items, class_names)
        cv2.imshow("Live", frame_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def compute_map_if_annotations():
    if ANNOT_JSON is None or not Path(ANNOT_JSON).exists():
        return
    from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class
    import json
    with open(ANNOT_JSON) as f:
        gt = json.load(f)
    gt_dict = {}
    for ann in gt["annotations"]:
        img_id = ann["image_id"]
        if img_id not in gt_dict:
            gt_dict[img_id] = []
        gt_dict[img_id].append([ann["category_id"], *ann["bbox"]])
    img_id_to_path = {im["id"]: Path(IMG_DIR) / im["file_name"] for im in gt["images"]}
    preds = []
    targets = []
    for img_id, path in img_id_to_path.items():
        img = cv2.imread(str(path))
        results = model(img, conf=CONF_THRESH, iou=IOU_THRESH, device=DEVICE, verbose=False)
        for r in results:
            for box in r.boxes:
                preds.append([img_id, box.cls.item(), box.conf.item(), *box.xyxy[0].tolist()])
        if img_id in gt_dict:
            for cls, x, y, w, h in gt_dict[img_id]:
                targets.append([img_id, cls, x, y, x + w, y + h])
    if preds:
        pred_tensor = torch.tensor(preds)[:, 1:]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["images", "video", "live", "all"], default="live")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--track", action="store_true")
    args = parser.parse_args()
    if args.mode == "images":
        test_images()
    elif args.mode == "video":
        test_video(track=args.track)
    elif args.mode == "live":
        live_tracking(args.camera, track=args.track)
    elif args.mode == "all":
        test_images()
        test_video(track=args.track)
        compute_map_if_annotations()
    print(f"All done. Visualizations saved in: {OUT_DIR.resolve()}")