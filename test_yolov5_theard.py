import torch
import threading
import cv2
import numpy as np

weights_yolov5 = "/home/aitraining/workspace/datnh14/Optimal/yolov5/runs/train/exp/weights/best.pt"
# weights_yolov5 = "/home/aitraining/workspace/thaonp27/Yolov5_StrongSORT_OSNet/yolov5s.engine"
device = "cuda:0"

yolov5 = torch.hub.load('/home/aitraining/workspace/datnh14/Optimal/yolov5', 'custom', path=weights_yolov5, source='local', device=device)  # local repo
print(f'threads: {torch.get_num_threads()} {torch.get_num_interop_threads()}')

class RunModel(threading.Thread):
    def __init__(self, path_rtsp, name):
        super().__init__()
        self.rtsp = path_rtsp
        self.name = name


    def run(self):
        frame_count = 0
        cap = cv2.VideoCapture(self.rtsp)

        while True:
            # try:
                timer = cv2.getTickCount()
                ret, frame = cap.read()
                if not ret:
                    cap = cv2.VideoCapture(self.rtsp)
                    continue

                print("---------------name video-------------", self.name)
                print("---------------frame_count-------------", frame_count)

                # Draw line
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Inference
                boxes = yolov5(frame_rgb).pandas().xyxy[0].sort_values('ymin')
                if boxes.empty:
                    print('No Lisence Plate!')
                else:
                    boxes = np.asarray(boxes)

                FPS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                print("FPS:", round(FPS))
                frame_count += 1
            # except:
            #     print("Error")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    runModel = RunModel("/home/aitraining/workspace/datnh14/Optimal/yolov5/Datasets/congtruong.mp4", 0)
    runModel1 = RunModel("/home/aitraining/workspace/datnh14/Optimal/yolov5/Datasets/congtruong.mp4", 1)
    runModel2 = RunModel("/home/aitraining/workspace/datnh14/Optimal/yolov5/Datasets/congtruong.mp4", 2)
    runModel3 = RunModel("/home/aitraining/workspace/datnh14/Optimal/yolov5/Datasets/congtruong.mp4", 3)
    runModel4 = RunModel("/home/aitraining/workspace/datnh14/Optimal/yolov5/Datasets/congtruong.mp4", 4)
    runModel5 = RunModel("/home/aitraining/workspace/datnh14/Optimal/yolov5/Datasets/congtruong.mp4", 5)
    runModel6 = RunModel("/home/aitraining/workspace/datnh14/Optimal/yolov5/Datasets/congtruong.mp4", 6)
    runModel7 = RunModel("/home/aitraining/workspace/datnh14/Optimal/yolov5/Datasets/congtruong.mp4", 7)
    runModel8 = RunModel("/home/aitraining/workspace/datnh14/Optimal/yolov5/Datasets/congtruong.mp4", 8)
    runModel.start()
    runModel1.start()
    runModel2.start()
    runModel3.start()
    runModel4.start()
    runModel5.start()
    runModel6.start()
    runModel7.start()
    runModel8.start()