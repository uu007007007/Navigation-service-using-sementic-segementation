import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from shapely.geometry import Polygon
import random

# CUDA 설정 최적화
torch.backends.cudnn.benchmark = True  # GPU에서 최적의 커널 선택

# COCO 데이터셋 등록
from detectron2.data.datasets import register_coco_instances
register_coco_instances("val_dataset", {}, '/home/road2022/parking_project/detectron2/val.json', '/home/road2022/parking_project/detectron2/image')
test_metadata = MetadataCatalog.get("val_dataset")
train_dicts = DatasetCatalog.get("val_dataset")

# Config 설정
cfg = get_cfg()
cfg.merge_from_file("/home/road2022/parking_project/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "/home/road2022/parking_project/detectron2/model_final.pth"  # 모델 경로
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
cfg.MODEL.DEVICE = "cuda"  # GPU 사용

#predictor 정의
predictor = DefaultPredictor(cfg)

# 클래스별 고정 색상 정의
class_colors = {
    "Car": [255, 0, 0],
    "Van": [0, 255, 0],
    "Motorbike": [0, 0, 255],
    "Other Vehicle": [255, 255, 0],
    "Movable Obstacle": [255, 0, 255],
    "Traffic Light": [0, 255, 255],
    "Traffic Cone": [128, 128, 0],
    "Traffic Pole": [128, 0, 128],
    "Parking Block": [0, 128, 128],
    "Parking Sign": [64, 64, 64],
    "No Parking Stand": [192, 192, 192],
    "Electric Car Charger": [255, 165, 0],
    "Electric Car Parking Space": [0, 255, 127],
    "Driveable Space": [123, 104, 238],
    "Disabled Parking Space": [255, 182, 193],
    "Parking Space": [0, 191, 255],
    "Human": [165, 42, 42],
	"Recommend" : [0,0,255]
}

#inference 할 영상 파일
input_video_path = "/home/road2022/parking_project/detectron2/night.mp4"
output_video_path = 'night_test.mov'



class parking:
	def __init__ (self):
		self.mask_list = []
		self.black_board = np.zeros((720, 1280))
		mode = 'video'
		# mode = 'image'
		self.main(mode)


	def main(self, mode):
		if mode == 'video':
			# Gamma 보정 함수
			

		# 비디오 처리
			
			cap = cv2.VideoCapture(input_video_path)
			if not cap.isOpened():
				print("Error: Could not open video source.")
				exit()
			

			# 비디오 프레임의 너비, 높이, 초당 프레임 수 가져오기
			frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			fps = int(cap.get(cv2.CAP_PROP_FPS))

			# VideoWriter 객체 생성 (코덱 설정, 파일 경로, FPS, 프레임 크기)
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (XVID, MP4V 등)
			out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

			# 프레임 스킵 설정 (속도를 올리기 위해 프레임 건너뜀)
			frame_skip = 0  # 2 프레임마다 1프레임 처리

			while True:
				for _ in range(frame_skip):
					ret, _ = cap.read()  # 프레임 건너뜀
				ret, im = cap.read()
				
				if not ret:
					break
				
				st = time.time() #시작 시간

				# Gamma 보정 (필요시 비활성화 가능)
				gamma = 1  #밝기 배수
				im = self.adjust_gamma(im, gamma=gamma)

				# Mixed Precision (FP16) 적용
				with torch.cuda.amp.autocast():
					outputs = predictor(im)  # 모델 예측

				pred_classes, pred_masks, pred_score = self.get_output(outputs)
				# num_parking = pred_classes.numpy().count(15) # parking class 개수
				num_parking = np.sum(pred_classes.numpy() == 15)
				print(num_parking)

				# image,  distance= self.image_process(pred_classes, pred_masks, pred_score, im)
				out_image = self.process(im, pred_classes, pred_masks, pred_score)
				# self.maintain_area()
				

				# FPS 계산			
				total_time = time.time() - st
				FPS = int(1 / total_time)
				cv2.putText(out_image, f"FPS: {FPS}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
				cv2.putText(out_image, f"Num Parking: {num_parking}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
				out.write(out_image)
				# 결과 출력
				cv2.imshow("Video", out_image)

				# Esc 키 누르면 종료
				if cv2.waitKey(1) & 0xFF == 27:
					break


			cap.release()
			out.release()
			cv2.destroyAllWindows()


	def process(self, im, pred_classes, pred_masks, pred_scores):
		# instances = outputs["instances"]
		# target_classes = [15] # 시각화할 클래스 번호
		compare_dis = None
		best_idx = None
		for i in range(len(pred_classes)):
			class_id = pred_classes[i].item()
			class_name = test_metadata.thing_classes[class_id]
			score = pred_scores[i].item()
			mask = pred_masks[i].numpy()
			y, x = np.where(mask)

			
			if class_id == 15:
				mask = pred_masks[i]

				# 무게중심 계산
				coords = torch.nonzero(mask).cpu().numpy()
				centroid_y, centroid_x = np.mean(coords, axis=0)

				#무게중심 시각화
				cv2.circle(im, (int(centroid_x), int(centroid_y)), 2, (0,0,255), 2)

				#무게중심까지의 상대거리 계산
				distance = self.get_distance(centroid_x, centroid_y, im.shape[1], im.shape[0])
				if compare_dis is not None:
					if compare_dis > distance:
						compare_dis = distance
						best_idx = i
				else:
					compare_dis = distance
					best_idx = i

					
				# 클래스명 및 컨피던스 표시
				
				if len(y) > 0 and len(x) > 0:
					y, x = y[0], x[0]  # 첫 번째 마스크 위치
					cv2.putText(
						im, f"{class_name}: {score:.2f}, Dis: {distance:.2f}", 
						(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
					)
			else:
				if len(y) > 0 and len(x) > 0:
					y, x = y[0], x[0]  # 첫 번째 마스크 위치
					cv2.putText(
						im, f"{class_name}: {score:.2f}", 
						(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
					)
		self.masking(im,  pred_classes, pred_masks, best_idx)
			# 클래스명 및 신뢰도 로그 출력
			# print(f"Detected {class_name} with confidence: {score:.2f}")
					# 마스크 적용
		return im


	def masking(self,im, pred_classes, pred_masks, best_idx):
		for i in range(len(pred_classes)):
			mask = pred_masks[i].numpy()
			class_id = pred_classes[i].item()
			class_name = test_metadata.thing_classes[class_id]
			color = class_colors.get(class_name, [255, 255, 255])  # Default to white if not found
			if best_idx is not None:
				if i == best_idx:
					color = [0,0,255]
					for c in range(3):  # BGR 순으로 채널
						im[:, :, c] = np.where(mask == 1,
												im[:, :, c] * 0.3 + color[2-c] * 0.7,  # BGR -> RGB 매핑
												im[:, :, c])
					# 윤곽선 추출
					mask_uint8 = (mask * 255).astype(np.uint8)  # mask를 uint8로 변환
					contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

					# 원본 이미지에 윤곽선 그리기
					cv2.drawContours(im, contours, -1, (0, 255, 0), 5)  # 초록색 윤곽선
				else:
					for c in range(3):  # BGR 순으로 채널
						im[:, :, c] = np.where(mask == 1,
												im[:, :, c] * 0.5 + color[2-c] * 0.5,  # BGR -> RGB 매핑
												im[:, :, c])
			else:
				for c in range(3):  # BGR 순으로 채널
					im[:, :, c] = np.where(mask == 1,
											im[:, :, c] * 0.5 + color[2-c] * 0.5,  # BGR -> RGB 매핑
											im[:, :, c])

		return im

	def get_output(self, outputs):
		instances = outputs["instances"].to("cpu")
		pred_classes = instances.pred_classes
		pred_masks = instances.pred_masks
		# self.mask_list.append(pred_masks)
		# if len(self.mask_list) >= 3:
		# 	self.mask_list.pop(0)
		# print(f'마스킹 길이 {len(self.mask_list)}')
		pred_score = instances.scores
		return pred_classes, pred_masks, pred_score

	# def draw_minRect(self, img, coords):
	# 	rect = cv2.minAreaRect(coords)
	# 	box = cv2.boxPoints(rect)
	# 	box = np.int0(box)
	# 	# print(box)
	# 	cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
	# 	return img



	def image_process(self, pred_classes, pred_masks, pred_score, image):
		distance = None
		for i in range(len(pred_classes)):
			if pred_classes[i] == 15:
				mask = pred_masks[i]

				# 무게중심 계산
				coords = torch.nonzero(mask).cpu().numpy()
				centroid_y, centroid_x = np.mean(coords, axis=0)

				#무게중심 시각화
				cv2.circle(image, (int(centroid_x), int(centroid_y)), 2, (0,0,255), 2)

				#무게중심까지의 상대거리 계산
				distance = self.get_distance(centroid_x, centroid_y, image.shape[1], image.shape[0])


		return image, distance
	
	def get_distance(self, u, v, width, height):
		fx = width/2
		fy = height/2
		Cx = width/2
		Cy = height/2

		Z = fy / (Cy-v) #전방 거리
		X = ((u-Cx)*Z)/fx #측방 거리
		distance = np.sqrt(X**2 + Z**2) #2D 상대거리
		
		return distance



	def adjust_gamma(self, image, gamma=1.0):
			inv_gamma = 1.0 / gamma
			table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
			return cv2.LUT(image, table)

if __name__ == "__main__":
	start = parking()