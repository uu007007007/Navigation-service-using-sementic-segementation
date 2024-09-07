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
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
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
    "Human": [165, 42, 42]
}

#inference 할 영상 파일
input_video_path = "/home/road2022/parking_project/detectron2/front2.mp4"
output_video_path = 'output_video.mov'



class parking:
	def __init__ (self):
		self.mask_list = []

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
			fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 코덱 설정 (XVID, MP4V 등)
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
				black_board = np.zeros((720, 1280), dtype=np.uint8)
				black_board = np.stack((black_board,) * 3, axis=-1)

				# Mixed Precision (FP16) 적용
				with torch.cuda.amp.autocast():
					outputs = predictor(im)  # 모델 예측

				pred_classes, pred_masks, pred_scores = self.get_output(outputs)
				navi = self.BEV_seg(black_board, pred_masks)
				image_ori = self.visualize(im, pred_classes, pred_masks, pred_scores)
				# self.maintain_area()
				out_image = self.image_process(pred_classes, pred_masks, pred_scores, image_ori)

				# FPS 계산			
				total_time = time.time() - st
				FPS = int(1 / total_time)
				cv2.putText(out_image, f"FPS: {FPS}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
				out.write(out_image)
				# 결과 출력
				cv2.imshow("Video", out_image)
				cv2.imshow("black board", navi)

				# Esc 키 누르면 종료
				if cv2.waitKey(1) & 0xFF == 27:
					break


			cap.release()
			out.release()
			cv2.destroyAllWindows()


	def get_fixed_color(self, i):
		random.seed(i)  # 클래스 인덱스에 따라 색상을 고정
		r = random.randint(0, 255)/255
		g = random.randint(0, 255)/255
		b = random.randint(0, 255)/255
		return (r, g, b)

	def visualize(self, im, pred_classes, pred_masks, pred_scores):
		# instances = outputs["instances"]
		target_classes = [15] # 시각화할 클래스 번호
		
		for i in range(len(pred_classes)):
			class_id = pred_classes[i].item()
			class_name = test_metadata.thing_classes[class_id]
			score = pred_scores[i].item()
			mask = pred_masks[i].numpy()
			color = class_colors.get(class_name, [255, 255, 255])  # Default to white if not found

			# 마스크 적용
			for c in range(3):  # BGR 순으로 채널
				im[:, :, c] = np.where(mask == 1,
										im[:, :, c] * 0.5 + color[2-c] * 0.5,  # BGR -> RGB 매핑
										im[:, :, c])
			# 클래스명 및 컨피던스 표시
			y, x = np.where(mask)
			if len(y) > 0 and len(x) > 0:
				y, x = y[0], x[0]  # 첫 번째 마스크 위치
				cv2.putText(
					im, f"{class_name}: {score:.2f}", 
					(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
				)

			# 클래스명 및 신뢰도 로그 출력
			# print(f"Detected {class_name} with confidence: {score:.2f}")

		return im
	# def maintain_area(self):
	# 	for mask in self.mask_list[0]:
			
		# 교차 영역 계산
		# intersection_area, intersection_polygon = cv2.intersectConvexConvex(polygon1, polygon2)

	def get_output(self, outputs):
		instances = outputs["instances"].to("cpu")
		pred_classes = instances.pred_classes
		pred_masks = instances.pred_masks
		self.mask_list.append(pred_masks)
		if len(self.mask_list) >= 3:
			self.mask_list.pop(0)
		print(f'마스킹 길이 {len(self.mask_list)}')
		pred_score = instances.scores
		return pred_classes, pred_masks, pred_score

	def draw_minRect(self, img, coords):
		rect = cv2.minAreaRect(coords)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		# print(box)
		cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
		return img



	def image_process(self, pred_classes, pred_masks, pred_score, image):
		print(image.shape)
		# image = self.bird_eye_view(image)
		for i in range(len(pred_classes)):
			if pred_classes[i] == 15:
				if pred_score[i] >= 0.9:
					# print("pred_score :", pred_score[i])
					mask = pred_masks[i]
					# 객체의 마스크에서 활성화된 픽셀의 (x, y) 좌표 추출
					coords = torch.nonzero(mask).cpu().numpy()
					# print(f"Object {i} mask coordinates:")
					# print(coords)
					for coord in coords:
						c0 = coord[0]
						c1 = coord[1]
						coord[0] = c1
						coord[1] = c0
					# draw_minRect(image, coords)

		return image
	def bird_eye_view(self, image):
		# src_points = np.float32([[0, 0], [1280, 0], [1280, 720], [0, 720]])
		
		

		src_points = np.float32([[340, 380], [690, 380], [1340, 600], [-60, 600]])
		# points = [[340, 400], [690, 400], [1270, 600], [10, 600]]
		# for point in points:
		# 	print(point)
		# 	cv2.circle(image, point, 1, (0,0,255), thickness=2)
		# 변환할 버드아이뷰의 목표 영역 정의 (출력 이미지 크기와 동일하게 설정)
		# 출력 이미지에서의 대응 좌표 설정
		# dst_points = np.float32([[0, 0], [1280, 0], [1280, 720], [0, 720]])
		dst_points = np.float32([[0, 0], [1280, 0], [880, 720], [400, 720]])
		

		# 변환 행렬 계산
		M = cv2.getPerspectiveTransform(src_points, dst_points)

		# 투시 변환을 적용 (출력 이미지 크기를 (400, 600)으로 설정)
		output_image = cv2.warpPerspective(image, M, (1280, 720))

		return output_image
	

	def BEV_seg(self, im, masks):
		src_points = np.float32([[340, 380], [690, 380], [1340, 600], [-60, 600]])
		dst_points = np.float32([[0, 0], [1280, 0], [880, 720], [400, 720]])
		for mask in masks:
			mask = mask.numpy()
			color = [123, 104, 238]
			M = cv2.getPerspectiveTransform(src_points, dst_points)
			print(type(mask))
			mask_transformed = cv2.warpPerspective(mask.astype(np.float32), M, (1280, 720))
			for c in range(3):  # BGR 순으로 채널
					im[:, :, c] = np.where(mask_transformed == 1,
											im[:, :, c] * 0.5 + color[2-c] * 0.5,  # BGR -> RGB 매핑
											im[:, :, c])
		return im

	def adjust_gamma(self, image, gamma=1.0):
			inv_gamma = 1.0 / gamma
			table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
			return cv2.LUT(image, table)

if __name__ == "__main__":
	start = parking()