class ProcessingParams:
	def __init__(self, frame_resized, yolo_model, window_scale_factor, car_fix, car_fix2, car_back_img,
				 car_back_imgS, car_front_imgS, car_front_img, stop_img, mtx, dist,focal_length_px, vehicle_height_m, moto_back_img,
				 moto_back_imgS, car_fix_curve_left, car_fix_curve_right, car_fix_move, car_back_imgM, car_front_imgM, moto_back_imgM,
				 car_fix2_move, car_fix_curve_left_move, car_fix_curve_right_move,
				 truck_back_img, truck_back_imgM, truck_back_imgS,traffic,accident,road_close):
		self.frame_resized = frame_resized
		self.yolo_model = yolo_model
		self.window_scale_factor = window_scale_factor
		self.car_fix = car_fix
		self.car_fix2 = car_fix2
		self.car_back_img = car_back_img
		self.car_back_imgS = car_back_imgS
		self.car_front_imgS = car_front_imgS
		self.car_front_img = car_front_img
		self.stop_img = stop_img
		self.mtx = mtx
		self.dist = dist
		self.focal_length_px = focal_length_px
		self.vehicle_height_m = vehicle_height_m
		self.moto_back_img = moto_back_img
		self.moto_back_imgS = moto_back_imgS
		self.car_fix_curve_left = car_fix_curve_left
		self.car_fix_curve_right = car_fix_curve_right
		self.car_fix_move = car_fix_move
		self.car_back_imgM = car_back_imgM
		self.car_front_imgM = car_front_imgM
		self.moto_back_imgM = moto_back_imgM
		self.car_fix2_move = car_fix2_move
		self.car_fix_curve_left_move = car_fix_curve_left_move
		self.car_fix_curve_right_move = car_fix_curve_right_move
		self.truck_back_img = truck_back_img
		self.truck_back_imgM = truck_back_imgM
		self.truck_back_imgS = truck_back_imgS
		self.traffic=traffic
		self.accident=accident
		self.road_close=road_close