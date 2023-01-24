#import libraryที่จำเป็น
import numpy as np 
import cv2

#รายชื่อหมวดหมู่ทั้งหมด เรียงตามลำดับ
CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
	"BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
	"DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
	"SOFA", "TRAIN", "TVMONITOR"]
#สีตัวกรอบที่วาดrandomใหม่ทุกครั้ง
COLORS = np.random.uniform(0,100, size=(len(CLASSES), 3))
#โหลดmodelจากแฟ้ม
net = cv2.dnn.readNetFromCaffe("C:/Users/admin/Documents/Work/PYTHONWORK/object_detection/MobileNetSSD/MobileNetSSD.prototxt","C:/Users/admin/Documents/Work/PYTHONWORK/object_detection/MobileNetSSD/MobileNetSSD.caffemodel")
#เลือกวิดีโอ/เปิดกล้อง
cap = cv2.VideoCapture("C:/Users/admin/Documents/Work/PYTHONWORK/object_detection/road_car_view.mp4")

while True:
	#เริ่มอ่านในแต่ละเฟรม
	ret, frame = cap.read()
	if ret:
		#------------- lane detection -------------
		frame = cv2.GaussianBlur(frame, (5, 5), 0)
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		#low_yellow = np.array([18, 94, 140])
		#up_yellow = np.array([48, 255, 255])
		low_yellow = np.array([20, 100, 100])
		up_yellow = np.array([30, 255, 255])
		mask = cv2.inRange(hsv, low_yellow, up_yellow)
		edges = cv2.Canny(mask, 75, 150)
		lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)

		if lines is not None:
			for line in lines:
				x1, y1, x2, y2 = line[0]
				cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
		
		#---------- object detection ---------------
		(h,w) = frame.shape[:2]
		#ทำpreprocessing
		blob = cv2.dnn.blobFromImage(frame, 0.0007843, (300,300), 127.5)
		net.setInput(blob)
		#feedเข้าmodelพร้อมได้ผลลัพธ์ทั้งหมดเก็บมาในตัวแปร detections
		detections = net.forward()

		for i in np.arange(0, detections.shape[2]):
			percent = detections[0,0,i,2]
			#กรองเอาเฉพาะค่าpercentที่สูงกว่า0.5 เพิ่มลดได้ตามต้องการ
			if percent > 0.2:
				class_index = int(detections[0,0,i,1])
				box = detections[0,0,i,3:7]*np.array([w,h,w,h])
				(startX, startY, endX, endY) = box.astype("int")

				#ส่วนตกแต่งสามารถลองแก้กันได้ วาดกรอบและชื่อ
				label = "{} [{:.2f}%]".format(CLASSES[class_index], percent*100)
				cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
				cv2.rectangle(frame, (startX-1, startY-30), (endX+1, startY), COLORS[class_index], cv2.FILLED)
				y = startY - 15 if startY-15>15 else startY+15
				cv2.putText(frame, label, (startX+20, y+5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)

		cv2.imshow("Frame", frame)
		cv2.imshow("edges", edges)
		key = cv2.waitKey(1)
		if  key == 27:
			break

#หลังเลิกใช้แล้วเคลียร์memoryและปิดกล้อง
cap.release()
cv2.destroyAllWindows() 