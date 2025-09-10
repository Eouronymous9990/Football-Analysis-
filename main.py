import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# تهيئة Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# دالة لحساب الزاوية بين 3 نقاط (مثلاً: ورك - ركبة - كاحل)
def calculate_angle(a, b, c):
    a = np.array(a)  # أول نقطة
    b = np.array(b)  # النقطة الوسطى (المفصل اللي عايز الزاوية عنده)
    c = np.array(c)  # النقطة الأخيرة

    # اتجاهات المتجهات
    ba = a - b
    bc = c - b

    # حساب الزاوية
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# مسار الفيديو
video_path = r"C:\Users\zbook 17 g3\Downloads\1-3.mp4"
cap = cv2.VideoCapture(video_path)

# إعداد ملف حفظ الفيديو الناتج
output_path = "output_pose.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, int(cap.get(5)), 
                      (int(cap.get(3)), int(cap.get(4))))

# إعداد ملف CSV للزوايا
csv_file = open("angles_output.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Left Knee Angle", "Right Knee Angle"])

with mp_pose.Pose(min_detection_confidence=0.7,
                  min_tracking_confidence=0.7,
                  model_complexity=2) as pose:

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # تحويل الألوان
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # رجوع للصورة للعرض
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # إحداثيات الورك والركبة والكاحل (يمين وشمال)
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # حساب الزوايا
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # كتابة الزوايا على الصورة
            cv2.putText(image, f"L_Knee: {int(left_knee_angle)} deg",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(image, f"R_Knee: {int(right_knee_angle)} deg",
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # حفظ الزوايا في CSV
            csv_writer.writerow([frame_num, left_knee_angle, right_knee_angle])

        except:
            pass

        # رسمة الهيكل العظمي
        mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                                  )

        # حفظ الفيديو
        out.write(image)

        # عرض الفيديو
        cv2.imshow("3D Pose Estimation - Football Kick", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        frame_num += 1

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

print(f"[INFO] الفيديو محفوظ في: {os.path.abspath(output_path)}")
print(f"[INFO] ملف الزوايا محفوظ في: {os.path.abspath('angles_output.csv')}")
