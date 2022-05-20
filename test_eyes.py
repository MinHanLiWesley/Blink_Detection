import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]

Eyes_points_12=[36,37,38,39,40,41,42,43,44,45,46,47]

def read_ground_truth(path):
    gt=[]
    with open(path) as f:
        lines = f.readlines()
    for idx,line in enumerate(lines):
        if idx <= 2 or idx == 71:
            continue
        x,y  = line.split(" ")
        x = float(x)
        y = float(y)
        gt.append((x,y))
        # print(f"{x} {y}")
    return gt

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    # distance = (x1 - x)**2 + (y1 - y)**2
    return distance
# For static images:
IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
    error_per_image=[]
    eyes_extracted=[]
    for idx in range(1,301):
        # fname = "./300w_cropped/02_Outdoor/outdoor_%03d" % idx
        fname = "./300w_cropped/01_Indoor/indoor_%03d" % idx
        # image = cv2.imread("./300w_cropped/01_Indoor/indoor_%03d.png" % idx)
        image = cv2.imread(fname+".png")
        # print("./300w_cropped/01_Indoor/indoor_%03d.png" % idx)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            # print(idx)
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            landmarks_extracted = []
            for index in landmark_points_68:
                x = float(face_landmarks.landmark[index].x * image.shape[1])
                y = float(face_landmarks.landmark[index].y * image.shape[0])
                landmarks_extracted.append((x, y))
#         mp_drawing.draw_landmarks(
#             image=annotated_image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_LEFT_EYE,
#             landmark_drawing_spec=None,
# )
#         mp_drawing.draw_landmarks(
#             image=annotated_image,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
#             landmark_drawing_spec=None,
# )
#         cv2.imwrite('./eyes/' + str(idx) + '.png', annotated_image)
        # for x, y in landmarks_extracted:
            # print("%.3f %.3f" % (x,y))
        for idx2, landmark in enumerate(landmarks_extracted):
            if idx2 >= 36 and idx2 <= 47:
                eyes_extracted.append(landmark)
        ground_truths = read_ground_truth(fname+".pts")
        # interocular_distance_predict = euclaideanDistance(landmarks_extracted[36],landmarks_extracted[45])
        interocular_distance_gt = (euclaideanDistance(ground_truths[36],ground_truths[45]))
        total_error = 0.
        for j in range(68):
            if j >= 36 and j <= 47:
                total_error += 100*euclaideanDistance(landmarks_extracted[j],ground_truths[j])/1.1
        total_error /= (12*interocular_distance_gt)
        # print(f"{idx}: {total_error}")
        if total_error> 6:
            print(idx, total_error)
        error_per_image.append(total_error)

print(f"mean error: {sum(error_per_image)/len(error_per_image)}")

# for error in error_per_image:
#     print(error)
print(len(error_per_image))
        ## 
#data
# x = [1, 3, 5, 7, 9]
# y = [2, 4, 6, 8, 10]

# 製作figure  
fig = plt.figure()   

#圖表的設定
ax = fig.add_subplot(1, 1, 1)

#散佈圖
ax.scatter(list(range(1,len(error_per_image)+1)), error_per_image, color='red')
plt.xlabel('Image ID',fontsize=15)
plt.ylabel('NME(%)',fontsize=15)
# plt.title('Overall NME of Outdoor Images',fontsize=20)
plt.title('NME of ROI of Indoor Images',fontsize=20)
# plt.show() 

        # print('face_landmarks:', face_landmarks)
    #     mp_drawing.draw_landmarks(
    #         image=annotated_image,
    #         landmark_list=face_landmarks,
    #         connections=mp_face_mesh.FACEMESH_TESSELATION,
    #         landmark_drawing_spec=None,
    #         connection_drawing_spec=mp_drawing_styles
    #         .get_default_face_mesh_tesselation_style())
    #     mp_drawing.draw_landmarks(
    #         image=annotated_image,
    #         landmark_list=face_landmarks,
    #         connections=mp_face_mesh.FACEMESH_CONTOURS,
    #         landmark_drawing_spec=None,
    #         connection_drawing_spec=mp_drawing_styles
    #         .get_default_face_mesh_contours_style())
    #     mp_drawing.draw_landmarks(
    #         image=annotated_image,
    #         landmark_list=face_landmarks,
    #         connections=mp_face_mesh.FACEMESH_IRISES,
    #         landmark_drawing_spec=None,
    #         connection_drawing_spec=mp_drawing_styles
    #         .get_default_face_mesh_iris_connections_style())
    # cv2.imwrite('./annotated_image/' + str(idx) + '.png', annotated_image)