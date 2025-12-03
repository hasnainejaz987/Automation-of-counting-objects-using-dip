from PIL import Image, ImageTk
from skimage.feature import blob_log
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from skimage.feature import canny
from math import sqrt
import tkinter as tk
from tkinter import filedialog
import cv2
from skimage.io import imread
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn

#Global variables
im = None
file_path = None
input_image_label = None
output_image_label = None
object_count_label = None

#Loading Pretrained HED Model 
model = maskrcnn_resnet50_fpn(pretrained=True) 
model.eval()

def preprocess_frame(frame):
    transform = T.Compose([
        T.ToPILImage(),
        T.Grayscale(num_output_channels=1),
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    return transform(frame).unsqueeze(0)

#Function to load an image
def load_image():
    global input_image_label
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
    )
    if file_path:
        img = imread(file_path, as_gray=True)
        display_image(img, "Input Image", input_image_label)
        return img, file_path
    else:
        print("No file selected.")
        return None, None

#Function to display images in the GUI
def display_image(image, title, label):
    global root
    image = (image * 255).astype(np.uint8)  #Converting to uint8 for display
    img_pil = Image.fromarray(image)
    img_tk = ImageTk.PhotoImage(img_pil.resize((250, 250)))  #Resizes for display
    label.config(image=img_tk)
    label.image = img_tk  #Keeping reference to avoid garbage collection

#Function for Blob Detection
def perform_blob_detection():
    global im, object_count_label
    im, _ = load_image()
    if im is None:
        return

    blobs_log = blob_log(im, max_sigma=30, num_sigma=10, threshold=.1)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    num_blobs = len(blobs_log)

    #Creating a copy of the image for drawing
    output_image = np.copy(im)
    for blob in blobs_log:
        y, x, r = blob
        cv2.circle(output_image, (int(x), int(y)), int(r), (0, 255, 0), 2)

    display_image(output_image, f"Counted Objects: {num_blobs}", output_image_label)

    #Updating the object count label
    object_count_label.config(text=f"Objects Counted: {num_blobs}")
    
#Function for Edge-Based Segmentation
def show_edge_detection():
    global im, object_count_label
    if im is None:
        im, _ = load_image()
        if im is None:
            return

    edges = canny(im, sigma=2)
    display_image(edges, "Edge-Based Segmentation", output_image_label)
    object_count_label.config(text="Edge-Based Segmentation Completed")

#Function for Region-Based Segmentation
def show_region_segmentation():
    global im, object_count_label
    if im is None:
        im, _ = load_image()
        if im is None:
            return

    elevation_map = sobel(im)
    markers = np.zeros_like(im, dtype=np.int32)
    markers[im < 0.3] = 1
    markers[im > 0.7] = 2
    segmentation = watershed(elevation_map, markers)
    segmentation = remove_small_objects(segmentation > 1, 20)
    display_image(segmentation, "Region-Based Segmentation", output_image_label)
    object_count_label.config(text="Region-Based Segmentation Completed")

#live edge detection function
def live_edge_detection():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Cannot access webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #Preprocessing the frame for HED model
        input_tensor = preprocess_frame(frame)

        #Performing edge detection
        with torch.no_grad():
            output = model(input_tensor)

        #Checking  if output is empty
        if isinstance(output, list) and len(output) > 0 and 'masks' in output[0] and len(output[0]['masks']) > 0:
            edge_map = output[0]['masks'][0, 0].cpu().numpy()
        else:
            edge_map = np.zeros((256, 256))  # Empty edge map if no objects detected

        #Converting edge map to displayable format
        edge_map = (edge_map * 255).astype('uint8')

        #Displaying original frame and edge map
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Edge Detection", edge_map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#Real-Time Object Counting
def real_time_object_counting():
    cap = cv2.VideoCapture(0)  #Open webcam
    if not cap.isOpened():
        print("Cannot access webcam")
        return

    model = YOLO("yolov8n.pt")  #Loading YOLOv8 model

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from webcam")
            break

        #Pass frame to YOLO model
        results = model(frame)
        result = results[0]

        #Object labels, boxes, confidence scores, and class indices
        labels = result.names
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        clss = result.boxes.cls
        count_dict = {}

        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = box
            label = labels[int(cls)]
            confidence = conf.item()

            #Draw rectangle and label on frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", 
                        (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            #Updating count dictionary
            count_dict[label] = count_dict.get(label, 0) + 1

        #Shows counts on frame
        y_offset = 30
        for label, count in count_dict.items():
            cv2.putText(frame, f"{label}: {count}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_offset += 30

        #Displaying the frame
        cv2.imshow("Real-Time Object Counting", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#GUI Setup
root = tk.Tk()
root.geometry("950x500")
root.title("Object Counter and Segmentation Techniques")

#Layout
frame = tk.Frame(root)
frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

input_image_label = tk.Label(frame)
input_image_label.pack(side=tk.LEFT, padx=10, pady=10)

output_image_label = tk.Label(frame)
output_image_label.pack(side=tk.RIGHT, padx=10, pady=10)

object_count_label = tk.Label(root, text="Object Count: N/A", font=('Times New Roman', 16), bg='lightblue', fg='darkblue')
object_count_label.pack(pady=10)

button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, fill=tk.X)

btn_blob = tk.Button(button_frame, text="Object Counting",font=('Times New Roman', 12), command=perform_blob_detection, bg='green', fg='white')
btn_blob.pack(side=tk.LEFT, padx=10, pady=5)

btn_edge = tk.Button(button_frame, text="Edge-Based Segmentation",font=('Times New Roman', 12), command=show_edge_detection, bg='blue', fg='white')
btn_edge.pack(side=tk.LEFT, padx=10, pady=5)

btn_region = tk.Button(button_frame, text="Region-Based Segmentation",font=('Times New Roman', 12), command=show_region_segmentation, bg='orange', fg='white')
btn_region.pack(side=tk.LEFT, padx=10, pady=5)

btn_hed_edge = tk.Button(button_frame, text="Live Edge Detection",font=('Times New Roman', 12), command=live_edge_detection, bg='pink', fg='black')
btn_hed_edge.pack(side=tk.LEFT, padx=10, pady=5)

btn_real_time = tk.Button(button_frame, text="Real-Time Object Counting",font=('Times New Roman', 12), command=real_time_object_counting, bg='purple', fg='white')
btn_real_time.pack(side=tk.LEFT, padx=10, pady=5)

root.mainloop()
