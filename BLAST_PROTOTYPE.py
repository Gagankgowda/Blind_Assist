# prototype 3 - Blind Assist 
# ==============================================================================

import cv2
import time
import torch
import speech_recognition as sr
import pyttsx3
import threading
import queue
import google.generativeai as genai 
from PIL import Image

# --- IMPORT FOR WINDOWS FIX ---
try:
    import pythoncom
except ImportError:
    pythoncom = None

from ultralytics import YOLO

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

#API
GEMINI_API_KEY = "AIzaSyBUi4zrBbfCxErpoPNVhNh_EEJGZcBfwIY"

IP_CAMERA_URL = "https://192.168.31.197:8080/video" 

# !!! MICROPHONE CONFIGURATION !!!
MIC_ID = 2 

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
try:
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
except:
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')

print(f"Initializing Blind Assist (Cam: Phone, Mic ID: {MIC_ID})...")

# ==============================================================================
# 2. THREADED CAMERA
# ==============================================================================
class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.status, self.frame = self.capture.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                if status:
                    with self.lock:
                        self.status = status
                        self.frame = frame
                else:
                    time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.status, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()

# ==============================================================================
# 3. VOICE ENGINE
# ==============================================================================
speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None: break
        try:
            if pythoncom: pythoncom.CoInitialize()
            temp_engine = pyttsx3.init()
            temp_engine.setProperty('rate', 170)
            print(f"Blas speaking: '{text}'")
            temp_engine.say(text)
            temp_engine.runAndWait()
            del temp_engine
            if pythoncom: pythoncom.CoUninitialize()
        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

# ==============================================================================
# 4. LOAD MODELS
# ==============================================================================
try:
    model = YOLO("yolov8x.pt") 
    if torch.cuda.is_available(): model.to('cuda')
    print("YOLOv8 loaded.")
except Exception: exit()

try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    if torch.cuda.is_available(): midas.to('cuda')
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    print("MiDaS loaded.")
except Exception: midas = None

memory = {}
last_listen_time = time.time()
LISTEN_COOLDOWN = 6 

# ==============================================================================
# 5. HELPER FUNCTIONS
# ==============================================================================

def get_zone(x, y, w, h):
    if y < h/3: return "top" 
    elif y > 2*h/3: return "bottom"
    else: return "center"

def get_object_distance_meters(depth_map, box):
    """
    Returns estimated distance in METERS.
    """
    x1, y1, x2, y2 = map(int, box)
    x_center = (x1 + x2) // 2
    y_center = (y1 + y2) // 2
    h, w = depth_map.shape
    
    x_safe = min(w-1, max(0, x_center))
    y_safe = min(h-1, max(0, y_center))
    
    depth_val = depth_map[y_safe, x_safe]
    
    if depth_val <= 0: return 0.0
    
    # --- CALIBRATION FORMULA ---
    # MiDaS Output (depth_val) is 'Inverse Depth'.
    # Distance = Coefficient / depth_val
    # 800 is a rough default. Increase this number if it says objects are too close.
    # Decrease it if it says objects are too far.
    CALIBRATION_COEFF = 800.0 
    
    estimated_meters = CALIBRATION_COEFF / depth_val
    return estimated_meters 

def update_memory(label, zone, distance=None):
    memory[label] = {"last_seen": zone, "timestamp": time.strftime("%H:%M:%S"), "distance": distance}

def speak(text):
    speech_queue.put(text)

def listen_command():
    r = sr.Recognizer()
    try:
        with sr.Microphone(device_index=MIC_ID) as source:
            print(f"\nListening (Mic ID {MIC_ID})...")
            r.adjust_for_ambient_noise(source, duration=0.2)
            r.energy_threshold = 300 
            audio = r.listen(source, timeout=4, phrase_time_limit=5)
            print("Processing audio...")
            text = r.recognize_google(audio)
            print(f"You said: '{text}'")
            return text
    except sr.WaitTimeoutError:
        print("Timeout (No speech detected).")
        return None
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except Exception as e:
        print(f"Mic Error: {e}")
        return None

# --- GEMINI FUNCTIONS ---
def describe_scene_gemini(frame):
    def task():
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            response = gemini_model.generate_content(["Describe this scene briefly.", pil_image])
            speak(response.text)
        except: speak("Connection error.")
    threading.Thread(target=task).start()

def read_text_gemini(frame):
    def task():
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            response = gemini_model.generate_content(["Read the text in this image.", pil_image])
            speak(response.text)
        except: speak("Reading error.")
    threading.Thread(target=task).start()

def answer_query(query, frame):
    if not query:
        speak("I didn't hear a command.")
        return

    words = query.lower().split()
    
    if "read" in query: read_text_gemini(frame); return
    if "describe" in query: describe_scene_gemini(frame); return
    
    if "how far" in query or "distance" in query:
        found = False
        for obj_name in memory.keys():
            if obj_name in words:
                info = memory[obj_name]
                dist = info.get('distance', 0)
                speak(f"The {obj_name} is about {dist:.1f} meters away.")
                found = True
                break
        if not found: speak("I don't see that.")
        return

    if "where" in query or "find" in query:
        found = False
        for obj_name in memory.keys():
            if obj_name in words:
                speak(f"The {obj_name} is {memory[obj_name]['last_seen']}.")
                found = True
                break
        if not found: speak("I haven't seen that.")
        return
        
    speak("I heard you, but I don't know that command.")

# ==============================================================================
# 6. MAIN LOOP
# ==============================================================================
def main():
    global last_listen_time
    
    print(f"Connecting to Phone Camera at: {IP_CAMERA_URL}")
    threaded_cam = ThreadedCamera(IP_CAMERA_URL)
    time.sleep(2.0)

    print("\n--- Wireless Blind Assist Running ---")
    print("Press 's' on LAPTOP KEYBOARD to speak.")

    while True:
        ret, frame = threaded_cam.read()
        if not ret or frame is None: continue

        frame_small = cv2.resize(frame, (640, 480))
        yolo_results = model(frame_small, verbose=False)[0]
        
        depth_map = None
        if midas:
            img = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            t_img = transform(img).to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                pred = midas(t_img)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=frame_small.shape[:2], mode="bicubic", align_corners=False
                ).squeeze()
            depth_map = pred.cpu().numpy()

        for r in yolo_results.boxes:
            box = r.xyxy[0]
            label = model.names[int(r.cls[0])]
            confidence = float(r.conf[0]) # Get confidence score
            
            x1, y1, x2, y2 = map(int, box)
            
            # 1. DRAW BOX
            cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 2. CALCULATE METERS
            dist_m = get_object_distance_meters(depth_map, box) if depth_map is not None else 0
            update_memory(label, "center", dist_m)

            # 3. CREATE LABEL TEXT (Name + Distance in Meters)
            if dist_m > 0:
                label_text = f"{label}: {dist_m:.1f}m"
            else:
                label_text = f"{label}"
            
            # 4. DRAW LABEL BACKGROUND (Black Box for Readability)
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_small, (x1, y1 - 20), (x1 + w, y1), (0, 0, 0), -1)
            
            # 5. DRAW TEXT
            cv2.putText(frame_small, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Blind Assist Laptop View", frame_small)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if time.time() - last_listen_time > LISTEN_COOLDOWN:
                query = listen_command()
                answer_query(query, frame_small.copy())
                last_listen_time = time.time()
        elif key == ord('q'):
            break

    threaded_cam.stop()
    cv2.destroyAllWindows()
    speech_queue.put(None)

if __name__ == "__main__":
    main()