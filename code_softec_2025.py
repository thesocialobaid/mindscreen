import cv2
import time
import datetime
import numpy as np
import mediapipe as mp
import os
import requests
import json
import pyautogui  # For triggering system events
import threading  # For non-blocking API calls
import openai
# Disable PyAutoGUI fail-safe to prevent interruptions
pyautogui.FAILSAFE = False

class CursorOverlay:
    """Creates a q overlay window to show cursor position on screen"""
    
    def __init__(self, size=15, color=(0, 120, 255)):  # Blue color by default
        self.size = size
        self.color = color
        self.position = (0, 0)
        self.visible = False
        self.overlay_window = None
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Create overlay window
        self._create_overlay()
    
    def _create_overlay(self):
        """Create the transparent overlay window"""
        if cv2.getWindowProperty("CursorOverlay", cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow("CursorOverlay")
            
        # Create a named window with click-through property
        cv2.namedWindow("CursorOverlay", cv2.WND_PROP_TOPMOST)
        cv2.setWindowProperty("CursorOverlay", cv2.WND_PROP_TOPMOST, 1)
        
        # Make it frameless
        cv2.setWindowProperty("CursorOverlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        # Create transparent image the size of screen
        self.overlay_img = np.zeros((self.screen_height, self.screen_width, 4), dtype=np.uint8)
        
        # Initial position
        self._update_overlay()
        
        # Show window
        cv2.imshow("CursorOverlay", self.overlay_img)
        self.visible = True
    
    def set_position(self, x, y):
        """Update the cursor position"""
        self.position = (int(x), int(y))
        self._update_overlay()
    
    def _update_overlay(self):
        """Redraw the overlay with the cursor"""
        # Create transparent image
        self.overlay_img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Draw cursor dart
        x, y = self.position
        # Draw circle
        cv2.circle(self.overlay_img, (x, y), self.size, self.color, -1)
        # Draw "dart" lines
        cv2.line(self.overlay_img, (x-self.size*2, y), (x+self.size*2, y), self.color, 2)
        cv2.line(self.overlay_img, (x, y-self.size*2), (x, y+self.size*2), self.color, 2)
        
        # Display
        if self.visible:
            cv2.imshow("CursorOverlay", self.overlay_img)
    
    def show(self):
        """Show the overlay"""
        if not self.visible:
            self._create_overlay()
            self.visible = True
    
    def hide(self):
        """Hide the overlay"""
        if self.visible:
            cv2.destroyWindow("CursorOverlay")
            self.visible = False
    
    def cleanup(self):
        """Clean up resources"""
        self.hide()

class HandDetector:
    # Keep your existing HandDetector class implementation
    # ...all methods and properties from your original code...
    def __init__(self, static_mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        
        # For drawing hand landmarks
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Store previous hand positions for movement detection
        self.prev_positions = []
        self.gesture_history = []
        
        # Tap detection variables
        self.tap_start_time = None
        self.tap_location = None
        self.tap_in_progress = False
        self.tap_count = 0
        self.last_tap_time = 0
        
        # Pinch drag variables
        self.drag_start_position = None
        self.drag_in_progress = False
        self.drag_history = []
        
        # Scroll variables
        self.scroll_start_position = None
        self.scroll_active = False
        self.last_scroll_pos = None
        
        # Visual zones for actions
        self.zones = []
    
    def add_interaction_zone(self, name, x, y, width, height, color=(0, 255, 0)):
        """Add an interactive zone to the screen"""
        self.zones.append({
            'name': name,
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'color': color
        })
    
    def find_hands(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        self.results = self.hands.process(rgb_frame)
        
        # Initialize hand data
        hands_data = []
        
        # Check if hands were detected
        if self.results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                # Get hand type (Left or Right)
                if self.results.multi_handedness:
                    handedness = self.results.multi_handedness[hand_idx].classification[0].label
                else:
                    handedness = "Unknown"
                
                # Extract landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((cx, cy))
                
                # Store hand data
                hands_data.append({
                    'type': handedness,
                    'landmarks': landmarks
                })
                
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Return processed frame and hand data
        return frame, hands_data
    
    def draw_interaction_zones(self, frame):
        """Draw all defined interaction zones on the frame"""
        for zone in self.zones:
            cv2.rectangle(frame, 
                         (zone['x'], zone['y']), 
                         (zone['x'] + zone['width'], zone['y'] + zone['height']), 
                         zone['color'], 2)
            cv2.putText(frame, zone['name'], (zone['x'], zone['y'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone['color'], 2)
        return frame
    
    def is_point_in_zone(self, point, zone):
        """Check if a point is inside a zone"""
        x, y = point
        return (zone['x'] <= x <= zone['x'] + zone['width'] and 
                zone['y'] <= y <= zone['y'] + zone['height'])
    
    def check_point_in_zones(self, point):
        """Return the name of the zone containing the point, or None"""
        for zone in self.zones:
            if self.is_point_in_zone(point, zone):
                return zone['name']
        return None
    
    def detect_tap(self, index_tip, thumb_tip, frame):
        """Detect tapping gesture and location"""
        # Calculate distance between index tip and thumb tip
        distance = ((index_tip[0] - thumb_tip[0])**2 + (index_tip[1] - thumb_tip[1])**2)**0.5
        
        # Current time for timing calculations
        current_time = time.time()
        
        # Tap detection logic
        tap_event = None
        tap_location = None
        
        # Check if fingers are close (potential tap)
        if distance < 40:  # Close enough to be a tap
            # If we're not already tracking a tap, start tracking
            if not self.tap_in_progress:
                self.tap_in_progress = True
                self.tap_start_time = current_time
                self.tap_location = index_tip
        else:
            # If we were tracking a tap and fingers now separated
            if self.tap_in_progress:
                # Check if it was a quick tap (not a hold)
                tap_duration = current_time - self.tap_start_time
                if tap_duration < 0.3:  # Quick tap under 300ms
                    # Check if it's in any of our interaction zones
                    zone_name = self.check_point_in_zones(self.tap_location)
                    tap_event = f"tap at {self.tap_location} " + (f"in {zone_name}" if zone_name else "")
                    tap_location = self.tap_location
                    
                    # Double tap detection
                    if current_time - self.last_tap_time < 0.5:  # Within 500ms
                        tap_event = f"double {tap_event}"
                    
                    self.last_tap_time = current_time
                
                # Reset tap tracking
                self.tap_in_progress = False
        
        return tap_event, tap_location
    
    def detect_pinch_drag(self, index_tip, thumb_tip, frame):
        """Detect drag with pinch gesture"""
        # Calculate distance between index tip and thumb tip
        distance = ((index_tip[0] - thumb_tip[0])**2 + (index_tip[1] - thumb_tip[1])**2)**0.5
        
        # Average position between thumb and index (pinch center point)
        pinch_point = ((index_tip[0] + thumb_tip[0]) // 2, (index_tip[1] + thumb_tip[1]) // 2)
        
        drag_event = None
        
        # Pinch gesture detected
        if distance < 40:  # Fingers close enough for pinch
            # Start new drag if not already in progress
            if not self.drag_in_progress:
                self.drag_in_progress = True
                self.drag_start_position = pinch_point
                self.drag_history = [pinch_point]
                drag_event = f"pinch started at {pinch_point}"
            else:
                # Continue existing drag
                if len(self.drag_history) > 0:
                    last_point = self.drag_history[-1]
                    dx = pinch_point[0] - last_point[0]
                    dy = pinch_point[1] - last_point[1]
                    
                    # Only record movement if it's significant
                    if (dx**2 + dy**2) > 25:  # Movement threshold
                        self.drag_history.append(pinch_point)
                        
                        # Determine drag direction
                        if abs(dx) > abs(dy):  # Horizontal movement dominant
                            direction = "right" if dx > 0 else "left"
                        else:  # Vertical movement dominant
                            direction = "down" if dy > 0 else "up"
                            
                        # Check if in zone
                        zone_name = self.check_point_in_zones(pinch_point)
                        zone_text = f" in {zone_name}" if zone_name else ""
                        
                        drag_event = f"dragging {direction}{zone_text}"
                        
                        # Draw drag line
                        if len(self.drag_history) > 1:
                            for i in range(1, len(self.drag_history)):
                                cv2.line(frame, self.drag_history[i-1], self.drag_history[i], (255, 0, 255), 2)
        else:
            # End drag if it was in progress
            if self.drag_in_progress:
                self.drag_in_progress = False
                if len(self.drag_history) > 1:
                    drag_event = f"drag ended at {pinch_point}"
                self.drag_history = []
        
        return drag_event, pinch_point if self.drag_in_progress else None
    
    def detect_scroll(self, hand_landmarks, frame_height):
        """Detect scrolling gesture (open palm moving up or down)"""
        # Check if all fingers are extended (open palm)
        index_tip = hand_landmarks[8]
        middle_tip = hand_landmarks[12]
        ring_tip = hand_landmarks[16]
        pinky_tip = hand_landmarks[20]
        wrist = hand_landmarks[0]
        
        # Basic check if fingers are extended (above wrist height)
        fingers_extended = all(tip[1] < wrist[1] for tip in [index_tip, middle_tip, ring_tip, pinky_tip])
        
        scroll_event = None
        palm_center = None
        
        if fingers_extended:
            # Calculate palm center
            palm_center = (
                (index_tip[0] + middle_tip[0] + ring_tip[0] + pinky_tip[0]) // 4,
                (index_tip[1] + middle_tip[1] + ring_tip[1] + pinky_tip[1]) // 4
            )
            
            # Start new scroll if not already scrolling
            if not self.scroll_active:
                self.scroll_active = True
                self.scroll_start_position = palm_center
                self.last_scroll_pos = palm_center
            else:
                # Continue existing scroll
                if self.last_scroll_pos:
                    dy = palm_center[1] - self.last_scroll_pos[1]
                    
                    # Only trigger scroll event if movement is significant
                    if abs(dy) > 15:  # Scroll threshold
                        direction = "down" if dy > 0 else "up"
                        scroll_amount = abs(dy) // 5  # Scale the scroll amount
                        
                        # Check the relative position in the frame
                        position = "top" if palm_center[1] < frame_height/3 else \
                                  "middle" if palm_center[1] < 2*frame_height/3 else "bottom"
                        
                        scroll_event = f"scrolling {direction} ({scroll_amount} units) in {position} area"
                        self.last_scroll_pos = palm_center
        else:
            # End scroll if it was active
            if self.scroll_active:
                self.scroll_active = False
                scroll_event = "scroll ended"
                self.last_scroll_pos = None
        
        return scroll_event, palm_center if self.scroll_active else None
    
    def describe_hand_movements(self, hands_data, frame_width, frame_height, frame):
        """Generate descriptions of hand positions and movements"""
        if not hands_data:
            return "No hands detected", []
        
        descriptions = []
        events = []
        
        # Check if we have previous positions to compare
        has_prev = len(self.prev_positions) > 0 and len(self.prev_positions) == len(hands_data)
        
        for i, hand in enumerate(hands_data):
            hand_type = hand['type']
            landmarks = hand['landmarks']
            
            # Get key points
            wrist = landmarks[0]
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            
            # Position in frame - correcting for mirroring
            position_x = "right" if wrist[0] < frame_width/3 else \
                        "center" if wrist[0] < 2*frame_width/3 else "left"
            position_y = "top" if wrist[1] < frame_height/3 else \
                        "middle" if wrist[1] < 2*frame_height/3 else "bottom"
            
            # Describe basic position
            position_desc = f"{hand_type} hand in {position_y} {position_x} of frame"
            
            # Check for simple gestures
            # Distance between thumb and index finger
            thumb_index_dist = ((thumb_tip[0] - index_tip[0])**2 + 
                            (thumb_tip[1] - index_tip[1])**2)**0.5
            
            gesture = ""
            # Check if fingers are extended (simple approach)
            if index_tip[1] < landmarks[5][1] and middle_tip[1] < landmarks[9][1]:
                if ring_tip[1] < landmarks[13][1] and pinky_tip[1] < landmarks[17][1]:
                    gesture = "open palm"
                    # Treat open palm as a tap (click)
                    if time.time() - self.last_tap_time > 0.5:
                        events.append({'type': 'tap', 'location': index_tip})
                        self.last_tap_time = time.time()
                else:
                    gesture = "peace sign"
            elif thumb_index_dist < 40:  # Close together
                gesture = "pinch gesture"
            elif all(tip[1] > landmarks[0][1] for tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
                gesture = "closed fist"
            
            # Movement detection - also correct for mirroring
            movement = "stationary"
            if has_prev:
                prev_wrist = self.prev_positions[i]['landmarks'][0]
                dx = wrist[0] - prev_wrist[0]
                dy = wrist[1] - prev_wrist[1]
                distance = (dx**2 + dy**2)**0.5
                
                if distance > 25:  # Threshold for movement
                    # Determine direction - invert horizontal directions
                    direction = ""
                    if abs(dx) > abs(dy):  # Horizontal movement is dominant
                        direction = "left" if dx > 0 else "right"  # Inverted for mirroring
                    else:  # Vertical movement is dominant
                        direction = "down" if dy > 0 else "up"
                    
                    movement = f"moving {direction}"
            
            # Advanced gesture detection
            tap_event, tap_location = self.detect_tap(index_tip, thumb_tip, frame)
            drag_event, drag_location = self.detect_pinch_drag(index_tip, thumb_tip, frame)
            scroll_event, scroll_location = self.detect_scroll(landmarks, frame_height)
            
            # Draw active gestures on the frame
            if tap_location:
                cv2.circle(frame, tap_location, 15, (0, 255, 255), -1)
            
            if drag_location:
                cv2.circle(frame, drag_location, 15, (255, 0, 255), -1)
            
            if scroll_location:
                cv2.circle(frame, scroll_location, 15, (255, 255, 0), -1)
            
            # Combine descriptions
            hand_desc = f"{position_desc}, {movement}"
            if gesture:
                hand_desc += f", {gesture}"
            
            # Add special events if detected
            special_events = []
            if tap_event:
                special_events.append(tap_event)
                events.append({'type': 'tap', 'location': tap_location})
            
            if drag_event:
                special_events.append(drag_event)
                events.append({'type': 'drag', 'location': drag_location})
            
            if scroll_event:
                special_events.append(scroll_event)
                events.append({'type': 'scroll', 'location': scroll_location})
            
            if special_events:
                hand_desc += f" - {'; '.join(special_events)}"
                
            descriptions.append(hand_desc)
        
        # Update previous positions for next frame
        self.prev_positions = hands_data.copy()
        
        return " | ".join(descriptions), events

class GestureController:
    """Handles gesture detection and action triggering"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.last_action_time = time.time()
        self.action_cooldown = 0.3  # Seconds between actions
        self.screen_width, self.screen_height = pyautogui.size()
        self.api_response_queue = []
        self.last_recognized_action = "none"
        
        # Start API worker thread
        self.api_thread_running = True
        self.api_thread = threading.Thread(target=self._api_worker)
        self.api_thread.daemon = True
        self.api_thread.start()
    
    def map_camera_to_screen(self, point, camera_width, camera_height):
        """Map camera coordinates to screen coordinates without mirroring"""
        camera_x, camera_y = point
        screen_x = int((camera_x / camera_width) * self.screen_width)
        screen_y = int((camera_y / camera_height) * self.screen_height)
        return (screen_x, screen_y)


    def process_gestures(self, description, events, camera_width, camera_height):
        """Process gestures directly without waiting for DeepSeek API response"""
        current_time = time.time()
        
        # First, look for clear gestures we can process immediately
        for event in events:
            if current_time - self.last_action_time < self.action_cooldown:
                return  # Still in cooldown period
                
            if event['type'] == 'tap' and 'double tap' in description.lower():
                screen_pos = self.map_camera_to_screen(event['location'], camera_width, camera_height)
                self._perform_action('double_click', screen_pos)
                return
                
            if event['type'] == 'tap' and 'tap at' in description.lower():
                screen_pos = self.map_camera_to_screen(event['location'], camera_width, camera_height)
                self._perform_action('click', screen_pos)
                return
                
            if event['type'] == 'scroll' and 'scrolling up' in description.lower():
                self._perform_action('scroll_up')
                return
                
            if event['type'] == 'scroll' and 'scrolling down' in description.lower():
                self._perform_action('scroll_down')
                return
                
            if event['type'] == 'drag' and 'dragging' in description.lower():
                if event['location']:
                    screen_pos = self.map_camera_to_screen(event['location'], camera_width, camera_height)
                    self._perform_action('drag', screen_pos)
                    return
        
        # If no clear gesture was processed, queue for DeepSeek analysis
        # (but only if there are actual hand events to analyze)
        if events and description != "No hands detected":
            # Add to API request queue
            self.api_response_queue.append((description, events))
    
    def _api_worker(self):
        """Worker thread for processing API requests"""
        while self.api_thread_running:
            # Check if there are items in the queue
            if self.api_response_queue:
                description, events = self.api_response_queue.pop(0)
                try:
                    action = self._send_to_deepseek(description)
                    if action != "none" and action != self.last_recognized_action:
                        # Use the last event location of the appropriate type
                        location = None
                        for event in events:
                            if (action == "click" or action == "double_click") and event['type'] == 'tap':
                                location = event['location']
                            elif action == "drag" and event['type'] == 'drag':
                                location = event['location']
                            elif (action == "scroll_up" or action == "scroll_down") and event['type'] == 'scroll':
                                location = event['location']
                        
                        if location:
                            self._perform_action(action, location)
                            self.last_recognized_action = action
                except Exception as e:
                    print(f"API worker error: {e}")
            
            # Sleep to avoid hammering CPU
            time.sleep(0.1)
    
    def _send_to_deepseek(self, description):
        """Send description to DeepSeek API"""
        try:
            # Direct analysis for faster response - simplifying by replacing API call
            # with direct logic for common gestures
            description_lower = description.lower()
            
            if "double tap" in description_lower:
                return "double_click"
            elif "tap at" in description_lower:
                return "click"
            elif "scrolling up" in description_lower:
                return "scroll_up"
            elif "scrolling down" in description_lower:
                return "scroll_down"
            elif "dragging" in description_lower:
                return "drag"
            elif "right" in description_lower and "pinch" in description_lower:
                return "right_click"
            
            # For full API integration, uncomment this code and customize with your API
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Prepare the prompt for DeepSeek
            prompt = f"Analyze this hand gesture description and determine which action to trigger: click, double_click, right_click, drag, scroll_up, scroll_down, or none.\nDescription: {description}"
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 20,
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=1
            )
            
            if response.status_code == 200:
                result = response.json()
                action = result['choices'][0]['message']['content'].strip().lower()
                return action
            
            
            return "none"
            
        except Exception as e:
            print(f"DeepSeek API error: {e}")
            return "none"
    
    def _perform_action(self, action, location=None):
        """Actually perform the system action"""
        print(f"Performing {action} at {location}")
        
        try:
            if action == "click":
                if location:
                    pyautogui.click(location[0], location[1])
                else:
                    pyautogui.click()
                    
            elif action == "double_click":
                if location:
                    pyautogui.doubleClick(location[0], location[1])
                else:
                    pyautogui.doubleClick()
                    
            elif action == "right_click":
                if location:
                    pyautogui.rightClick(location[0], location[1])
                else:
                    pyautogui.rightClick()
                    
            elif action == "drag":
                if location:
                    # Move to the location
                    pyautogui.moveTo(location[0], location[1])
                    # Hold down the mouse button (start drag)
                    pyautogui.mouseDown()
                    # We'll release in a subsequent drag or on explicit command
                
            elif action == "drag_release":
                # Release the mouse button (end drag)
                pyautogui.mouseUp()
                
            elif action == "scroll_up":
                pyautogui.scroll(120)  # Positive for up
                
            elif action == "scroll_down":
                pyautogui.scroll(-120)  # Negative for down
                
            self.last_action_time = time.time()
            
        except Exception as e:
            print(f"Error performing action: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        self.api_thread_running = False
        if self.api_thread.is_alive():
            self.api_thread.join(timeout=1.0)

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FPS, 24)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera running at {actual_fps} FPS")
    
    # Get actual frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Print screen size for debugging
    screen_width, screen_height = pyautogui.size()
    print(f"Screen resolution: {screen_width}x{screen_height}")
    print(f"Camera resolution: {frame_width}x{frame_height}")
    
    # Initialize hand detector
    detector = HandDetector(detection_confidence=0.7, tracking_confidence=0.7)
    
    # Define interaction zones (example)
    detector.add_interaction_zone("Button 1", 50, 50, 100, 50, (0, 255, 0))
    detector.add_interaction_zone("Button 2", 200, 50, 100, 50, (0, 0, 255))
    detector.add_interaction_zone("Scroll Area", frame_width-100, 50, 80, frame_height-100, (255, 0, 0))
    
    # Initialize gesture controller
    api_key = "sk-40a9ca651aa4432aaeac59665841df9f"  # Your DeepSeek API key
    controller = GestureController(api_key)
    
    # Initialize variables
    frame_count = 0
    start_time = time.time()
    
    # Check if GUI is available
    gui_available = True
    try:
        # Try to create a test window
        cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("Test")
        print("GUI features available - will show video feed")
    except:
        gui_available = False
        print("GUI features not available - will run in text-only mode")
    
    # Create output file for text descriptions
    output_file = f"hand_movements_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    print("\n=== Hand Gesture Mouse Control Started ===")
    print("• Point at the screen to move the cursor")
    print("• Pinch index finger and thumb together to click")
    print("• Double pinch for double-click")
    print("• Pinch and move for drag operations")
    print("• Open palm and move up/down to scroll")
    print(f"• Movement and event logs saved to {output_file}")
    print("• Press 'q' to quit, 'f' to toggle fullscreen\n")
    
    # Create full screen window if GUI is available
    if gui_available:
        cv2.namedWindow("Hand Gesture Control", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Hand Gesture Control", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Fullscreen toggle variable
    is_fullscreen = True
    
    try:
        with open(output_file, 'w') as f:
            running = True
            
            while running:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                # Flip the frame for display to match reality
                display_frame = cv2.flip(frame.copy(), 1)  # Flip horizontally for display
                
                # Find hands on the display frame
                display_frame, hands_data = detector.find_hands(display_frame)
                
                # Draw interaction zones
                display_frame = detector.draw_interaction_zones(display_frame)
                
                # Get hand movement description
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                description, events = detector.describe_hand_movements(
                    hands_data, frame_width, frame_height, display_frame)
                full_description = f"[{timestamp}] {description}"
                
                # Process gestures
                controller.process_gestures(description, events, frame_width, frame_height)
                
                # Move cursor to index finger tip if present (for basic pointing)
                if hands_data and events:
                    # Find index finger tip (landmark 8)
                    for hand in hands_data:
                        if hand['type'] == 'Right':  # Usually the dominant hand
                            index_tip = hand['landmarks'][8]
                            # Map to screen coordinates
                            screen_x, screen_y = controller.map_camera_to_screen(
                                index_tip, frame_width, frame_height)
                            # Move cursor smoothly
                            try:
                                pyautogui.moveTo(screen_x, screen_y, duration=0.1)
                            except:
                                pass
                            break
                
                # Write to file and display to console
                f.write(full_description + "\n")
                f.flush()  # Make sure it's written immediately
                print(full_description)
                
                # Add text to display frame
                if gui_available and display_frame is not None:
                    # Add main description at top
                    cv2.putText(display_frame, description, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add FPS counter at bottom
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 1.0:
                        fps = frame_count / elapsed_time
                        frame_count = 0
                        start_time = time.time()
                    else:
                        fps = frame_count / (elapsed_time if elapsed_time > 0 else 1)
                    
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                               (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    
                    # Display active zone if any event is in a zone
                    for event in events:
                        if 'location' in event and event['location']:
                            zone_name = detector.check_point_in_zones(event['location'])
                            if zone_name:
                                cv2.putText(display_frame, f"Active: {zone_name}", 
                                          (frame_width - 200, frame_height - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Show the frame
                    cv2.imshow("Hand Gesture Control", display_frame)
                
                # Check for key presses
                if gui_available:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        running = False
                    elif key == ord('f'):  # Toggle fullscreen
                        is_fullscreen = not is_fullscreen
                        if is_fullscreen:
                            cv2.setWindowProperty("Hand Gesture Control", 
                                                cv2.WND_PROP_FULLSCREEN, 
                                                cv2.WINDOW_FULLSCREEN)
                        else:
                            cv2.setWindowProperty("Hand Gesture Control", 
                                                cv2.WND_PROP_FULLSCREEN, 
                                                cv2.WINDOW_NORMAL)
                else:
                    # Check if we should exit in non-GUI mode
                    if input() == 'q':
                        running = False
    
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if cap.isOpened():
            cap.release()
        if gui_available:
            cv2.destroyAllWindows()
        controller.cleanup()
        print("Program ended")

if __name__ == "__main__":
    main()