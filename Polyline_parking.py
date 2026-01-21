import cv2
import numpy as np
import torch
import json
import os
from collections import deque
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO('yolov8x.pt')
model.to(device)

drawing = False
parking_spaces = []
current_polygon = []
occupied_parking_lots = 0
lot_status = []

# ============================================================================
# TCOD (Temporal Confidence-Based Occupancy Decision) Parameters
# ============================================================================
# λ (lambda): Exponential smoothing factor (0 < λ < 1)
# Higher λ = more weight to previous confidence (smoother, slower response)
# Lower λ = more weight to current detection (faster response, more jitter)
LAMBDA = 0.7  # Sensible default: 70% previous, 30% current

# τ_occ (tau_occupied): Threshold to mark slot as occupied
TAU_OCC = 0.6  # If St > 0.6, mark as occupied

# τ_free (tau_free): Threshold to mark slot as free
TAU_FREE = 0.4  # If St < 0.4, mark as free
# Hysteresis zone: [0.4, 0.6] maintains previous state

# St^i: Confidence value for each parking slot i (initialized when slots are created)
slot_confidences = []  # Will store St for each slot

# ============================================================================
# DPR (Dynamic Polygon Refinement) Parameters
# ============================================================================
# N: Number of frames to track centroid history
N_HISTORY = 30  # Track last 30 frames of vehicle centroids

# α (alpha_refine): Refinement rate for polygon vertex updates
# Lower α = slower, more stable refinement
# Higher α = faster, more aggressive refinement
ALPHA_REFINE = 0.05  # 5% adjustment per refinement step

# ε (epsilon): Maximum allowed drift constraint (pixels)
# Prevents polygon from drifting too far from original position
EPSILON = 20.0  # Max 20 pixels drift from original polygon

# Centroid history: List of deques, one per parking slot
# Each deque stores (cx, cy) tuples of detected vehicle centroids
centroid_history = []  # Will store centroid history for each slot

# Original polygons: Store initial polygon positions for drift constraint
original_polygons = []  # Will store original polygon vertices

# ============================================================================
# Global Resolution Settings
# ============================================================================
# TARGET_WIDTH/HEIGHT defines the resolution for detection and coordinate system
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# ============================================================================
# Evaluation Metrics (for research paper evaluation)
# ============================================================================
# Track stability and performance metrics

import time

# State change tracking: Count how many times each slot changes occupancy state
state_change_count = []  # Will store count for each slot
previous_states = []  # Track previous frame's state for change detection

# Time tracking for Stability Index computation
metrics_start_time = time.time()
total_frames_processed = 0

# Metric logging interval (frames)
METRIC_LOG_INTERVAL = 300  # Log metrics every 300 frames (~60 seconds at 5 FPS)

# ============================================================================
# Slot-Level Analytics (for dashboard/API)
# ============================================================================
# Track detailed timing information per parking slot

from datetime import datetime
import json as json_module

# Analytics data structure for each slot
slot_analytics = []  # List of dicts, one per slot

# Session start time
session_start_time = datetime.now()

def initialize_slot_analytics(slot_id):
    """
    Initialize analytics data structure for a new parking slot.
    Returns a dict with all tracking fields.
    """
    return {
        'slot_id': slot_id,
        'total_occupied_duration': 0.0,  # Total seconds occupied
        'total_free_duration': 0.0,      # Total seconds free
        'current_state': 'FREE',         # Current state: 'OCCUPIED' or 'FREE'
        'last_state_change_time': time.time(),  # Timestamp of last state change
        'state_history': [],  # List of state change events
        'occupancy_events': []  # List of occupancy periods
    }

def update_slot_analytics(slot_id, is_occupied, confidence):
    """
    Update analytics when slot state is checked.
    Tracks durations and timestamps.
    """
    if slot_id >= len(slot_analytics):
        return  # Slot not initialized yet
    
    analytics = slot_analytics[slot_id]
    current_time = time.time()
    current_state = 'OCCUPIED' if is_occupied else 'FREE'
    
    # Calculate duration since last update
    duration = current_time - analytics['last_state_change_time']
    
    # Add duration to appropriate counter
    if analytics['current_state'] == 'OCCUPIED':
        analytics['total_occupied_duration'] += duration
    else:
        analytics['total_free_duration'] += duration
    
    # Check if state changed
    if current_state != analytics['current_state']:
        # Record state change event
        event = {
            'timestamp': datetime.now().isoformat(),
            'from_state': analytics['current_state'],
            'to_state': current_state,
            'confidence': confidence,
            'duration_in_previous_state': duration
        }
        analytics['state_history'].append(event)
        
        # If transitioning to occupied, start new occupancy event
        if current_state == 'OCCUPIED':
            analytics['occupancy_events'].append({
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'duration': None
            })
        # If transitioning to free, close last occupancy event
        elif current_state == 'FREE' and analytics['occupancy_events']:
            last_event = analytics['occupancy_events'][-1]
            if last_event['end_time'] is None:
                last_event['end_time'] = datetime.now().isoformat()
                last_event['duration'] = duration
        
        # Update current state
        analytics['current_state'] = current_state
    
    # Update last change time
    analytics['last_state_change_time'] = current_time

def get_analytics_summary():
    """
    Get analytics summary for all slots in JSON-serializable format.
    Suitable for dashboard/API responses.
    """
    session_duration = (datetime.now() - session_start_time).total_seconds()
    
    summary = {
        'session_start': session_start_time.isoformat(),
        'session_duration_seconds': session_duration,
        'total_slots': len(slot_analytics),
        'slots': []
    }
    
    for analytics in slot_analytics:
        # Calculate utilization rate
        total_time = analytics['total_occupied_duration'] + analytics['total_free_duration']
        utilization = (analytics['total_occupied_duration'] / total_time * 100) if total_time > 0 else 0
        
        slot_summary = {
            'slot_id': analytics['slot_id'],
            'current_state': analytics['current_state'],
            'total_occupied_duration_seconds': round(analytics['total_occupied_duration'], 2),
            'total_free_duration_seconds': round(analytics['total_free_duration'], 2),
            'utilization_percentage': round(utilization, 2),
            'total_state_changes': len(analytics['state_history']),
            'total_occupancy_events': len(analytics['occupancy_events']),
            'recent_events': analytics['state_history'][-5:] if analytics['state_history'] else []  # Last 5 events
        }
        summary['slots'].append(slot_summary)
    
    return summary

def export_analytics_json(filename='slot_analytics.json'):
    """
    Export analytics to JSON file for external processing.
    """
    try:
        summary = get_analytics_summary()
        with open(filename, 'w') as f:
            json_module.dump(summary, f, indent=2)
        print(f"✓ Exported analytics to {filename}")
        return True
    except Exception as e:
        print(f"✗ Error exporting analytics: {e}")
        return False

def draw_polygon(event, x, y, flags, param):
    global drawing, current_polygon, lot_status, slot_confidences, centroid_history, original_polygons
    global state_change_count, previous_states, slot_analytics

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_polygon = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_polygon.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_polygon.append((x, y))
        polygon_array = np.array(current_polygon, np.int32)
        parking_spaces.append(polygon_array)
        lot_status.append(False)
        # Initialize confidence St^i = 0.0 for new slot
        slot_confidences.append(0.0)
        # Initialize centroid history (deque with maxlen=N_HISTORY)
        from collections import deque
        centroid_history.append(deque(maxlen=N_HISTORY))
        # Store original polygon for drift constraint
        original_polygons.append(polygon_array.copy())
        # Initialize evaluation metrics for new slot
        state_change_count.append(0)
        previous_states.append(False)
        # Initialize slot-level analytics
        slot_id = len(parking_spaces) - 1
        slot_analytics.append(initialize_slot_analytics(slot_id))
        current_polygon = []
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Delete specific polygon on right click
        for i, space in enumerate(parking_spaces):
            if cv2.pointPolygonTest(space, (x, y), False) >= 0:
                print(f"🗑️ Deleting parking slot {i+1}")
                parking_spaces.pop(i)
                lot_status.pop(i)
                slot_confidences.pop(i)
                centroid_history.pop(i)
                original_polygons.pop(i)
                state_change_count.pop(i)
                previous_states.pop(i)
                slot_analytics.pop(i)
                # Re-index analytics slot IDs
                for idx, analytics in enumerate(slot_analytics):
                    analytics['slot_id'] = idx
                break

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# ============================================================================
# Persistent Storage Functions
# ============================================================================

def get_polygon_filename(video_source):
    """Derive a video-specific filename for polygons."""
    if isinstance(video_source, int):
        return f"parking_polygons_webcam_{video_source}.json"
    base = os.path.basename(video_source)
    return f"parking_polygons_{base}.json"

def save_polygons(filename=None):
    """
    Save parking space polygons to JSON file.
    """
    global parking_spaces, original_polygons, video_path
    
    if filename is None:
        filename = get_polygon_filename(video_path)
    
    if len(parking_spaces) == 0:
        print("No parking spaces to save.")
        return
    
    data = {
        "polygons": [
            {
                "id": i,
                "vertices": space.tolist()  # Convert numpy array to list
            }
            for i, space in enumerate(parking_spaces)
        ]
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved {len(parking_spaces)} parking spaces to {filename}")
    except Exception as e:
        print(f"✗ Error saving polygons: {e}")

def load_polygons(filename=None):
    """
    Load parking space polygons from JSON file.
    Returns True if successful, False otherwise.
    """
    global parking_spaces, lot_status, slot_confidences, centroid_history, original_polygons, video_path
    
    if filename is None:
        filename = get_polygon_filename(video_path)
    
    if not os.path.exists(filename):
        print(f"No saved polygons found at {filename}")
        return False
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Clear existing data
        parking_spaces.clear()
        lot_status.clear()
        slot_confidences.clear()
        centroid_history.clear()
        original_polygons.clear()
        
        # Load polygons from JSON
        for poly_data in data["polygons"]:
            vertices = np.array(poly_data["vertices"], dtype=np.int32)
            parking_spaces.append(vertices)
            lot_status.append(False)
            slot_confidences.append(0.0)
            centroid_history.append(deque(maxlen=N_HISTORY))
            original_polygons.append(vertices.copy())
            
            # Initialize slot-level analytics for loaded slots
            slot_id = len(parking_spaces) - 1
            slot_analytics.append(initialize_slot_analytics(slot_id))
            state_change_count.append(0)
            previous_states.append(False)
        
        print(f"✓ Loaded {len(parking_spaces)} parking spaces from {filename}")
        
        # SCALE POLYGONS IF NEEDED (from legacy 640x360 to standard TARGET_WIDTH/HEIGHT)
        # Check if polygons seem to be in 640x360 space (max coord < 1000)
        # and scale them to our target resolution
        if len(parking_spaces) > 0:
            # Check a sample point to estimate current resolution
            sample_pt = parking_spaces[0][0]
            # If the point is clearly in a smaller coordinate space, scale it
            if sample_pt[0] < 800 and sample_pt[1] < 500:
                print(f"ℹ Scaling polygons from legacy 640x360 to {TARGET_WIDTH}x{TARGET_HEIGHT}...")
                scale_x = TARGET_WIDTH / 640.0
                scale_y = TARGET_HEIGHT / 360.0
                
                for i in range(len(parking_spaces)):
                    parking_spaces[i] = (parking_spaces[i].astype(float) * [scale_x, scale_y]).astype(np.int32)
                    original_polygons[i] = (original_polygons[i].astype(float) * [scale_x, scale_y]).astype(np.int32)
            
        return True
        
    except Exception as e:
        print(f"✗ Error loading polygons: {e}")
        return False

# ============================================================================
# Evaluation Metric Functions
# ============================================================================

def compute_stability_index():
    """
    Compute Stability Index (SI) for each parking slot.
    SI = (number of state changes) / elapsed time (seconds)
    Lower SI = more stable (fewer false positives/flickering)
    """
    elapsed_time = time.time() - metrics_start_time
    if elapsed_time == 0:
        return [0.0] * len(state_change_count)
    
    stability_indices = [
        changes / elapsed_time for changes in state_change_count
    ]
    return stability_indices

def print_evaluation_metrics():
    """
    Print comprehensive evaluation metrics for research paper.
    Includes:
    - Stability Index per slot
    - Total state changes
    - Average confidence values
    - Frame processing rate
    """
    elapsed_time = time.time() - metrics_start_time
    fps = total_frames_processed / elapsed_time if elapsed_time > 0 else 0
    
    stability_indices = compute_stability_index()
    
    print("\n" + "="*70)
    print("EVALUATION METRICS (Research Paper Data)")
    print("="*70)
    print(f"Elapsed Time:        {elapsed_time:.2f} seconds")
    print(f"Total Frames:        {total_frames_processed}")
    print(f"Processing FPS:      {fps:.2f}")
    print(f"Total Parking Slots: {len(parking_spaces)}")
    print()
    
    if len(parking_spaces) > 0:
        print("Per-Slot Metrics:")
        print(f"{'Slot':<6} {'State Changes':<15} {'Stability Index':<18} {'Avg Confidence':<16} {'Current State'}")
        print("-" * 70)
        
        for i in range(len(parking_spaces)):
            avg_conf = slot_confidences[i]
            si = stability_indices[i]
            state = "OCCUPIED" if lot_status[i] else "FREE"
            
            print(f"{i+1:<6} {state_change_count[i]:<15} {si:<18.4f} {avg_conf:<16.3f} {state}")
        
        print()
        print("Aggregate Metrics:")
        total_changes = sum(state_change_count)
        avg_si = sum(stability_indices) / len(stability_indices) if stability_indices else 0
        avg_conf_all = sum(slot_confidences) / len(slot_confidences) if slot_confidences else 0
        
        print(f"  Total State Changes:     {total_changes}")
        print(f"  Average Stability Index: {avg_si:.4f} changes/sec")
        print(f"  Average Confidence:      {avg_conf_all:.3f}")
        print(f"  False Occupancy Rate:    {(total_changes / total_frames_processed * 100):.2f}% (flickering indicator)")
    
    print("="*70 + "\n")

video_path = "/Users/suryodaypratapsingh/Desktop/SEM-6/Signals/1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cv2.namedWindow('Parking Detection')
cv2.setMouseCallback('Parking Detection', draw_polygon)

# ============================================================================
# Auto-load saved polygons on startup
# ============================================================================
print("\n" + "="*60)
print("PARKING DETECTION SYSTEM")
print("="*60)
load_polygons()  # Attempt to load saved polygons
print("Controls:")
print("  - Click and drag to draw parking spaces")
print("  - Press 's' to save parking spaces")
print("  - Press SPACE to quit")
print("="*60 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Retrying...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Increase resolution for better distant car detection
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    # imgsz=1280 allows YOLO to see smaller details
    # conf=0.15 allows catching faint/distant detections (TCOD will filter noise)
    results = model.predict(frame, device=device, imgsz=1280, conf=0.15, verbose=False)

    boxes = results[0].boxes.data.cpu().numpy()

    occupied_parking_lots = 0

    for i, space in enumerate(parking_spaces):
        # ====================================================================
        # Step 1: Calculate instantaneous detection score st for this frame
        # ====================================================================
        # st is derived from:
        # - Vehicle detection confidence (from YOLO)
        # - Polygon inclusion (whether vehicle center is in parking slot)
        
        max_detection_score = 0.0  # st starts at 0 (no vehicle detected)
        vehicle_detected = False   # Track if ANY vehicle was detected in this slot
        detected_centroid = None   # Track centroid of detected vehicle for DPR
        
        for box in boxes:
            # Extract YOLO detection: x1, y1, x2, y2, confidence, class_id
            x1, y1, x2, y2, conf, cls_id = box[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(cls_id)
            class_name = model.names[cls_id]

            # Only consider vehicle classes
            if class_name in ['car', 'truck', 'bus']:
                # Calculate vehicle center point
                car_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Check if vehicle center is inside parking slot polygon
                if point_in_polygon(car_center, space):
                    # st = detection confidence (from YOLO) if vehicle is in slot
                    # Use max confidence if multiple vehicles detected in same slot
                    if conf > max_detection_score:
                        max_detection_score = conf
                        vehicle_detected = True
                        # Store centroid for DPR (use vehicle with highest confidence)
                        detected_centroid = car_center
        
        # ====================================================================
        # Step 2: Apply Exponential Smoothing to update St^i
        # ====================================================================
        # Formula: St = λ * S(t-1) + (1-λ) * st
        # Where:
        #   St = new confidence for slot i at time t
        #   S(t-1) = previous confidence (slot_confidences[i])
        #   st = instantaneous detection score (max_detection_score)
        #   λ = LAMBDA (smoothing factor)
        
        S_prev = slot_confidences[i]  # S(t-1)
        
        # ====================================================================
        # OCCLUSION-ROBUST HANDLING
        # ====================================================================
        # Problem: If no vehicle detected (occlusion, passing person, etc.),
        #          setting st=0 causes instant confidence drop
        # Solution: Use decay factor instead of hard 0 when no detection
        #
        # If vehicle detected:
        #   st = detection confidence (normal case)
        # If NO vehicle detected (occlusion):
        #   st = S(t-1) * decay_factor (gradual decay, not instant drop)
        #
        # This allows system to "remember" occupancy through brief occlusions
        # ====================================================================
        
        if vehicle_detected:
            # Normal case: vehicle detected, use YOLO confidence
            st = max_detection_score
        else:
            # Occlusion case: no vehicle detected
            # Instead of st=0 (instant drop), decay previous confidence
            # decay_factor = 0.9 means confidence drops by 10% per frame
            # This gives ~7 frames of "memory" before confidence halves
            decay_factor = 0.9
            st = S_prev * decay_factor
        
        # Exponential smoothing update
        St = LAMBDA * S_prev + (1 - LAMBDA) * st
        
        # Update stored confidence
        slot_confidences[i] = St
        
        # ====================================================================
        # Step 3: Apply Hysteresis to determine occupancy state
        # ====================================================================
        # Hysteresis prevents flickering by maintaining state in uncertain zone
        # Rules:
        #   If St > τ_occ (0.6): Mark as OCCUPIED
        #   If St < τ_free (0.4): Mark as FREE
        #   If τ_free <= St <= τ_occ: MAINTAIN previous state (hysteresis)
        
        if St > TAU_OCC:
            # High confidence → Occupied
            lot_occupied = True
        elif St < TAU_FREE:
            # Low confidence → Free
            lot_occupied = False
        else:
            # Hysteresis zone → Keep previous state
            lot_occupied = lot_status[i]
        
        # Update state and count
        lot_status[i] = lot_occupied
        if lot_occupied:
            occupied_parking_lots += 1
            color = (0, 0, 255)  # Red for occupied
        else:
            color = (0, 255, 0)  # Green for free
        
        # ====================================================================
        # Evaluation Metrics: Track State Changes
        # ====================================================================
        # Detect if state changed from previous frame
        if i < len(previous_states):
            if lot_occupied != previous_states[i]:
                # State changed! Increment counter
                state_change_count[i] += 1
            # Update previous state for next frame
            previous_states[i] = lot_occupied
        
        # ====================================================================
        # Slot-Level Analytics: Update Duration Tracking
        # ====================================================================
        update_slot_analytics(i, lot_occupied, St)
        
        # ====================================================================
        # Step 4: Dynamic Polygon Refinement (DPR)
        # ====================================================================
        # Track vehicle centroids and refine polygon position over time
        # Only refine when slot is stably occupied (St > TAU_OCC)
        # ====================================================================
        
        # Track centroid if vehicle was detected
        if detected_centroid is not None:
            centroid_history[i].append(detected_centroid)
        
        # Refine polygon if slot is stably occupied and we have enough history
        if St > TAU_OCC and len(centroid_history[i]) >= N_HISTORY:
            # ================================================================
            # Compute mean centroid (Cx_mean, Cy_mean) from history
            # ================================================================
            centroids_array = np.array(list(centroid_history[i]))
            Cx_mean = np.mean(centroids_array[:, 0])
            Cy_mean = np.mean(centroids_array[:, 1])
            
            # ================================================================
            # Update polygon vertices using refinement formula:
            #   xj(new) = xj + α * (Cx_mean - xj)
            #   yj(new) = yj + α * (Cy_mean - yj)
            # Where:
            #   xj, yj = current vertex coordinates
            #   Cx_mean, Cy_mean = mean centroid from history
            #   α = ALPHA_REFINE (refinement rate, 0.05 = 5% adjustment)
            # ================================================================
            
            new_polygon = space.copy().astype(np.float32)
            
            for j in range(len(new_polygon)):
                xj, yj = new_polygon[j]
                
                # Apply refinement formula
                xj_new = xj + ALPHA_REFINE * (Cx_mean - xj)
                yj_new = yj + ALPHA_REFINE * (Cy_mean - yj)
                
                new_polygon[j] = [xj_new, yj_new]
            
            # ================================================================
            # Enforce drift constraint: ||P(new) - P(original)|| < ε
            # ================================================================
            # Compute maximum distance between new and original polygon
            # If drift exceeds EPSILON, reject the refinement
            # ================================================================
            
            original_poly = original_polygons[i].astype(np.float32)
            max_drift = 0.0
            
            for j in range(len(new_polygon)):
                # Euclidean distance between new and original vertex
                drift = np.sqrt(
                    (new_polygon[j][0] - original_poly[j][0])**2 +
                    (new_polygon[j][1] - original_poly[j][1])**2
                )
                max_drift = max(max_drift, drift)
            
            # Only apply refinement if drift is within bounds
            if max_drift < EPSILON:
                # Update parking space polygon
                parking_spaces[i] = new_polygon.astype(np.int32)
                space = parking_spaces[i]  # Update local reference
            # else: reject refinement, polygon drifted too far

        # ====================================================================
        # Visualization Overlays (for debugging and research figures)
        # ====================================================================
        # Does not affect logic - purely visual feedback
        # ====================================================================
        
        # Draw polygon with color-coded occupancy state
        cv2.polylines(frame, [space], isClosed=True, color=color, thickness=2)
        
        # Calculate polygon centroid for text placement
        M = cv2.moments(space)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # ================================================================
            # Overlay 1: Slot ID and Confidence Value
            # ================================================================
            # Format: "ID (confidence)"
            # Example: "1 (0.85)" means slot 1 with 85% confidence
            # ================================================================
            
            slot_id = i + 1  # 1-indexed for user display
            confidence_display = f"{St:.2f}"  # Confidence in range [0.00, 1.00]
            label = f"{slot_id} ({confidence_display})"
            
            # Calculate text size for background box
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Position text at polygon centroid
            text_x = cx - text_width // 2
            text_y = cy + text_height // 2
            
            # Draw semi-transparent background for readability
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (text_x - 5, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + baseline + 5),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw text (white for high contrast)
            cv2.putText(frame, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if len(current_polygon) > 1:
        cv2.polylines(frame, [np.array(current_polygon, np.int32)], isClosed=False, color=(255, 255, 0), thickness=2)

    total_parking_lots = len(parking_spaces)

    cv2.rectangle(frame, (10, 10), (230, 50), (0, 0, 0), -1)

    cv2.putText(frame, f"Parking Lots: {occupied_parking_lots}/{total_parking_lots}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Parking Detection', frame)
    
    # ========================================================================
    # Evaluation Metrics: Frame Counter and Periodic Logging
    # ========================================================================
    total_frames_processed += 1
    
    # Log metrics periodically for research paper evaluation
    if total_frames_processed % METRIC_LOG_INTERVAL == 0:
        print_evaluation_metrics()

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        # Quit
        break
    elif key == ord('s'):
        # Save parking spaces
        save_polygons()
    elif key == ord('a'):
        # Export analytics to JSON
        export_analytics_json()
    elif key == ord('m'):
        # Print evaluation metrics immediately
        print_evaluation_metrics()
    elif key == ord('c'):
        # Clear all polygons
        print("🧹 Clearing all parking spaces...")
        parking_spaces.clear()
        lot_status.clear()
        slot_confidences.clear()
        centroid_history.clear()
        original_polygons.clear()
        state_change_count.clear()
        previous_states.clear()
        slot_analytics.clear()
    elif key == ord('u'):
        # Undo last polygon
        if parking_spaces:
            print(f"↩️ Undoing last parking slot (Slot {len(parking_spaces)})")
            parking_spaces.pop()
            lot_status.pop()
            slot_confidences.pop()
            centroid_history.pop()
            original_polygons.pop()
            state_change_count.pop()
            previous_states.pop()
            slot_analytics.pop()

cap.release()
cv2.destroyAllWindows()