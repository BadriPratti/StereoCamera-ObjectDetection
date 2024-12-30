import cv2

def annotate_frame(frame, x1, y1, x2, y2, label, depth, real_width, real_height):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    depth_text = f"{label} {depth:.2f} m" if depth != float('inf') else f"{label} far away"
    width_text = f"Width: {real_width:.2f} m" if real_width != float('inf') else "Width: far away"
    height_text = f"Height: {real_height:.2f} m" if real_height != float('inf') else "Height: far away"

    cv2.putText(frame, depth_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, width_text, (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, height_text, (x1, y1 - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
