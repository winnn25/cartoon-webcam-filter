import cv2
import numpy as np

# Start video capture from the default webcam (0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # --- Step 3: Smooth color image with bilateral filter ---
    color = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    # --- Step 4: Create edge mask ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=9, C=2)

    # --- Step 5: Combine smoothed color with edges ---
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_colored)

    # --- Step 6: Show output ---
    cv2.imshow('Cartoon Feed', cartoon)
    # Optional: also show original frame
    # cv2.imshow('Original', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

