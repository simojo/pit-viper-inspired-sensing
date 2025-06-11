import cv2
import apriltag
import matplotlib.pyplot as plt
import np
# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(frame_width)
print(frame_height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))


# FOR NOW JUST COPIED THIS SHOULD BE IMPORTED FROM A SHARED FILE
def calculate_heading_angle(centroid, image_shape, camera_fov_y, april_tag_length_pixels):
    """
    Calculate the heading angle to turn the robot towards the centroid in the image.
    
    Parameters:
    - centroid: (cx, cy) - the centroid coordinates of the hot area.
    - image_shape: (height, width) - the dimensions of the image (height, width).
    - camera_fov_x: float - the horizontal field of view of the camera in degrees.
    
    Returns:
    - heading_angle: float - the angle (in degrees) the robot needs to turn to face the centroid.
    """

    # NOTE REMEMBER THAT X IS UP DOWN AND Y IS SIDE IN CAMERA SPACE
    
    # Get the center of the image (cx_center, cy_center)
    print("Centroid:", centroid[1])
    cx_center, cy_center = image_shape[0] // 2, image_shape[1] // 2  # image shape: (height, width)
    
    # Get the displacement from the center to the centroid
    delta_y = centroid[1] - cy_center  # horizontal displacement in pixels
    
    print("Center of image:", cx_center, cy_center)

    # Calculate the angle per pixel
    angle_per_pixel = camera_fov_y / image_shape[1]  # field of view in degrees per pixel
    
    # Calculate the heading angle in degrees (relative to the center of the image)
    heading_angle = delta_y * angle_per_pixel

    # tag_image_size = abs(centroid[0] - cx_center)  # approximate horizontal image size of the tag
    # caculate
    focal_length = 1920 #2.2/1000 # 2.2mm
    tag_size = 3/100 #3 cm
    distance = (tag_size * focal_length) / april_tag_length_pixels  # distance in meters
    
    return distance, heading_angle



while True:
    ret, frame = cam.read()

    # Write the frame to the output file
    # out.write(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(frame)
    plt.show()
    print("[INFO] detecting AprilTags...")
    options = apriltag.DetectorOptions(families="tag36h11", refine_edges=False, quad_contours=False)
    detector = apriltag.Detector(options)
    results = detector.detect(gray) # This returns the number of april tags detected
    # Check properties
    print(f"Width: {frame_width}, Height: {frame_height}")
    print(f"Exposure: {cam.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"Brightness: {cam.get(cv2.CAP_PROP_BRIGHTNESS)}")


    

    # Calculate the Euclidean distance between the two adjacent corners
    # Base case: just one april tag want to steer towards it
    for r in results:
        (ptA, ptB, ptC, ptD) = r.corners
        print( r.corners)
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        corner1 = np.array(ptB)  # Corner 1 (x1, y1)
        corner2 = np.array(ptA)  # Corner 2 (x2, y2)
        print(f"April tag is at {(int(r.center[1]), int(r.center[0]))}")
        centroid =(int(r.center[1]), int(r.center[0])) # the centroid reads out x as y in camera frame
        image_shape = (frame_height, frame_width) # 1080x1920 in pixel coordaintes 
        camera_fov_y = 77

        # FROM THIS WE CAN CALIBRATE THIS MANUALY TO SCALE I THINK THE MATH IS NOT CORRECT CURRENTLY
        # NOT ROTATION INVARIANT YET
        april_tag_length_pixels = np.linalg.norm(corner1 - corner2)
        print("Pixel  length:", april_tag_length_pixels)

        distance, heading_angle = calculate_heading_angle(centroid, image_shape, camera_fov_y, april_tag_length_pixels)
        print(f"The robot needs to turn {heading_angle:.2f} degrees to face the centroid.")
        print(f"Distance {distance:.2f} in m.")




    break

    print(results)
    print("[INFO] {} total AprilTags detected".format(len(results)))
    image = frame
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("[INFO] tag family: {}".format(tagFamily))
        # show the output image after AprilTag detection
        cv2.imshow("Image", image)

        

    break
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
