import cv2

image = cv2.imread('game_images/game_2/1.png')
points = []

def capture_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Corners", image)
        
        if len(points) == 4:
            cv2.destroyAllWindows()
            print("Selected Points:")
            for p in points:
                print(f"        [{p[0]}, {p[1]}],")

cv2.imshow("Select Corners", image)
cv2.setMouseCallback("Select Corners", capture_points)

print("Please click on the four corners of the chessboard in the following order:")
print("Top-left, Top-right, Bottom-right, Bottom-left")

cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) == 4:
    print("The selected corner points are:")
    for p in points:
        print(f"        [{p[0]}, {p[1]}],")
else:
    print("You did not select exactly 4 points.")