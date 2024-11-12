import cv2
import numpy as np


image_name = input("Enter the image name: ")
image = cv2.imread('game_images/' + image_name)


if image is None:
    print(f"Error: Could not load image at game_images/{image_name}")
else:

    cv2.imshow('Original Image', image)
    cv2.waitKey(0)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    black_lower = np.array([0, 90, 0])
    black_upper = np.array([180, 100, 20])
    black_mask = cv2.inRange(hsv, black_lower, black_upper)

    cv2.imshow('Black Mask', black_mask)
    cv2.waitKey(0)

    white_lower = np.array([35, 50, 165])  
    white_upper = np.array([45, 150, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    cv2.imshow('White Mask', white_mask)
    cv2.waitKey(0)

    # combined_mask = cv2.bitwise_or(black_mask, white_mask)
    # cv2.imshow('Combined Mask (Black and White Pieces)', combined_mask)
    # cv2.waitKey(0)


    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    # cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('Cleaned Mask after Morphology', cleaned_mask)
    # cv2.waitKey(0)

    # inpainted_image = cv2.inpaint(image, cleaned_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # cv2.imshow('Inpainted Image', inpainted_image)
    # cv2.waitKey(0)
    # _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('Initial Threshold Mask', thresh)
    # cv2.waitKey(0)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    # cv2.imshow('Cleaned Mask after Morphology', mask_cleaned)
    # cv2.waitKey(0)

    # contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # mask_for_inpainting = np.zeros_like(gray)
    # cv2.drawContours(mask_for_inpainting, contours, -1, (255), thickness=cv2.FILLED)
    # cv2.imshow('Mask for Inpainting', mask_for_inpainting)
    # cv2.waitKey(0)

    # inpainted_image = cv2.inpaint(image, mask_for_inpainting, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # cv2.imshow('Inpainted Image', inpainted_image)
    # cv2.waitKey(0)

    # lwr = np.array([0, 0, 143])
    # upr = np.array([179, 61, 252])
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # msk = cv2.inRange(hsv, lwr, upr)

    # cv2.imshow('Mask', msk)
    # cv2.waitKey(0)

    # blurred = cv2.GaussianBlur(msk, (5, 5), 0)
    # cv2.imshow('Blurred Image', blurred)
    # cv2.waitKey(0)


    # krn = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    # # msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, krn)
    # dlt = cv2.dilate(msk, krn, iterations=5)
    # res = 255 - cv2.bitwise_and(dlt, msk)

    # cv2.imshow('Chessboard Extraction', res)
    # cv2.waitKey(0)

    # res = np.uint8(res)
    # ret, corners = cv2.findChessboardCornersSB(res, (8, 8),
    #                                            flags=cv2.CALIB_CB_EXHAUSTIVE)
    # if ret:
    #     print("Corners detected:")
    #     print(corners)
    #     fnl = cv2.drawChessboardCorners(image, (8, 8), corners, ret)
    #     cv2.imshow("Final Chessboard Corners", fnl)
    #     cv2.waitKey(0)
    # else:
    #     print("No Chessboard Found")

    cv2.destroyAllWindows()