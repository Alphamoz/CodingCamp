import cv2
img = cv2.imread("photo.jpg")
print(img.shape)

img = cv2.GaussianBlur(img, (9,9), 0)
green = (0, 255, 0)
blue = (255, 0, 0)
red = (0, 0, 255)

cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), green, 3)
cv2.line(img, (0, 500), (500, 0), red, 3)
cv2.rectangle(img, (350, 290), (400, 350), blue, 3)
cv2.circle(img, (380, 130), 100, (255, 255, 255), 3)

cv2.putText(img, 'Person', (300, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 2)
cv2.putText(img, 'Watch', (340, 270), cv2.FONT_HERSHEY_COMPLEX,1, (0, 0, 250), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()