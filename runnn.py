from keras.models import load_model
from skimage.transform import resize, pyramid_reduce
import numpy as np
import cv2

# model = load_model('Sign-Language-Recognition\\CNNmodel.h5')
model = load_model('CNNmodel.h5')

def get_square(image, square_size):

    height, width = image.shape
    if(height > width):
      differ = height
    else:
      differ = width
    differ += 4


    mask = np.zeros((differ, differ), dtype = "uint8")

    x_pos = int((differ - width) / 2)
    y_pos = int((differ - height) / 2)


    mask[y_pos: y_pos + height, x_pos: x_pos + width] = image[0: height, 0: width]


    if differ / square_size > 1:
      mask = pyramid_reduce(mask, differ / square_size)
    else:
      mask = cv2.resize(mask, (square_size, square_size), interpolation = cv2.INTER_AREA)
    return mask

def prediction(pred):
    return(chr(pred+ 65))


def keras_predict(model, image):
    data = np.asarray( image, dtype="int32" )

    pred_probab = model.predict(data)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def keras_process_image(img):

    image_x = 28
    image_y = 28
    img = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
    # img = get_square(img, 28)
    # img = np.reshape(img, (image_x, image_y))

    return img


def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

cam_capture = cv2.VideoCapture(0)
while True:
    # capturing the image from webcam
    _, image_frame = cam_capture.read()
    # (480, 640, 3)
    # to crop required part
    
    im2 = crop_image(image_frame, 90,160,300,300)

    # convert to grayscale
    image_grayscale = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # blurring the image
    image_grayscale_blurred =cv2.GaussianBlur(image_grayscale, (15,15), 0)

    # resize the image to 28x28
    im3 = cv2.resize(image_grayscale_blurred, (28,28), interpolation = cv2.INTER_AREA)

    # expand the dimensions from 28x28 to 1x28x28x1
    im4 = np.resize(im3, (28, 28, 1))
    im5 = np.expand_dims(im4, axis=0)

    pred_probab, pred_class = keras_predict(model, im5)
    curr = prediction(pred_class)
    cv2.putText(image_frame, curr, (400, 320), cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.rectangle(image_frame, (90, 160), (390, 460), (255, 255, 00), 3)
    cv2.imshow("frame",image_frame)
            
    # cv2.imshow("Image2",im2)
    cv2.imshow("Image4",im4)
    cv2.imshow("Image3",image_grayscale_blurred)

    if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

cam_capture.release()

