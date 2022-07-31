import numpy as np
from keras.preprocessing import image
# Keras VGG16
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# http://kikei.github.io/ai/2018/08/05/vgg16.html

class Image2Text(object):
    def __init__(self):
        print('creating the estimating model...')
        self.model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
        print('the model have been created!')

    def __load_image(self, filename):
        img = image.load_img(filename, target_size=(224, 224))
        return image.img_to_array(img)        

    def predict(self, filename):
        inputs = np.zeros((1, 224, 224, 3))
        inputs[0] = self.__load_image(filename)

        pred = self.model.predict(preprocess_input(inputs))

        top = decode_predictions(pred, top=5)   # トップ10を表示
        for j in range(0, len(top[0])):
            name, desc, score = top[0][j]
            print('  {rank}    {desc} {score:02.1f}%'
                .format(rank=j+1, desc=desc, score=score*100))
        name, desc, score = top[0][0]
        return desc