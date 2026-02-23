from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from matplotlib.pyplot import imread, imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
#Acne and Rosacea Photos


#Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions

#Atopic Dermatitis Photos

#Bullous Disease Photos
# Define your class names manually (in the same order as training)


from tensorflow.keras.applications import ResNet50
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Define class names manually (make sure they match the original training order)


class_names = [
    "healthy_spiral",
    "healthy_wave",
    "parkinson_spiral",
    "parkinson_wave"
]



# Set up label encoder (if needed for prediction)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(class_names)

num_classes = len(label_encoder.classes_)

# Rebuild EfficientNetB0 model structure exactly as in training
efficientnet_model = EfficientNetB0(input_shape=(75,75, 3), include_top=False, weights='imagenet')

inputs = efficientnet_model.input
conv_output = efficientnet_model.layers[-1].output
x = GlobalAveragePooling2D()(conv_output)

x = Dense(128, kernel_regularizer=l1(0.0001), activation='relu')(x)
x = BatchNormalization(renorm=True)(x)
x = Dropout(0.3)(x)

x = Dense(64, kernel_regularizer=l1(0.0001), activation='relu')(x)
x = BatchNormalization(renorm=True)(x)
x = Dropout(0.3)(x)

x = Dense(32, kernel_regularizer=l1(0.0001), activation='relu')(x)
x = BatchNormalization(renorm=True)(x)
x = Dropout(0.3)(x)

outputs = Dense(num_classes, activation='softmax')(x)

# Combine into the model
model = models.Model(inputs=inputs, outputs=outputs)

# Compile before loading weights
custom_optimizer = RMSprop(learning_rate=0.0001)
model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load weights
model.load_weights("parkinson_model.h5")

print("EfficientNetB0-based model loaded successfully!")





from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

def augment_image2(input_path='input.png', output_path='output.png'):
    # Load the image using OpenCV
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {input_path}")

    # Convert BGR (OpenCV format) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize (optional, if needed by your model or ImageDataGenerator)
    image_rgb = cv2.resize(image_rgb, (75,75))  # adjust size as needed

    # Expand dimensions to match expected input shape: (1, height, width, channels)
    image_array = np.expand_dims(image_rgb, axis=0)

    # Define the ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Generate one augmented image
    aug_iter = datagen.flow(image_array, batch_size=1)
    aug_image = next(aug_iter)[0].astype(np.uint8)

    # Convert to image and save using OpenCV
    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, aug_image_bgr)



# Preprocess the image
def preprocess_single_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    # Resize the image to target size
    image = cv2.resize(image, (75,75))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(clahe_enhanced, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    # Create an RGB edge map (edges in red)
    edges_colored = np.zeros_like(image)
    edges_colored[:, :, 2] = edges
    
    # Overlay the edges onto the original image
    processed_image = cv2.addWeighted(image, 0.8, edges_colored, 0.5, 0)

    cv2.imwrite("static/output_image.png",processed_image)
    
    # Normalize the image (scaling pixel values between 0 and 1)
    processed_image = processed_image / 255.0
    
    return np.expand_dims(processed_image, axis=0)





def pred_skin_disease(img_path):
# Path to the image
            image_path =img_path
            #augment_image2(image_path, 'aug2.png')

            preprocessed_image = preprocess_single_image(image_path)

            # Make prediction
            predictions = model.predict(preprocessed_image)

            # Decode the predicted label
            predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])

            print("full probability",predicted_label,predictions)
            confidence = np.max(predictions)

            print(f"Predicted Label: {predicted_label[0]}, Confidence: {confidence * 100:.2f}%")

            return predicted_label[0]


#pred_sugar_cane("benign-familial-chronic-pemphigus-10.jpg")