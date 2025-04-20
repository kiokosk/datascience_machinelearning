# Import necessary libraries
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize
import cv2


DATA_PATH = 'competition_data/'
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TEST_PATH = os.path.join(DATA_PATH, 'test')
TRAIN_IMAGE_PATH = os.path.join(TRAIN_PATH, 'images')
TRAIN_MASK_PATH = os.path.join(TRAIN_PATH, 'masks')
TEST_IMAGE_PATH = os.path.join(TEST_PATH, 'images')


IMG_HEIGHT = 128  
IMG_WIDTH = 128   
IMG_CHANNELS = 1
BATCH_SIZE = 32
EPOCHS = 30

def unet(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = Input(input_size)
    
    # Contracting Path (Encoder)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bottom
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Expansive Path (Decoder) - Using same padding to ensure dimensions match
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    if input_size[0] > 101 or input_size[1] > 101:
        diff_h = input_size[0] - 101
        diff_w = input_size[1] - 101
        crop_top = diff_h // 2
        crop_bottom = diff_h - crop_top
        crop_left = diff_w // 2
        crop_right = diff_w - crop_left
        outputs = Cropping2D(cropping=((crop_top, crop_bottom), (crop_left, crop_right)))(outputs)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Data loading and preprocessing for TGS dataset
def load_and_preprocess_data():
    # Get the file IDs from directory
    train_ids = [f.split('.')[0] for f in os.listdir(TRAIN_IMAGE_PATH) if f.endswith('.png')]
    test_ids = [f.split('.')[0] for f in os.listdir(TEST_IMAGE_PATH) if f.endswith('.png')]
    
    # Create empty arrays for training data
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    Y_train = np.zeros((len(train_ids), 101, 101, 1), dtype=np.float32)  # Keep original mask size
    
    # Load training images
    print('Loading training images...')
    for n, id_ in enumerate(train_ids):
        # Load image - TGS images are grayscale PNG files
        img_path = os.path.join(TRAIN_IMAGE_PATH, id_ + '.png')
        img = imread(img_path, as_gray=True)
        
        # Resize to our target size
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        img = img.reshape((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        X_train[n] = img / 255.0  # Normalize
        
        # Load mask - TGS masks are also grayscale PNG files
        mask_path = os.path.join(TRAIN_MASK_PATH, id_ + '.png')
        mask = imread(mask_path, as_gray=True)
        
        # Keep masks at original size
        mask = (mask > 0).astype(np.float32)  # Convert to binary
        mask = mask.reshape((101, 101, 1))
        Y_train[n] = mask
    
    # Create empty array for test data
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    
    # Load test images
    print('Loading test images...')
    for n, id_ in enumerate(test_ids):
        img_path = os.path.join(TEST_IMAGE_PATH, id_ + '.png')
        img = imread(img_path, as_gray=True)
        
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        img = img.reshape((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        X_test[n] = img / 255.0  
    
    # Split training data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    
    return X_train, Y_train, X_val, Y_val, X_test, test_ids

def simple_unet(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = Input(input_size)
    
    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bridge
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    # Crop to original size if needed
    if input_size[0] > 101 or input_size[1] > 101:
        diff_h = input_size[0] - 101
        diff_w = input_size[1] - 101
        crop_top = diff_h // 2
        crop_bottom = diff_h - crop_top
        crop_left = diff_w // 2
        crop_right = diff_w - crop_left
        outputs = Cropping2D(cropping=((crop_top, crop_bottom), (crop_left, crop_right)))(outputs)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Run-length encoding for TGS submission format
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() == 1)[0]  # .T is important here
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join([str(x) for x in run_lengths])

# Create submission file for Kaggle
def create_submission(test_ids, preds):
    sub = []
    for i, id_ in enumerate(test_ids):
        # Threshold prediction
        pred = (preds[i] > 0.5).astype(np.uint8)
        # Remove single dimension
        pred = pred.squeeze()
        # RLE encoding
        rle = rle_encoding(pred)
        # Add to submission list
        sub.append([id_, rle])
    
    # Create submission DataFrame
    import pandas as pd
    sub_df = pd.DataFrame(sub, columns=['id', 'rle_mask'])
    sub_df.to_csv('submission.csv', index=False)
    return sub_df

def main():
    # Check for GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Load data
    print("Loading and preprocessing data...")
    X_train, Y_train, X_val, Y_val, X_test, test_ids = load_and_preprocess_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Training masks shape: {Y_train.shape}")
    
    print("Creating U-Net model...")
    model = simple_unet()
    model.summary()
    
    checkpoint = ModelCheckpoint('unet_tgs_salt.h5', 
                                monitor='val_loss', 
                                save_best_only=True, 
                                mode='min', 
                                verbose=1)
    early_stopping = EarlyStopping(patience=10, monitor='val_loss', mode='min', verbose=1)
    callbacks_list = [checkpoint, early_stopping]
    
    # Train the model
    print("Training model...")
    
    history = model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, Y_val),
        callbacks=callbacks_list
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig('training_history.png')
    
    # Load best model for prediction
    print("Loading best model for prediction...")
    model.load_weights('unet_tgs_salt.h5')
    
    # Predict on test data
    print('Predicting on test data...')
    preds_test = model.predict(X_test, verbose=1)
    
    # Create submission file
    print('Creating submission file...')
    submission = create_submission(test_ids, preds_test)
    print('Submission file created: submission.csv')
    
    # Save some example predictions for visualization
    print('Saving  predictions...')
    os.makedirs('predictions', exist_ok=True)
    for i in range(min(10, len(X_val))):
        # Original image
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(X_val[i].squeeze(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # True mask
        plt.subplot(1, 3, 2)
        plt.imshow(Y_val[i].squeeze(), cmap='gray')
        plt.title('True Mask')
        plt.axis('off')
        
        # Predicted mask
        pred = model.predict(X_val[i:i+1])[0]
        plt.subplot(1, 3, 3)
        plt.imshow(pred.squeeze(), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        
        plt.savefig(f'predictions/prediction_{i}.png')
        plt.close()
    
    print('Process completed successfully!')

if __name__ == '__main__':
    main()