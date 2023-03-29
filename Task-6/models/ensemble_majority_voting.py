import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
from pathlib import Path

ROOT_DIR = '../results/models/'


def majority_voting(custom_prediction, vgg_prediction):
    """ Find the majority vote among the predictions of the given models

    Args:
        custom_prediction (numpy array): predictions of custom CNN model.

        vgg_prediction (numpy array): predictions of CNN-VGG16 model.

    Returns:
        (numpy array): predictions of ensemble-majority voting model

    """
    # loop over all predictions
    final_prediction = list()
    for custom, vgg in zip(custom_prediction,
                                   vgg_prediction):
        # Keep track of votes per class
        br = e = h = lb = 0

        # Loop over all models
        image_predictions = [custom, vgg]
        for img_prediction in image_predictions:
            # Voting
            if img_prediction == 'Black_rot':
                br += 1
            elif img_prediction == 'Esca':
                e += 1
            elif img_prediction == 'healthy':
                h += 1
            elif img_prediction == 'Leaf_blight':
                lb += 1

        # Find max vote
        count_dict = {'br': br, 'e': e, 'h': h, 'lb': lb}
        highest = max(count_dict.values())
        max_values = [k for k, v in count_dict.items() if v == highest]
        ensemble_prediction = []
        for max_value in max_values:
            if max_value == 'br':
                ensemble_prediction.append('Black_rot')
            elif max_value == 'e':
                ensemble_prediction.append('Esca')
            elif max_value == 'h':
                ensemble_prediction.append('healthy')
            elif max_value == 'lb':
                ensemble_prediction.append('Leaf_blight')

        predict = ''
        if len(ensemble_prediction) > 1:
            predict = custom
        else:
            predict = ensemble_prediction[0]

        # Store max vote
        final_prediction.append(predict)

    return np.array(final_prediction)


def main():
    """ Load data.
    Normalize and encode.
    Train ensemble-majority voting model.
    Print accuracy of the model.
    """

    test_dir = Path('../dataset/test')
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(180, 180),
            color_mode="rgb",
            shuffle = False,
            class_mode='categorical',
            batch_size=1)

    filenames = test_generator.filenames
    nb_samples = len(filenames)
    labels = test_generator.labels
    class_dict= test_generator.class_indices
    print (class_dict) # have a look at the dictionary
    new_dict={} 
    for key in class_dict: # set key in new_dict to value in class_dict and value in new_dict to key in class_dict
        value=class_dict[key]
        new_dict[value]=key
    # print(new_dict)
    try:
        custom_model = load_model(os.path.join(ROOT_DIR, 'custom.h5'))
        vgg_model = load_model(os.path.join(ROOT_DIR, 'vgg16.h5'))

        custom_prediction = custom_model.predict(test_generator,steps = nb_samples)
        vgg_prediction = vgg_model.predict(test_generator,steps = nb_samples)

        np1 = []
        np2 = []
        for i, p in enumerate(custom_prediction):
            pred_index=np.argmax(p) # get the index that has the highest probability
            pred_class=new_dict[pred_index]  # find the predicted class based on the index
            true_class=new_dict[labels[i]] # use the test label to get the true class of the test file
            file=filenames[i]
            np1.append(pred_class)
            # print(f'    {pred_class}       {true_class}       {file}')
        
        for i, p in enumerate(vgg_prediction):
            pred_index=np.argmax(p) # get the index that has the highest probability
            pred_class=new_dict[pred_index]  # find the predicted class based on the index
            true_class=new_dict[labels[i]] # use the test label to get the true class of the test file
            file=filenames[i]
            np2.append(pred_class)

        # print(np1)
        # print(np2)
        final_prediction = majority_voting(np1,
                                           np2)
        
        # print(final_prediction)
        new_final_prediction = []
        for cl in final_prediction:
            new_final_prediction.append(class_dict[cl])
        # print(new_final_prediction)
        # print(labels)
        # Compute accuracy
        print("ACCURACY:", accuracy_score(labels, new_final_prediction))

        # Save model
        np.save(os.path.join(ROOT_DIR, 'Ensemble.npy'), new_final_prediction)

    except FileNotFoundError as err:
        print('[ERROR] Train random forest, SVM, CNN-custom '
              'and VGG16 models before executing ensemble model!')
        print('[ERROR MESSAGE]', err)


if __name__ == "__main__":
    main()
