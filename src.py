import tensorflow as tf

def make_test_train():
    train = tf.keras.preprocessing.image_dataset_from_directory(
        '_data/_bylabel/_train', 
        batch_size=32, 
        image_size=(180, 180), 
        shuffle=True, 
        validation_split=None,
    )

    test = tf.keras.preprocessing.image_dataset_from_directory(
        '_data/_bylabel/_test', 
        batch_size=32, 
        image_size=(180, 180), 
        shuffle=True, 
        validation_split=None,
    )
    
    return test, train