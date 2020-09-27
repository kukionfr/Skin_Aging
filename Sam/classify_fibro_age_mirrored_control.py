import tensorflow as tf
from tensorflow_docs import modeling
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from time import time
# solution #1
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

memorysize = 6500 #2070
# solution #2
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memorysize)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

tfds.disable_progress_bar()  # disable tqdm progress bar
AUTOTUNE = tf.data.experimental.AUTOTUNE
print("TensorFlow Version: ", tf.__version__)
print("Number of GPU available: ", len(tf.config.experimental.list_physical_devices("GPU")))

IMG_HEIGHT = 100
IMG_WIDTH = 100
BATCH_SIZE = 64
val_fraction = 30
max_epochs= 300

augment_degree = 0.10
samplesize = [1200, 1600] #old, young
shuffle_buffer_size = 1000000  # take first 100 from dataset and shuffle and pick one.

def read_and_label(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    img = occlude(img, file_path)
    return img, label

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return tf.reshape(tf.where(parts[-4] == CLASS_NAMES), [])

def occlude(image, file_path):
    maskpth = tf.strings.regex_replace(file_path, 'image', 'label')
    mask = tf.io.read_file(maskpth)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float16)
    mask = tf.image.resize(mask, [IMG_WIDTH, IMG_HEIGHT])
    mask = tf.math.greater(mask, 0.25)
    # invert mask
    mask = tf.math.logical_not(mask)
    maskedimg = tf.where(mask, image, tf.ones(tf.shape(image)))
    return maskedimg

def augment(image, label):
    degree = augment_degree
    if degree == 0:
        return image, label
    image = tf.image.random_hue(image, max_delta=degree, seed=5)
    image = tf.image.random_contrast(image, 1-degree, 1+degree, seed=5)  # tissue quality
    image = tf.image.random_saturation(image, 1-degree, 1+degree, seed=5)  # stain quality
    image = tf.image.random_brightness(image, max_delta=degree)  # tissue thickness, glass transparency (clean)
    image = tf.image.random_flip_left_right(image, seed=5)  # cell orientation
    image = tf.image.random_flip_up_down(image, seed=5)  # cell orientation
    return image, label

def balance(data_dir):
    tmp = [0]
    for CLASS, n in zip(CLASS_NAMES, samplesize):
        secs = [_ for _ in data_dir.glob(CLASS+'/*')]
        for idx,sec in enumerate(secs):
            sec = os.path.join(sec, 'image/*.jpg')
            list_ds = tf.data.Dataset.list_files(sec)
            # subsample
            list_ds = (list_ds
                       .shuffle(shuffle_buffer_size)
                       .take(n)
                       )
            labeled_ds_org = list_ds.map(read_and_label, num_parallel_calls=AUTOTUNE)
            labeled_ds = labeled_ds_org.map(augment,num_parallel_calls=AUTOTUNE)

            # add augment
            sampleN = len(list(labeled_ds))
            while sampleN < n:
                labeled_ds_aug = (labeled_ds_org
                                  .shuffle(shuffle_buffer_size)
                                  .take(n-sampleN)
                                  .map(augment,num_parallel_calls=AUTOTUNE)
                                  )
                labeled_ds = labeled_ds.concatenate(labeled_ds_aug)
                sampleN = len(list(labeled_ds))
            print('list_ds: ', len(list(labeled_ds)),CLASS)
            # append
            if tmp[0] == 0:
                tmp[idx] = labeled_ds
            else:
                labeled_ds = tmp[0].concatenate(labeled_ds)
                tmp[0] = labeled_ds
        print(CLASS, ': sample size =', len(list(tmp[0])))
    return tmp[0].shuffle(shuffle_buffer_size)

# list location of all training images
train_data_dir = '/home/kuki2070s2/Desktop/Synology/aging/data/cnn_dataset/train'
train_data_dir = pathlib.Path(train_data_dir)
CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*') if item.name != ".DS_store"])
CLASS_NAMES = sorted(CLASS_NAMES, key=str.lower) #sort alphabetically case-insensitive


train_labeled_ds = balance(train_data_dir)
train_image_count = len(list(train_labeled_ds))
print('training set size : ', train_image_count)
val_image_count = train_image_count // 100 * val_fraction
print('validation size: ', val_image_count)
train_image_count2 = train_image_count-val_image_count
print('training set size after split : ', train_image_count2)
STEPS_PER_EPOCH = train_image_count2 // BATCH_SIZE
VALIDATION_STEPS = val_image_count // BATCH_SIZE
print('train step #',STEPS_PER_EPOCH)
print('validation step #',VALIDATION_STEPS)

plt.figure(figsize=(10,10))
for idx, elem in enumerate(train_labeled_ds.take(100)):
    img = elem[0]
    label = elem[1]
    ax = plt.subplot(10,10,idx+1)
    plt.imshow(img)
    plt.title(CLASS_NAMES[label].title())
    plt.axis('off')
plt.show()
target= 'cnn/ResV2_hub_t2'
if not os.path.exists(target): os.mkdir(target)
plt.savefig(target + '/aug10_all_training data.png')

train_ds = (train_labeled_ds
            .skip(val_image_count)
            .shuffle(buffer_size=shuffle_buffer_size)
            .repeat()
            .batch(BATCH_SIZE)
            .prefetch(buffer_size=AUTOTUNE)
            )

val_ds = (train_labeled_ds
          .take(val_image_count)
          .repeat()
          .batch(BATCH_SIZE)
          .prefetch(buffer_size=AUTOTUNE))

test_data_dir = '/home/kuki2070s2/Desktop/Synology/aging/data/cnn_dataset/test'
test_data_dir = pathlib.Path(test_data_dir)
test_labeled_ds = balance(test_data_dir)

plt.figure(figsize=(10,10))
for idx,elem in enumerate(test_labeled_ds.take(100)):
    img = elem[0]
    label = elem[1]
    ax = plt.subplot(10,10,idx+1)
    plt.imshow(img)
    plt.title(CLASS_NAMES[label].title())
    plt.axis('off')
plt.show()
plt.savefig(target + '/aug10_all_testing data.png')


test_ds = (test_labeled_ds
           # .cache("./cache/fibro_test.tfcache")
           .shuffle(buffer_size=shuffle_buffer_size)
           .repeat()
           .batch(BATCH_SIZE)
           .prefetch(buffer_size=AUTOTUNE)  # time it takes to produce next element
           )
test_image_count = len(list(test_labeled_ds))
print('test set size : ', test_image_count)
TEST_STEPS = test_image_count // BATCH_SIZE
print('test step # ', TEST_STEPS)

# checkpoint_dir = "training_1"
# shutil.rmtree(checkpoint_dir, ignore_errors=True)

def get_callbacks(name):
    return [
        modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                         patience=50, restore_best_weights=True),
        # tf.keras.callbacks.TensorBoard(log_dir/name, histogram_freq=1),
        # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/{}/cp.ckpt".format(name),
        #                                    verbose=0,
        #                                    monitor='val_sparse_categorical_crossentropy',
        #                                    save_weights_only=True,
        #                                    save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                             factor=0.1, patience=20, verbose=0, mode='auto',
                                             min_delta=0.0001, cooldown=0, min_lr=0),
    ]

# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#     1e-4,
#     decay_steps=STEPS_PER_EPOCH * 100,
#     decay_rate=1,
#     staircase=False)

def compilefit(model, name, max_epochs, train_ds, val_ds):
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model_history = model.fit(train_ds,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              epochs=max_epochs,
                              verbose=1,
                              validation_data=val_ds,
                              callbacks=get_callbacks(name),
                              validation_steps=VALIDATION_STEPS,
                              use_multiprocessing=True
                              )
    namename = os.path.dirname(name)
    if not os.path.isdir(os.path.abspath(namename)):
        os.mkdir(os.path.abspath(namename))
    if not os.path.isdir(os.path.abspath(name)):
        os.mkdir(os.path.abspath(name))
    if not os.path.isfile(pathlib.Path(name) / 'full_model.h5'):
        try:
            model.save(pathlib.Path(name) / 'full_model.h5')
        except:
            print('model not saved?')
    return model_history

def plotdf(dfobj, condition, repeat, lr=None):
    # pd.DataFrame(dfobj).plot(title=condition+repeat)
    dfobj1 = dfobj.copy()
    dfobj.pop('lr')
    dfobj.pop('loss')
    dfobj.pop('val_loss')
    pd.DataFrame(dfobj).plot(title=condition+'_'+repeat)
    plt.savefig('cnn/'+condition+'/'+repeat+'_accuracy.png')
    dfobj1.pop('lr')
    dfobj1.pop('accuracy')
    dfobj1.pop('val_accuracy')
    pd.DataFrame(dfobj1).plot(title=condition+'_'+repeat)
    plt.savefig('cnn/'+condition+'/'+repeat+'_loss.png')
    plt.show()

histories = {}

def evaluateit(network,networkname,repeat, train_ds, val_ds, test_ds):
    histories[networkname] = compilefit(network, 'cnn/'+networkname+'/'+repeat, max_epochs, train_ds, val_ds)
    results = network.evaluate(test_ds, steps=TEST_STEPS)
    plotdf(histories[networkname].history,networkname,repeat)
    print('test acc', results[-1] * 100)

trials = ['t'+str(_)+'_12001600_aug10_all' for _ in range(1,4)]
# trials = trials + ['t'+str(_) for _ in range(6,11)]

duration=[]

mirrored_strategy = tf.distribute.MirroredStrategy()
for trial in trials:
    start = time()
    with mirrored_strategy.scope():
        print('downloading model')
        IncV3_hub = tf.keras.Sequential([
            # tf.keras.layers.InputLayer(input_shape=(100, 100, 3)),
            hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
                           trainable=True, arguments=dict(batch_norm_momentum=0.99)),  # Can be True, see below.
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        IncV3_hub.build([None, 100, 100, 3])  # Batch input shape.
        print('training...........')
        evaluateit(IncV3_hub,'IncV3_hub',trial,train_ds,val_ds,test_ds)
    end = time()
    duration.append(end-start)
    print('duration : ', end-start)

print('duration : ', duration)
print('5res+5inc :',np.sum(duration))