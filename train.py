# coding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from PIL import Image
import keras
import numpy as np
import random
import tensorflow as tf
from utils import visualize
from utils.config import Config
from utils.anchors import get_anchors
from utils.utils import mold_inputs,unmold_detections
from nets.mrcnn import get_train_model,get_predict_model
from nets.mrcnn_training import data_generator,load_image_gt
from dataset import ShapesDataset


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)


class ShapesConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    # ======= modify ========
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1
    NUM_CLASSES = 1 + 2
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    STEPS_PER_EPOCH = 25
    VALIDATION_STEPS = 25

if __name__ == "__main__":
    learning_rate = 1e-5
    init_epoch = 0
    epoch = 20

    dataset_root_path="./train_dataset/"
    img_floder = dataset_root_path + "imgs/"
    mask_floder = dataset_root_path + "mask/"
    yaml_floder = dataset_root_path + "yaml/"
    imglist = os.listdir(img_floder)

    count = len(imglist)
    np.random.seed(10101)
    np.random.shuffle(imglist)
    train_imglist = imglist[:int(count*0.9)]
    val_imglist = imglist[int(count*0.9):]

    MODEL_DIR = "logs"

    COCO_MODEL_PATH = "model_data/mask_rcnn_coco.h5"
    config = ShapesConfig()
    config.display()

    # ËÆ≠Á??∞ÊçÆ?ÜÂ?Â§?
    dataset_train = ShapesDataset()
    dataset_train.load_shapes(len(train_imglist), img_floder, mask_floder, train_imglist, yaml_floder)
    dataset_train.prepare()

    # È™åË??∞ÊçÆ?ÜÂ?Â§?
    dataset_val = ShapesDataset()
    dataset_val.load_shapes(len(val_imglist), img_floder, mask_floder, val_imglist, yaml_floder)
    dataset_val.prepare()

    # ?∑Â?ËÆ≠Á?Ê®°Â?
    model = get_train_model(config)
    model.load_weights(COCO_MODEL_PATH,by_name=True,skip_mismatch=True)

    # ?∞ÊçÆ?üÊ???
    train_generator = data_generator(dataset_train, config, shuffle=True,
                                        batch_size=config.BATCH_SIZE)
    val_generator = data_generator(dataset_val, config, shuffle=True,
                                    batch_size=config.BATCH_SIZE)

    # ?ûÊâß?ΩÊï∞
    # ÊØèÊ¨°ËÆ≠Á?‰∏Ä‰∏™‰?‰ª?ÉΩ‰ºö‰?Â≠?
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=MODEL_DIR,
                                    histogram_freq=0, write_graph=True, write_images=False),
        keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "epoch{epoch:03d}_loss{loss:.3f}_val_loss{val_loss:.3f}.h5"),
                                        verbose=0, save_weights_only=True),
                                        

    ]
    

    log("\nStarting at epoch {}. LR={}\n".format(init_epoch, learning_rate))
    log("Checkpoint Path: {}".format(MODEL_DIR))

    # ‰ΩøÁî®?Ñ‰??ñÂô®??
    optimizer = keras.optimizers.Adam(lr=learning_rate)

    # ËÆæÁΩÆ‰∏Ä‰∏ãloss‰ø°ÊÅØ
    model._losses = []
    model._per_input_losses = {}
    loss_names = [
        "rpn_class_loss",  "rpn_bbox_loss",
        "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
    for name in loss_names:
        layer = model.get_layer(name)
        if layer.output in model.losses:
            continue
        loss = (
            tf.reduce_mean(layer.output, keepdims=True)
            * config.LOSS_WEIGHTS.get(name, 1.))
        model.add_loss(loss)

    # Â¢ûÂ?L2Ê≠???ñÔ??æÁΩÆËøáÊ???
    reg_losses = [
        keras.regularizers.l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
        for w in model.trainable_weights
        if 'gamma' not in w.name and 'beta' not in w.name]
    model.add_loss(tf.add_n(reg_losses))


    # ËøõË?ÁºñË?
    model.compile(
        optimizer=optimizer,
        loss=[None] * len(model.outputs)
    )

    # ?®‰??æÁ§∫ËÆ≠Á??ÖÂÜµ
    for name in loss_names:
        if name in model.metrics_names:
            print(name)
            continue
        layer = model.get_layer(name)
        model.metrics_names.append(name)
        loss = (
            tf.reduce_mean(layer.output, keepdims=True)
            * config.LOSS_WEIGHTS.get(name, 1.))
        model.metrics_tensors.append(loss)


    model.fit_generator(
        train_generator,
        initial_epoch=init_epoch,
        epochs=epoch,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        callbacks=callbacks,
        validation_data=val_generator,
        validation_steps=config.VALIDATION_STEPS,
        max_queue_size=100,
        workers=1, 
        use_multiprocessing=False
    )


