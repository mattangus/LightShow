import tensorflow as tf
import helpers

def random_time_slice(item):
    spect = item[helpers.DATA_KEY]

    #change from [0,1] -> [0.75, 1.25]
    scale_factor = ((tf.random.uniform((), dtype=tf.float32) - 0.5) / 2.) + 1.

    pad_left_pct = ((tf.random.uniform((), dtype=tf.float32)) / 10.)
    pad_right_pct = ((tf.random.uniform((), dtype=tf.float32)) / 10.)

    dyn_shape = tf.shape(spect)
    stat_shape = spect.shape.as_list()
    new_shape = [stat_shape[0], tf.to_int32(tf.to_float(dyn_shape[1])*scale_factor)]

    spect_resize = tf.image.resize_bilinear(
        tf.expand_dims(spect,0),
        new_shape)
    
    left_padding = tf.zeros([1,stat_shape[0], tf.to_int32(tf.to_float(new_shape[1])*pad_left_pct), 1]) 
    right_padding = tf.zeros([1,stat_shape[0], tf.to_int32(tf.to_float(new_shape[1])*pad_right_pct), 1])

    spect_pad = tf.concat([left_padding, spect_resize, right_padding],2) 

    actual_width = tf.shape(spect_pad)[2]
    target_width = 150
    crop_width = tf.minimum(actual_width, target_width)

    spect_crop = tf.image.random_crop(spect_pad,
                    [1,stat_shape[0], crop_width,1])

    spect_crop = tf.image.resize_image_with_crop_or_pad(spect_crop, stat_shape[0], target_width)

    spect_crop = tf.squeeze(spect_crop, 0)
    
    new_item = item
    new_item[helpers.DATA_KEY] = spect_crop

    return new_item

def random_scale_and_pad(item):
    spect = item[helpers.DATA_KEY]

    #change from [0,1] -> [0.75, 1.25]
    scale_factor = ((tf.random.uniform((), dtype=tf.float32) - 0.5) / 2.) + 1.

    pad_left_pct = ((tf.random.uniform((), dtype=tf.float32)) / 10.)
    pad_right_pct = ((tf.random.uniform((), dtype=tf.float32)) / 10.)

    dyn_shape = tf.shape(spect)
    stat_shape = spect.shape.as_list()
    start_pos = 0
    new_shape = [stat_shape[start_pos], tf.to_int32(tf.to_float(dyn_shape[1])*scale_factor)]

    spect_resize = tf.image.resize_bilinear(
        tf.expand_dims(spect,0),
        new_shape)
    
    left_padding = tf.zeros([1,stat_shape[start_pos], tf.to_int32(tf.to_float(new_shape[1])*pad_left_pct), 1]) 
    right_padding = tf.zeros([1,stat_shape[start_pos], tf.to_int32(tf.to_float(new_shape[1])*pad_right_pct), 1])

    spect_pad = tf.concat([left_padding, spect_resize, right_padding],2)

    spect_pad = tf.squeeze(spect_pad, 0)

    new_item = item
    new_item[helpers.DATA_KEY] = spect_pad

    return new_item

def random_noise(item):
    spect = item[helpers.DATA_KEY]

    spect = spect + tf.random_normal(tf.shape(spect), stddev=0.01)

    new_item = item
    new_item[helpers.DATA_KEY] = spect

    return new_item

def normalize(item):
    spect = item[helpers.DATA_KEY]

    spect = tf.image.per_image_standardization(spect)

    new_item = item
    new_item[helpers.DATA_KEY] = spect

    return new_item

def augment(item):
    item = random_time_slice(item)
    # item = random_scale_and_pad(item)
    # item = normalize(item)
    item = random_noise(item)

    return item

def _len_fn(item):
    spect = item[helpers.DATA_KEY]
    return tf.shape(spect)[2]

#TODO: include data augmentation here
def apply_preprocessing(dataset):
    with tf.variable_scope("Preprocess"):
        dataset = dataset.map(augment, num_parallel_calls=30)

        #tess [ 60.   82.4 104.8 127.2 149.6 172.  194.4 216.8 239.2 261.6 284. ]
        #crema [ 40.   51.7  63.4  75.1  86.8  98.5 110.2 121.9 133.6 145.3 157. ]
        #meld [   8.  361.  714. 1067. 1420. 1773. 2126. 2479. 2832. 3185. 3538.]
        #savee [161.  206.4 251.8 297.2 342.6 388.  433.4 478.8 524.2 569.6 615. ]
        # boundaries = [(i + 3)*30 for i in range(50)]
        # batch_sizes = [30] * (len(boundaries) + 1)

        # bucketer = tf.data.experimental.bucket_by_sequence_length(
        #         _len_fn,boundaries,batch_sizes)
        
        # dataset = dataset.apply(bucketer)

    return dataset