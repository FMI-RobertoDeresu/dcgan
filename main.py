import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
import time
import sklearn
from pathlib import Path

(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

EPOCHS = 50
NOISE_DIM = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.0002
KERNEL_INIT = "lecun_normal"
# KERNEL_INIT = "he_normal"

num_of_batches = train_images.shape[0] // BATCH_SIZE + int(train_images.shape[0] % BATCH_SIZE != 0)
test_noise = np.random.normal(size=[16, NOISE_DIM])


def make_generator_model():
    model = tf.keras.Sequential()

    # with tf.name_scope('fully_1'):
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,), kernel_initializer=KERNEL_INIT))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False,
                                     kernel_initializer=KERNEL_INIT))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                     kernel_initializer=KERNEL_INIT))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh',
                                     kernel_initializer=KERNEL_INIT))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1],
                            kernel_initializer=KERNEL_INIT))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=KERNEL_INIT))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid", kernel_initializer=KERNEL_INIT))

    return model


def generate_and_save_images(model, epoch, test_input):
    predictions = model.predict(test_input)

    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    path = Path('images/image_at_epoch_{:04d}.png'.format(epoch))
    path.mkdir(parents=True, exist_ok=True)

    plt.savefig(str(path))
    plt.show()


def generate_gif():
    with imageio.get_writer('dcgan.gif', mode='I') as writer:
        filenames = glob.glob('images/image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


# inputs
gen_input = tf.placeholder(tf.float32, shape=[None, NOISE_DIM], name='gen_input')
disc_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='disc_input')

# networks
generator = make_generator_model()
generator_sample = generator(gen_input)

discriminator = make_discriminator_model()
discriminator_real = discriminator(disc_input)
discriminator_fake = discriminator(generator_sample)

# losses
gen_loss = -tf.reduce_mean(tf.log(discriminator_fake))
disc_loss = -tf.reduce_mean(tf.log(discriminator_real) + tf.log(1. - discriminator_fake))

# optimizers
generator_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

# train ops
train_gen = generator_optimizer.minimize(gen_loss, var_list=generator.trainable_variables)
train_disc = discriminator_optimizer.minimize(disc_loss, var_list=discriminator.trainable_variables)

# initialize the variables
init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_images_batches = np.array_split(sklearn.utils.shuffle(train_images), num_of_batches)
        egl = 0
        edl = 0
        for index, images_batch in enumerate(train_images_batches):
            noise = np.random.normal(size=[images_batch.shape[0], NOISE_DIM])

            feed_dict = {gen_input: noise, disc_input: images_batch}
            _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss], feed_dict=feed_dict)
            print("{} {}".format(gl, dl))

            egl += gl / num_of_batches
            edl += dl / num_of_batches

        generate_and_save_images(generator, epoch + 1, test_noise)

        elapsed_time = time.time() - start_time
        print('Epoch {}: Time: {}, Gen Loss: {}, Disc Loss: {}'.format(epoch + 1, elapsed_time, egl, edl))

generate_gif()
