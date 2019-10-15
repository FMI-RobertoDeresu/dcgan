import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
import time
import sklearn
from pathlib import Path
import math

train_uid = str(time.time()).replace(".", "").ljust(17, "0")

(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images[:1000]
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

assert not np.any(np.isnan(train_images))

EPOCHS = 100
NOISE_DIM = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
KERNEL_INIT = "lecun_normal"
# KERNEL_INIT = "he_normal"

num_of_batches = train_images.shape[0] // BATCH_SIZE + int(train_images.shape[0] % BATCH_SIZE != 0)
test_noise = np.random.normal(size=[16, NOISE_DIM])


def make_generator_model():
    model = tf.keras.Sequential(name="generator")

    model.add(layers.Dense(7 * 7 * 512, use_bias=False, input_shape=(100,), kernel_initializer=KERNEL_INIT))
    model.add(layers.ReLU())
    assert model.output_shape == (None, 7 * 7 * 512)

    model.add(layers.Reshape((7, 7, 512)))
    assert model.output_shape == (None, 7, 7, 512)

    # model.add(layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding="same", use_bias=False,
    # kernel_initializer=KERNEL_INIT))
    # model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    # model.add(layers.ReLU())
    # assert model.output_shape == (None, 7, 7, 512)

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same", use_bias=False,
                                     kernel_initializer=KERNEL_INIT))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.ReLU())
    assert model.output_shape == (None, 14, 14, 256)

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False,
                                     kernel_initializer=KERNEL_INIT))
    model.add(layers.Activation("tanh"))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential(name="discriminator")

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1],
                            kernel_initializer=KERNEL_INIT, name="conv_1"))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="batch_norm_1"))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 14, 14, 64)

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=KERNEL_INIT, name="conv_2"))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="batch_norm_2"))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 7, 7, 128)

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same", kernel_initializer=KERNEL_INIT, name="conv_3"))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="batch_norm_3"))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 4, 4, 256)

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same", kernel_initializer=KERNEL_INIT, name="conv_4"))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="batch_norm_4"))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 2, 2, 512)

    model.add(layers.Flatten())
    assert model.output_shape == (None, 2 * 2 * 512)

    model.add(layers.Dense(1, activation="sigmoid", kernel_initializer=KERNEL_INIT, name="dense_1"))
    assert model.output_shape == (None, 1)

    return model


def variable_summaries(var):
    var_min = tf.reduce_min(var)
    var_max = tf.reduce_max(var)
    var_mean = tf.reduce_mean(var)
    var_stddev = tf.sqrt(tf.reduce_mean(tf.square(var - var_mean)))

    tf.summary.scalar("summaries/min", var_min)
    tf.summary.scalar("summaries/max", var_max)
    tf.summary.scalar("summaries/mean", var_mean)
    tf.summary.scalar("summaries/stddev", var_stddev)
    tf.summary.histogram("summaries//histogram", var)


def generate_and_save_images(model, epoch, test_input):
    predictions = model.predict(test_input)

    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    path = Path("./images/{}/image_at_epoch_{:04d}.png".format(train_uid, epoch))
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(str(path))
    plt.close()


def generate_gif():
    with imageio.get_writer("./images/{}/dcgan.gif".format(train_uid), mode="I") as writer:
        filenames = glob.glob("./images/{}/image*.png".format(train_uid))
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
gen_input = tf.placeholder(tf.float32, shape=[None, NOISE_DIM], name="gen_input")
disc_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="disc_input")

# networks
generator = make_generator_model()
generator_sample = generator(gen_input)

discriminator = make_discriminator_model()
discriminator_real = discriminator(disc_input)
discriminator_fake = discriminator(generator_sample)

# summary
variable_summaries(discriminator.weights[0])

# losses
gen_loss = -tf.reduce_mean(tf.log(discriminator_fake))
disc_loss = -tf.reduce_mean(tf.log(discriminator_real) + tf.log(1. - discriminator_fake))

# optimizers
generator_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

# train ops
train_gen = generator_optimizer.minimize(gen_loss, var_list=generator.trainable_variables)
train_disc = discriminator_optimizer.minimize(disc_loss, var_list=discriminator.trainable_variables)

# start training
with tf.Session() as sess:
    # merge summaries
    summary_merge = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("./logs/tensorboard/{}".format(train_uid), sess.graph)

    # initialize the variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # train
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_images_batches = np.array_split(sklearn.utils.shuffle(train_images), num_of_batches)
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        for index, images_batch in enumerate(train_images_batches):
            noise = np.random.normal(size=[images_batch.shape[0], NOISE_DIM])

            _, dl = sess.run([train_disc, disc_loss], feed_dict={gen_input: noise, disc_input: images_batch})
            epoch_disc_loss += dl / num_of_batches

            _, gl = sess.run([train_gen, gen_loss], feed_dict={gen_input: noise})
            epoch_gen_loss += gl / num_of_batches

        # write summaries
        summary = sess.run(summary_merge)

        # summary.value.add(tag="disc_loss", simple_value=epoch_disc_loss)
        # summary.value.add(tag="gen_loss", simple_value=epoch_gen_loss)

        summary_writer.add_summary(summary, epoch)

        # save samples
        generate_and_save_images(generator, epoch + 1, test_noise)

        elapsed_time = time.time() - start_time
        print("Epoch {}: Time: {}, Gen Loss: {}, Disc Loss: {}".format(epoch + 1,
                                                                       elapsed_time,
                                                                       epoch_gen_loss,
                                                                       epoch_disc_loss))

        if math.isnan(epoch_gen_loss) or math.isnan(epoch_disc_loss):
            print("STOP on nan")
            break

    summary_writer.close()

generate_gif()
