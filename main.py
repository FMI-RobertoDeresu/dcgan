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

tf.add_check_numerics_ops()

train_uid = str(time.time()).replace(".", "").ljust(17, "0")

(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images#[:512]
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

assert not np.any(np.isnan(train_images))

EPOCHS = 100
NOISE_DIM = 100
BATCH_SIZE = 256
# BATCH_SIZE = 64
LEARNING_RATE = 0.0001
KERNEL_INIT = "lecun_normal"
# KERNEL_INIT = "he_normal"

num_of_batches = train_images.shape[0] // BATCH_SIZE + int(train_images.shape[0] % BATCH_SIZE != 0)
test_noise = np.random.normal(size=[16, NOISE_DIM])


def make_generator_model():
    model = tf.keras.Sequential(name="generator")

    model.add(layers.Dense(name="dense_1", units=7 * 7 * 512, input_shape=(100,),
                           use_bias=False, kernel_initializer=KERNEL_INIT))
    model.add(layers.ReLU())
    assert model.output_shape == (None, 7 * 7 * 512)

    model.add(layers.Reshape(name="reshape", target_shape=(7, 7, 512)))
    assert model.output_shape == (None, 7, 7, 512)

    model.add(layers.Conv2DTranspose(name="deconv_1", filters=512, kernel_size=(5, 5), strides=(1, 1), padding="same",
                                     use_bias=False, kernel_initializer=KERNEL_INIT))
    # model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.ReLU())
    assert model.output_shape == (None, 7, 7, 512)

    model.add(layers.Conv2DTranspose(name="deconv_2", filters=256, kernel_size=(5, 5), strides=(2, 2), padding="same",
                                     use_bias=False, kernel_initializer=KERNEL_INIT))
    # model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.ReLU())
    assert model.output_shape == (None, 14, 14, 256)

    model.add(layers.Conv2DTranspose(name="deconv_3", filters=1, kernel_size=(5, 5), strides=(2, 2), padding="same",
                                     use_bias=False, kernel_initializer=KERNEL_INIT))
    model.add(layers.Activation("tanh"))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential(name="discriminator")

    model.add(layers.Conv2D(name="conv_1", filters=64, kernel_size=(5, 5), strides=(2, 2), padding="same",
                            use_bias=False, kernel_initializer=KERNEL_INIT, input_shape=[28, 28, 1]))
    # model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="batch_norm_1"))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 14, 14, 64)

    model.add(layers.Conv2D(name="conv_2", filters=128, kernel_size=(5, 5), strides=(2, 2), padding="same",
                            use_bias=False, kernel_initializer=KERNEL_INIT))
    # model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="batch_norm_2"))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 7, 7, 128)

    model.add(layers.Conv2D(name="conv_3", filters=256, kernel_size=(5, 5), strides=(2, 2), padding="same",
                            use_bias=False, kernel_initializer=KERNEL_INIT))
    # model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="batch_norm_3"))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 4, 4, 256)

    model.add(layers.Conv2D(name="conv_4", filters=512, kernel_size=(5, 5), strides=(2, 2), padding="same",
                            use_bias=False, kernel_initializer=KERNEL_INIT))
    # model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="batch_norm_4"))
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 2, 2, 512)

    model.add(layers.Flatten(name="flatten"))
    assert model.output_shape == (None, 2 * 2 * 512)

    model.add(layers.Dense(name="dense_1", units=1, activation="sigmoid",
                           use_bias=False, kernel_initializer=KERNEL_INIT))
    assert model.output_shape == (None, 1)

    return model


def model_summaries(model):
    for var in model.weights:
        variable_summaries(var, model.name)


def variable_summaries(var, name_prefix=""):
    name_prefix = "{}/{}".format(name_prefix, var.name)

    var_min = tf.reduce_min(var)
    var_max = tf.reduce_max(var)
    var_mean = tf.reduce_mean(var)
    var_stddev = tf.sqrt(tf.reduce_mean(tf.square(var - var_mean)))

    tf.summary.scalar(name_prefix + "/min", var_min)
    tf.summary.scalar(name_prefix + "/max", var_max)
    tf.summary.scalar(name_prefix + "/mean", var_mean)
    tf.summary.scalar(name_prefix + "/stddev", var_stddev)
    tf.summary.histogram(name_prefix, tf.where(tf.is_nan(var), tf.scalar_mul(1e-8, tf.ones_like(var)), var))


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

# losses
disk_real_nans = tf.reduce_sum(
    tf.where(tf.is_nan(discriminator_real), tf.ones_like(discriminator_real), tf.zeros_like(discriminator_real)))
disk_real_min = tf.reduce_min(discriminator_real)
disk_real_max = tf.reduce_max(discriminator_real)

disk_fake_nans = tf.reduce_sum(
    tf.where(tf.is_nan(discriminator_fake), tf.ones_like(discriminator_fake), tf.zeros_like(discriminator_fake)))
disk_fake_min = tf.reduce_min(discriminator_fake)
disk_fake_max = tf.reduce_max(discriminator_fake)

gen_loss = tf.log(discriminator_fake)
gen_loss_nans = tf.reduce_sum(tf.where(tf.is_nan(gen_loss), tf.ones_like(gen_loss), tf.zeros_like(gen_loss)))
gen_loss = -tf.reduce_mean(tf.where(tf.is_nan(gen_loss), tf.scalar_mul(1e-8, tf.ones_like(gen_loss)), gen_loss))

disc_loss = tf.log(discriminator_real) + tf.log(1. - discriminator_fake)
disc_loss_nans = tf.reduce_sum(tf.where(tf.is_nan(disc_loss), tf.ones_like(disc_loss), tf.zeros_like(disc_loss)))
disc_loss = -tf.reduce_mean(tf.where(tf.is_nan(disc_loss), tf.scalar_mul(1e-8, tf.ones_like(disc_loss)), disc_loss))

# optimizers
generator_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

# train ops
train_gen = generator_optimizer.minimize(gen_loss, var_list=generator.trainable_variables)
train_disc = discriminator_optimizer.minimize(disc_loss, var_list=discriminator.trainable_variables)

# summary
model_summaries(discriminator)
model_summaries(generator)

gen_output = tf.placeholder(dtype=tf.float32, name="generator/output")
variable_summaries(gen_output)

disc_output = tf.placeholder(dtype=tf.float32, name="discriminator/output")
variable_summaries(disc_output)

tf.summary.scalar("loss/generator", gen_loss)
tf.summary.scalar("loss/discriminator", disc_loss)

# start training
sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
with tf.Session(config=sess_config) as sess:
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

        epoch_gen_loss_val = 0
        epoch_disc_loss_val = 0

        epoch_gen_loss_nans_val = 0
        epoch_disc_loss_nans_val = 0

        epoch_disk_real_nans_val = 0
        epoch_disk_real_min_val = 0
        epoch_disk_real_max_val = 1

        epoch_disk_fake_nans_val = 0
        epoch_disk_fake_min_val = 0
        epoch_disk_fake_max_val = 1

        for index, images_batch in enumerate(train_images_batches):
            noise = np.random.normal(size=[images_batch.shape[0], NOISE_DIM])

            _, disc_loss_val = sess.run([train_disc, disc_loss], feed_dict={gen_input: noise, disc_input: images_batch})
            epoch_disc_loss_val += disc_loss_val / num_of_batches

            _, gen_loss_val = sess.run([train_gen, gen_loss], feed_dict={gen_input: noise})
            epoch_gen_loss_val += gen_loss_val / num_of_batches

            gen_loss_nans_val, disc_loss_nans_val = sess.run([gen_loss_nans, disc_loss_nans],
                                                             feed_dict={gen_input: noise, disc_input: images_batch})
            epoch_gen_loss_nans_val += gen_loss_nans_val
            epoch_disc_loss_nans_val += disc_loss_nans_val

            disk_fake_nans_val, disk_fake_min_val, disk_fake_max_val, disk_real_nans_val, disk_real_min_val, disk_real_max_val = sess.run(
                [disk_fake_nans, disk_fake_min, disk_fake_max, disk_real_nans, disk_real_min, disk_real_max],
                feed_dict={gen_input: noise, disc_input: images_batch})

            epoch_disk_real_nans_val += disk_real_nans_val
            epoch_disk_real_min_val = max(epoch_disk_real_min_val, disk_real_min_val)
            epoch_disk_real_max_val = min(epoch_disk_real_max_val, disk_real_max_val)

            epoch_disk_fake_nans_val += disk_fake_nans_val
            epoch_disk_fake_min_val = max(epoch_disk_fake_min_val, disk_fake_min_val)
            epoch_disk_fake_max_val = min(epoch_disk_fake_max_val, disk_fake_max_val)

        elapsed_time = time.time() - start_time
        print(
            "Epoch {}: Time: {}, Gen Loss: {}, Disc Loss: {}, Nans (g: {}, d:{}), real (nans: {}, min: {:.2e}, max: {:.2e}), fake (nans: {}, min: {:.2e}, max: {:.2e})"
                .format(epoch + 1,
                        elapsed_time,
                        epoch_gen_loss_val,
                        epoch_disc_loss_val,
                        epoch_gen_loss_nans_val,
                        epoch_disc_loss_nans_val,
                        epoch_disk_real_nans_val,
                        epoch_disk_real_min_val,
                        epoch_disk_real_max_val,
                        epoch_disk_fake_nans_val,
                        epoch_disk_fake_min_val,
                        epoch_disk_fake_max_val))

        # write summaries
        summary_images = train_images_batches[0]
        summary_noise = np.random.normal(size=[summary_images.shape[0], NOISE_DIM])

        gen_output_val, disc_output_val = sess.run([generator_sample, discriminator_fake],
                                                   feed_dict={gen_input: summary_noise})

        summary = sess.run(summary_merge, feed_dict={gen_input: summary_noise, disc_input: summary_images,
                                                     gen_output: gen_output_val, disc_output: disc_output_val})
        summary_writer.add_summary(summary, epoch)

        # save samples
        generate_and_save_images(generator, epoch + 1, test_noise)

        if math.isnan(epoch_gen_loss_val) or math.isnan(epoch_disc_loss_val):
            print("STOP on nan")
            break

    summary_writer.close()

generate_gif()
