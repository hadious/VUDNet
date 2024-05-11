import tensorflow as tf
from Surface_VH import Surface_VH
import time
import os
from GAN import generator_loss, discriminator_loss, Generator, Discriminator #  build_generator, build_discriminator,
from keras.models import load_model
from keras import backend as Kend
import matplotlib.pyplot as plt
from tqdm import tqdm


BATCH_SIZE = 2
EPOCHS = 150
path="2D_fringe"
INPUT_NUM = 1 #9
Kend.clear_session()
DYModel = load_model('./DYnet++.h5', custom_objects={"tf": tf})


train_dataset = tf.data.Dataset.from_generator(
    lambda: Surface_VH(path=path, image_suffix='.png', depthMap_suffix='.npz', transform=None, DYModel=DYModel),
    output_types=(tf.float32, tf.float32),
    output_shapes=((512, 256, INPUT_NUM), (512, 256,1))
)
train_dataset = train_dataset.batch(BATCH_SIZE)

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

generator = Generator()
discriminator = Discriminator()


# For saving checkpoints and resuming training
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)



# Training Function
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        print(f"Epoch {epoch+1}/{epochs}")

        for input_image, target in tqdm(dataset, desc='Training Progress'):
            train_step(input_image, target)

        # Saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))




@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))



def generate_images(model, test_input, tar):
    prediction = model(test_input, training=False)
    plt.figure(figsize=(15, 15))

    total_columns = max(INPUT_NUM, 5)  # Ensure at least 5 columns for a balanced layout
    fig, axs = plt.subplots(3, total_columns, figsize=(15, 5))
    
    if total_columns == 1:
        axs = axs.reshape(3, 1)  # Ensure axs is a 2D array when we have only 1 column

    central_index = total_columns // 2  # Central column for displaying ground truth and predicted images

    # Plot each channel of the input image
    for i in range(INPUT_NUM):
        # Convert the EagerTensor to numpy array if needed:
        input_image = test_input[0, :, :, i].numpy()
        axs[0, i].imshow(input_image, cmap='gray')
        axs[0, i].set_title(f'Input Channel {i+1}')
        axs[0, i].axis('off')

    # Plot ground truth image
    ground_truth = tf.squeeze(tar[0]).numpy()  # Remove dimensions of size 1 and convert to numpy
    axs[1, central_index].imshow(ground_truth, cmap='gray')
    axs[1, central_index].set_title('Ground Truth')
    axs[1, central_index].axis('off')

    # Align other cells in the ground truth row
    for i in range(total_columns):
        if i != central_index:
            axs[1, i].axis('off')

    # Plot predicted image
    predicted_image = tf.squeeze(prediction[0]).numpy()  # Remove dimensions of size 1 and convert to numpy
    axs[2, central_index].imshow(predicted_image, cmap='gray')
    axs[2, central_index].set_title('Predicted Image')
    axs[2, central_index].axis('off')

    # Align other cells in the predicted image row
    for i in range(total_columns):
        if i != central_index:
            axs[2, i].axis('off')

    plt.tight_layout()
    plt.show()




def test(dataset):
    for example_input, example_target in dataset.take(10):  
        generate_images(generator, example_input, example_target)


train(train_dataset, EPOCHS)

test(train_dataset)
