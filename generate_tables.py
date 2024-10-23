from datetime import timedelta

from trdg.generators import (
    GeneratorFromStrings,
)
from tqdm.auto import tqdm
import os
import numpy as np
import random
import pandas as pd
from faker import Faker
from PIL import Image, ImageDraw, ImageFont




NUM_IMAGES_TO_SAVE = 10
NUM_PRICES_TO_GENERATE = 10000

data = []

for i in range(1000):
    fake_person = Faker()
    fake_start = fake_person.date_time_this_month()
    duration = timedelta(hours=random.randint(3, 10))
    fake_end = fake_start + duration
    fake_location = fake_person.city()

    data.append([fake_person.name(), str(fake_start), str(fake_end), fake_location])

    df = pd.DataFrame(data, columns=["name", "start_time", "end_time", "location"])

    all_words = df[["name", "start_time", "end_time", "location"]].to_numpy().flatten()

# ignore np nan
num_before = len(all_words)
all_words = [x for x in all_words if str(x) != 'nan']
after_nan_filter = len(all_words)
print("removed: ", num_before - after_nan_filter, "words because of nan values")
all_words = list(set(all_words))
print("Removed", len(all_words), "duplicates")
print("Current number of words: ", len(all_words))




# now given word list and number list, get all combinations
all_combinations = []
for word in tqdm(all_words):
    all_combinations.append(word)

#generate the images
generator = GeneratorFromStrings(
    random.sample(all_combinations, min(len(all_combinations), 10000)),

    # uncomment the lines below for some image augmentation options
    # blur=6,
    # random_blur=True,
    # random_skew=True,
    # skewing_angle=20,
    # background_type=1,
    # text_color="red",
)

# save images from generator
# if output folder doesnt exist, create it
if not os.path.exists('output'):
    os.makedirs('output')
#if labels.txt doesnt exist, create it
if not os.path.exists('output/labels.txt'):
    f = open("output/labels.txt", "w")
    f.close()

#open txt file
current_index = len(os.listdir('output')) - 1 #all images minus the labels file
f = open("output/labels.txt", "a")

images = []

for counter, (img, lbl) in tqdm(enumerate(generator), total = NUM_IMAGES_TO_SAVE):
    if (counter >= NUM_IMAGES_TO_SAVE):
        break
    images.append(img)
    # img.show()
    #save pillow image
    current_index += 1
    # Do something with the pillow images here.


def create_final_table_image_from_images(images, cell_width, cell_height, cols, rows):
    # Step 1: Create a blank canvas large enough to hold the final table of images
    table_width = cell_width * cols
    table_height = cell_height * rows
    final_image = Image.new("RGB", (table_width, table_height), color="white")

    # Step 2: Place each word image into the corresponding cell in the grid
    for row in range(rows):
        for col in range(cols):
            img_index = row * cols + col
            if img_index >= len(images):
                break  # Stop if we run out of images
            img = images[img_index]

            # Resize the image to fit the cell, if necessary
            img_resized = img.resize((cell_width, cell_height))

            # Calculate the position for the image in the table
            x = col * cell_width
            y = row * cell_height

            # Paste the image into the final table image
            final_image.paste(img_resized, (x, y))

    # Step 3: Return the final combined image
    return final_image

final = create_final_table_image_from_images(images, 300, 150, 3, 3)

final.save(f'output/imagefinal.png')
#f.write(f'imagefinal.png {lbl}\n')

f.close()



