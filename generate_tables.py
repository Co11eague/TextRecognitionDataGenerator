from datetime import timedelta
from tkinter.font import names

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
from datetime import datetime, timedelta





NUM_IMAGES_TO_SAVE = 100
NUM_PRICES_TO_GENERATE = 1000

data = []


for i in range(1000):
    fake_person = Faker()
    fake_start_str = fake_person.time(pattern="%H:%M")
    fake_start = datetime.strptime(fake_start_str, "%H:%M")
    fake_date = fake_person.date()

    # Add a random duration (between 3 to 10 hours)
    duration = timedelta(hours=random.randint(3, 10))

    # Calculate the end time by adding the duration
    fake_end = fake_start + duration

    # Convert end time back to a string if needed for display purposes
    fake_end_str = fake_end.strftime("%H:%M")
    fake_location = fake_person.city()
    fake_day = fake_person.day_of_week()

    # Append the data
    data.append([fake_person.first_name(), fake_start_str, fake_end_str, fake_location, fake_day, fake_date])

# Create DataFrame from the generated data (do this once after loop)
df = pd.DataFrame(data, columns=["name", "start_time", "end_time", "location", "fake_day", "fake_date"])

# Split into 4 separate arrays (done after the loop)
names = df["name"].to_numpy()
start_times = df["start_time"].to_numpy()
end_times = df["end_time"].to_numpy()
locations = df["location"].to_numpy()
fake_days = df["fake_day"].to_numpy()
fake_dates = df["fake_date"].to_numpy()



def cleanup(data):
    num_before = len(data)
    data = [x for x in data if str(x) != 'nan']
    after_nan_filter = len(data)
    print("removed: ", num_before - after_nan_filter, "words because of nan values")
    data = list(set(data))
    print("Removed", len(data), "duplicates")
    print("Current number of words: ", len(data))
    return data

def combinations(data):
    combinations = []
    for word in tqdm(data):
        combinations.append(word)

    return combinations

names = cleanup(names)
start_times = cleanup(start_times)
end_times = cleanup(end_times)
locations = cleanup(locations)
fake_days = cleanup(fake_days)
fake_dates = cleanup(fake_dates)


# now given word list and number list, get all combinations
name_combinations = combinations(names)
start_time_combinations = combinations(start_times)
end_time_combinations = combinations(end_times)
location_combinations = combinations(locations)
fake_day_combinations = combinations(fake_days)
fake_dates_combinations = combinations(fake_dates)


#generate the images
nameGenerator = GeneratorFromStrings(
    random.sample(name_combinations, min(len(name_combinations), 10000)),

    # uncomment the lines below for some image augmentation options
    # blur=6,
    # random_blur=True,
    # random_skew=True,
    # skewing_angle=20,
    # background_type=1,
    # text_color="red",
)

startTimeGenerator = GeneratorFromStrings(
    random.sample(start_time_combinations, min(len(start_time_combinations), 10000)),
)

endTimeGenerator = GeneratorFromStrings(
    random.sample(end_time_combinations, min(len(end_time_combinations), 10000)),
)

locationGenerator = GeneratorFromStrings(
    random.sample(location_combinations, min(len(location_combinations), 10000)),
)

daysGenerator = GeneratorFromStrings(
    random.sample(fake_day_combinations, min(len(fake_day_combinations), 10000)),
)

datesGenerator = GeneratorFromStrings(
    random.sample(fake_dates_combinations, min(len(fake_dates_combinations), 10000)),
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


def create_final_rota_image(name_images, start_time_images, end_time_images, location_images, days_images, dates_images, cell_width,
                            cell_height, rows):
    # Set up table size
    num_days = 10  # Number of days to display
    cols = 1 + (num_days * 2) + 1  # 1 Name column, 2 columns per day (Start & End), 1 Location column
    table_width = cell_width * cols
    table_height = cell_height * (rows + 1)  # Adjust height for the weekday row
    final_image = Image.new("RGB", (table_width, table_height), color="white")

    for i, date_image in enumerate(dates_images):
        if i == num_days:
            break
        resized_date_image = date_image.resize((cell_width * 2, cell_height))  # Resize to span two columns
        final_image.paste(resized_date_image, (i * cell_width * 2 + cell_width, 0))  # Position date images


    # Step 1: Place the weekday images in the first row
    for i, day_image in enumerate(days_images):
        if i == num_days:
            break
        resized_day_image = day_image.resize((cell_width * 2, cell_height))  # Resize to span two columns
        final_image.paste(resized_day_image, (i * cell_width * 2 + cell_width, cell_height))  # Position weekday images below date images


    # Step 2: Place each word image into its corresponding cell in the grid
    for row in range(rows):
        # Get images for the current row
        if row >= len(name_images):
            break  # Stop if we run out of images

        images_in_row = [name_images[row]]

        # Add start and end times for each day
        for i in range(num_days):
            index = random.randint(0, len(start_time_images) - 1)
            images_in_row.append(start_time_images[index])  # Start time for day i
            images_in_row.append(end_time_images[index])  # End time for day i

        images_in_row.append(location_images[row])  # Location column

        for col in range(cols):
            img = images_in_row[col]
            img_resized = img.resize((cell_width, cell_height))
            x = col * cell_width
            y = (row + 1) * cell_height  # Offset by one row to accommodate the weekday images
            final_image.paste(img_resized, (x, y))

    return final_image

def print_progress(generator, label):
    images = []
    for img, lbl in generator:
        images.append(img)
        print(f"{label}: {len(images)} images processed")
        if len(images) >= NUM_IMAGES_TO_SAVE:
            break
    return images

name_images = print_progress(nameGenerator, "Name")
start_time_images = print_progress(startTimeGenerator, "Start Time")
end_time_images = print_progress(endTimeGenerator, "End Time")
location_images = print_progress(locationGenerator, "Location")
days_images = print_progress(daysGenerator, "Day")
dates_images = print_progress(datesGenerator, "Date")

final_rota_image = create_final_rota_image(
    name_images,
    start_time_images,
    end_time_images,
    location_images,
    days_images,
    dates_images,
    cell_width=300,
    cell_height=150,
    rows=20  # Adjust rows based on the data size
)

final_rota_image.save('output/rota_image.png')



#f.write(f'imagefinal.png {lbl}\n')

f.close()



