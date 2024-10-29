from random import randint

from trdg.generators import (
    GeneratorFromStrings,
)
from tqdm.auto import tqdm
import os
import random
import pandas as pd
from faker import Faker
from PIL import Image, ImageDraw
from datetime import datetime, timedelta
import json

NUMBER_OF_ROTAS = 10
NUM_IMAGES_TO_SAVE = 1000

data = []

for i in range(1000):
    fake_person = Faker()
    fake_start_str = fake_person.time(pattern="%H:%M")
    fake_start = datetime.strptime(fake_start_str, "%H:%M")
    fake_date = fake_person.date()

    # Add a random duration (between 3 to 10 hours)
    duration = timedelta(hours=random.randint(2, 12), minutes=random.randint(0, 59))

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
    random_blur=True,
    random_skew=True,
    # skewing_angle=20,
    # background_type=1,
    # text_color="red",
)

startTimeGenerator = GeneratorFromStrings(
    random.sample(start_time_combinations, min(len(start_time_combinations), 10000)),
    random_blur=True,
    random_skew=True,
)

endTimeGenerator = GeneratorFromStrings(
    random.sample(end_time_combinations, min(len(end_time_combinations), 10000)),
    random_blur=True,
    random_skew=True,
)

locationGenerator = GeneratorFromStrings(
    random.sample(location_combinations, min(len(location_combinations), 10000)),
    random_blur=True,
    random_skew=True,
)

daysGenerator = GeneratorFromStrings(
    random.sample(fake_day_combinations, min(len(fake_day_combinations), 10000)),
    random_blur=True,
    random_skew=True,
)

datesGenerator = GeneratorFromStrings(
    random.sample(fake_dates_combinations, min(len(fake_dates_combinations), 10000)),
    random_blur=True,
    random_skew=True,
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
labels = {}


def create_final_rota_image(name_images_with_labels, start_time_images_with_labels, end_time_images_with_labels, location_images_with_labels, days_images_with_labels, dates_images_with_labels, cell_width,
                            cell_height, rows):
    # Set up table size
    collected_labels = {}
    num_days = randint(5,10)  # Number of days to display
    cols = 1 + (num_days * 2) + 1  # 1 Name column, 2 columns per day (Start & End), 1 Location column
    table_width = cell_width * cols
    table_height = cell_height * (rows + 1)  # Adjust height for the weekday row
    background_colors = ["#f5f5f5", "#e0e0e0", "#ffffff", "#f0f8ff"]
    final_image = Image.new("RGB", (table_width, table_height), color=random.choice(background_colors))
    draw = ImageDraw.Draw(final_image)

    random.shuffle(dates_images_with_labels)

    # Draw the outline for the table
    for row in range(rows + 2):  # +2 for the date and weekday rows
        draw.line([(0, row * cell_height), (table_width, row * cell_height)], fill="black", width=20)  # Horizontal lines
    for col in range(cols):
        draw.line([(col * cell_width, 0), (col * cell_width, table_height)], fill="black", width=20)  # Vertical lines


    date_labels = []
    day_labels = []

    # Place the date images in the first row
    for i,  (date_image, date_label) in enumerate(dates_images_with_labels):
        if i == num_days:
            break
        resized_date_image = date_image.resize((cell_width * 2 - 10, cell_height - 10))  # Resize to fit with padding
        final_image.paste(resized_date_image, (i * cell_width * 2 + cell_width, 0))  # Position date images
        date_labels.append(date_label)

    random.shuffle(days_images_with_labels)

    # Step 1: Place the weekday images in the second row
    for i, (day_image, day_label) in enumerate(days_images_with_labels):
        if i == num_days:
            break

        resized_day_image = day_image.resize((cell_width * 2 - 10, cell_height - 10))  # Resize to fit with padding
        final_image.paste(resized_day_image, (i * cell_width * 2 + cell_width, cell_height))  # Position weekday images
        day_labels.append(day_label)

    # Step 2: Place each word image into its corresponding cell in the grid
    for row in range(rows):
        name_image, name_label = name_images_with_labels[random.randint(0, len(name_images_with_labels) - 1)]

        if name_label not in collected_labels:
            collected_labels[name_label] = []
        # Get images for the current row
        if row >= len(name_images_with_labels):
            break  # Stop if we run out of images

        images_in_row = [name_image]

        location_image, location_label = location_images_with_labels[random.randint(0, len(location_images_with_labels) - 1)]
        images_in_row.append(location_image)  # Location column

        # Add start and end times for each day
        for i in range(num_days):
            index = random.randint(0, len(start_time_images_with_labels) - 1)
            start_time_image, start_time_label = start_time_images_with_labels[index]
            end_time_image, end_time_label = end_time_images_with_labels[index]

            images_in_row.append(start_time_image)  # Start time for day i
            images_in_row.append(end_time_image)  # End time for day i

            collected_labels[name_label].append({
                "start_time": start_time_label,
                "end_time": end_time_label,
                "location": location_label,
                "day": day_labels[i],  # This should correspond to the day in this row
                "date": date_labels[i]
            })




        for col in range(cols):
            img = images_in_row[col]
            img_resized = img.resize((cell_width - 10, cell_height - 10))  # Resize to fit in the cell
            x = col * cell_width
            y = (row + 2) * cell_height  # Offset by one row to accommodate the weekday images
            final_image.paste(img_resized, (x, y))


    return final_image,collected_labels

def print_progress(generator, label):
    images_with_labels = []
    for img, lbl in generator:
        images_with_labels.append((img, lbl))  # Store as a tuple (image, label)
        print(f"{label}: {len(images_with_labels)} images processed")
        if len(images_with_labels) >= NUM_IMAGES_TO_SAVE:
            break
    return images_with_labels  # Return the list of tuples

name_images_with_labels = print_progress(nameGenerator, "Name")
start_time_images_with_labels = print_progress(startTimeGenerator, "Start Time")
end_time_images_with_labels = print_progress(endTimeGenerator, "End Time")
location_images_with_labels = print_progress(locationGenerator, "Location")
days_images_with_labels = print_progress(daysGenerator, "Day")
dates_images_with_labels = print_progress(datesGenerator, "Date")
all_rota_data = []


for i in range(NUMBER_OF_ROTAS):
    final_rota_image, current_labels  = create_final_rota_image(
    name_images_with_labels,
    start_time_images_with_labels,
    end_time_images_with_labels,
    location_images_with_labels,
    days_images_with_labels,
    dates_images_with_labels,
    cell_width=randint(280, 320),
    cell_height=randint(140, 160),
    rows=randint(5,25)  # Adjust rows based on the data size
    )
    current_index += 1
    final_rota_image.save(f'output/rotapicture_{current_index}.png')

    filename = f'rotapicture_{current_index}.png'
    final_rota_image.save(f'output/{filename}')

    rota_entry = f'Filename: {filename}\n'
    for name, shifts in current_labels.items():
        rota_entry += f'{name}:\n'
        for shift in shifts:
            rota_entry += f'    {shift["start_time"]}, {shift["end_time"]}, {shift["location"]}, {shift["day"]}, {shift["date"]}\n'

    all_rota_data.append(rota_entry)
    # Gather details for the current shift


with open('output/labels.txt', 'w') as text_file:
    for entry in all_rota_data:
        text_file.write(entry + '\n')

f.close()



