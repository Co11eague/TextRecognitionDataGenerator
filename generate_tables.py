from random import randint

from trdg.background_generator import image
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


TABLE_TYPE = 3

#Settings
NUMBER_OF_ROTAS = 2
NUM_IMAGES_TO_SAVE = 100
CELL_WIDTH_RANGE = (280, 320)
CELL_HEIGHT_RANGE = (140, 160)
ROWS_RANGE = (5, 25)

data = []
fake = Faker()

for _ in range(NUM_IMAGES_TO_SAVE):
    start_time = datetime.strptime(fake.time(pattern="%H:%M"), "%H:%M")
    end_time = start_time + timedelta(hours=randint(3, 10), minutes=randint(0, 59))

    date_str = fake.date()

    # Convert the string to a datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Format it to "YYYY-MM-DD (Day)"
    date_and_day = f"{date.strftime('%Y-%m-%d')} ({date.strftime('%A')})"

    data.append([
        fake.name(),
        start_time.strftime("%H:%M"),
        end_time.strftime("%H:%M"),
        fake.city(),
        fake.day_of_week(),
        fake.date(),
        date_and_day
    ])

df = pd.DataFrame(data, columns=["name", "start_time", "end_time", "location", "fake_day", "fake_date", "date_and_day"])

# Cleanup and generate unique combinations
def cleanup(values):
    filtered = [x for x in values if pd.notna(x)]
    unique_values = list(set(filtered))
    print(f"Cleaned {len(values) - len(filtered)} NaN values and removed {len(filtered) - len(unique_values)} duplicates")
    return unique_values


names = cleanup(df["name"].tolist())
start_times = cleanup(df["start_time"].tolist())
end_times = cleanup(df["end_time"].tolist())
locations = cleanup(df["location"].tolist())
fake_days = cleanup(df["fake_day"].tolist())
fake_dates = cleanup(df["fake_date"].tolist())
date_and_day = cleanup(df["date_and_day"].tolist())


# Set up generators for images
def create_generator(data, label):
    print(f"{label}: {len(data)} images to process")
    return GeneratorFromStrings(random.sample(data, min(len(data), 10000)), random_blur=True, random_skew=True)



name_gen = create_generator(names, "Names")
start_time_gen = create_generator(start_times, "Start Times")
end_time_gen = create_generator(end_times, "End Times")
location_gen = create_generator(locations, "Locations")
day_gen = create_generator(fake_days, "Days")
date_gen = create_generator(fake_dates, "Dates")
date_and_day_gen = create_generator(date_and_day, "Date and Day")

# Save generated images and their labels
def save_images(generator, label):
    images = []
    for img, lbl in generator:
        images.append((img, lbl))
        print(f"{label}: {len(images)} images processed")
        if len(images) >= NUM_IMAGES_TO_SAVE:
            break
    return images

# Load images and labels
name_images = save_images(name_gen, "Names")
start_images = save_images(start_time_gen, "Start Times")
end_images = save_images(end_time_gen, "End Times")
location_images = save_images(location_gen, "Locations")
day_images = save_images(day_gen, "Days")
date_images = save_images(date_gen, "Dates")
date_and_day_images = save_images(date_and_day_gen, "Date and Day")

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


def create_final_rota_image(name_images_with_labels, start_time_images_with_labels, end_time_images_with_labels, location_images_with_labels, days_images_with_labels, dates_images_with_labels, date_and_day_images_with_labels, cell_width,
                            cell_height, rows, table_type):
    # Set up table size
    collected_labels = {}
    num_days = randint(5,15)  # Number of days to display
    if table_type == 2:
        cols = 6
    elif table_type == 3:
        cols = num_days
        rows = 5
    else:
        cols = 1 + (num_days * 2) + 1  # 1 Name column, 2 columns per day (Start & End), 1 Location column



    table_width = cell_width * cols
    table_height = cell_height * (rows)  # Adjust height for the weekday row
    background_colors = ["#f5f5f5", "#e0e0e0", "#ffffff", "#f0f8ff"]
    final_image = Image.new("RGB", (table_width, table_height), color=random.choice(background_colors))
    draw = ImageDraw.Draw(final_image)

    random.shuffle(dates_images_with_labels)

    if table_type == 1:
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
    elif table_type == 2:
        for row in range(rows):  # +2 for the date and weekday rows
            draw.line([(0, row * cell_height), (table_width, row * cell_height)], fill="black", width=20)  # Horizontal lines
        for col in range(cols):
            draw.line([(col * cell_width, 0), (col * cell_width, table_height)], fill="black", width=20)  # Vertical lines


        for row in range(rows):
            name_image, name_label = name_images_with_labels[random.randint(0, len(name_images_with_labels) - 1)]
            day_image, day_label = days_images_with_labels[random.randint(0, len(days_images_with_labels) - 1)]
            date_image, date_label = dates_images_with_labels[random.randint(0, len(dates_images_with_labels) - 1)]
            location_image, location_label = location_images_with_labels[random.randint(0, len(location_images_with_labels) - 1)]

            index = random.randint(0, len(start_time_images_with_labels) - 1)
            start_time_image, start_time_label = start_time_images_with_labels[index]
            end_time_image, end_time_label = end_time_images_with_labels[index]


            if name_label not in collected_labels:
                collected_labels[name_label] = []
            # Get images for the current row

            images_in_row = [date_image, day_image, name_image, location_image, start_time_image, end_time_image]

            collected_labels[name_label].append({
                "start_time": start_time_label,
                 "end_time": end_time_label,
                "location": location_label,
                "day": day_label,  # This should correspond to the day in this row
                "date": date_label
            })

            for col in range(cols):
                img = images_in_row[col]
                img_resized = img.resize((cell_width - 10, cell_height - 10))  # Resize to fit in the cell
                x = col * cell_width
                y = (row) * cell_height  # Offset by one row to accommodate the weekday images
                final_image.paste(img_resized, (x, y))

    elif table_type == 3:
        names = []
        location = []
        start_time = []
        end_time = []
        days = []
        dates = []
        for row in range(rows):  # +2 for the date and weekday rows
            draw.line([(0, row * cell_height), (table_width, row * cell_height)], fill="black", width=20)  # Horizontal lines
        for col in range(cols):
            draw.line([(col * cell_width, 0), (col * cell_width, table_height)], fill="black", width=20)  # Vertical lines


        date_and_day_labels = []

        # Place the date images in the first row
        for i,  (date_and_day_image, date_and_day_label) in enumerate(date_and_day_images_with_labels):
            if i == cols:
                break
            resized_date_image = date_and_day_image.resize((cell_width - 10, cell_height - 10))  # Resize to fit with padding
            final_image.paste(resized_date_image, (i * cell_width, 0))  # Position date images
            date_and_day_labels.append(date_and_day_label)

        for i, (name_image, name_label) in enumerate(name_images_with_labels):
            if i == cols:
                break
            if name_label not in collected_labels:
                collected_labels[name_label] = []

            date_part = date_and_day_labels[i].split(' ')[0]  # Get the date part
            day_part = date_and_day_labels[i].split('(')[-1].strip(') ')  # Get the day part
            days.append(day_part)
            dates.append(date_part)


            resized_name_image = name_image.resize(
                (cell_width - 10, cell_height - 10))  # Resize to fit with padding
            final_image.paste(resized_name_image, (i * cell_width, cell_height))  # Position date images
            names.append(name_label)

        variables = [location_images_with_labels, start_time_images_with_labels, end_time_images_with_labels]

        images_in_row = []


        # Step 2: Place each word image into its corresponding cell in the grid
        for i in range(rows - 2):


            # Add start and end times for each day
            for j in range(cols):
                image, label = variables[i][random.randint(0, len(variables[i]) - 1)]

                images_in_row.append(image)

                if (i == 0):
                    location.append(label)
                elif (i == 1):
                    start_time.append(label)
                elif (i == 2):
                    end_time.append(label)

            for row in range(rows - 2):
                for col in range(cols):
                    index = (row * cols) + col

                    if index < len(images_in_row):  # Check if the index is within bounds
                        img = images_in_row[(row * cols) + col]
                        img_resized = img.resize((cell_width - 10, cell_height - 10))  # Resize to fit in the cell
                        x = col * cell_width
                        y = (row + 2) * cell_height  # Offset by one row to accommodate the weekday images
                        final_image.paste(img_resized, (x, y))
        for col in range(cols):
            collected_labels[names[col]].append({
                "start_time": start_time[col],
                "end_time": end_time[col],
                "location": location[col],
                "day": days[col],  # This should correspond to the day in this row
                "date": dates[col]
            })

    return final_image,collected_labels

all_rota_data = []

with open('output/labels.txt', 'w') as label_file:
    current_index = len(os.listdir('output')) - 1
    for _ in range(NUMBER_OF_ROTAS):
        rota_img, labels = create_final_rota_image(
            name_images, start_images, end_images, location_images, day_images, date_images, date_and_day_images,
            cell_width=randint(*CELL_WIDTH_RANGE), cell_height=randint(*CELL_HEIGHT_RANGE), rows=randint(*ROWS_RANGE), table_type=TABLE_TYPE
        )
        filename = f'output/rotapicture_{current_index}.png'
        rota_img.save(filename)

        entry = f'Filename: rotapicture_{current_index}.png\n'
        for name, shifts in labels.items():
            entry += f"{name}:\n"
            for shift in shifts:
                # Write each shift's available fields
                entry += "    " + ",".join(f"{v}" for k, v in shift.items()) + "\n"


        label_file.write(entry)
        current_index += 1

f.close()



### 9 5 3 4