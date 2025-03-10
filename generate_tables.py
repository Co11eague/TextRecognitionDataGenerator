import json
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
from io import BytesIO

TABLE_TYPE = 6

# Settings
NUMBER_OF_ROTAS = 1000
NUM_IMAGES_TO_SAVE = 2000
CELL_WIDTH_RANGE = (150, 500)
CELL_HEIGHT_RANGE = (100, 300)
ROWS_RANGE = (5, 15)
COLUMNS_RANGE = (5, 15)
TABLE_TYPE_RANGE = (1, 6)

data = []
fake = Faker()
SEPARATE = True

for _ in range(NUM_IMAGES_TO_SAVE):
	start_time = datetime.strptime(fake.time(pattern="%H:%M"), "%H:%M")
	end_time = start_time + timedelta(hours=randint(3, 10), minutes=randint(0, 59))
	name = fake.first_name()

	start_hour = str(start_time.strftime("%I")).lstrip('0')  # Get hour in 12-hour format without leading zeros
	end_hour = str(end_time.strftime("%I")).lstrip('0')  # Get hour in 12-hour format without leading zeros

	# Create the concise time range format
	concise_time_range = f"{name} ({start_hour}-{end_hour})"
	no_name_concise_time_range = f"({start_hour}-{end_hour})"

	date_str = fake.date()

	# Convert the string to a datetime object
	date = datetime.strptime(date_str, '%Y-%m-%d')

	# Format it to "YYYY-MM-DD (Day)"
	date_and_day = f"{date.strftime('%Y-%m-%d')} ({date.strftime('%A')})"

	time_range = f"{start_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}"

	data.append([
		name,
		start_time.strftime("%H:%M"),
		end_time.strftime("%H:%M"),
		fake.city(),
		fake.day_of_week(),
		fake.date(),
		date_and_day,
		time_range,
		concise_time_range,
		no_name_concise_time_range
	])

df = pd.DataFrame(data, columns=["name", "start_time", "end_time", "location", "fake_day", "fake_date", "date_and_day",
                                 "time_range", "concise_time_range", "no_name_concise_time_range"])


# Cleanup and generate unique combinations
def cleanup(values):
	filtered = [x for x in values if pd.notna(x)]
	unique_values = list(set(filtered))
	print(
		f"Cleaned {len(values) - len(filtered)} NaN values and removed {len(filtered) - len(unique_values)} duplicates")
	return unique_values


names = cleanup(df["name"].tolist())
start_times = cleanup(df["start_time"].tolist())
end_times = cleanup(df["end_time"].tolist())
locations = cleanup(df["location"].tolist())
fake_days = cleanup(df["fake_day"].tolist())
fake_dates = cleanup(df["fake_date"].tolist())
date_and_day = cleanup(df["date_and_day"].tolist())
time_range = cleanup(df["time_range"].tolist())
concise_time_range = cleanup(df["concise_time_range"].tolist())
no_name_concise_time_range = cleanup(df["no_name_concise_time_range"].tolist())


# Set up generators for images
def create_generator(data, label):
	print(f"{label}: {len(data)} images to process")




	return GeneratorFromStrings(random.sample(data, len(data)), image_dir=r"C:\Users\UniWork\Desktop\final-year-project\TextRecognitionDataGenerator\trdg\images"

	                            )


name_gen = create_generator(names, "Names")
start_time_gen = create_generator(start_times, "Start Times")
end_time_gen = create_generator(end_times, "End Times")
location_gen = create_generator(locations, "Locations")
day_gen = create_generator(fake_days, "Days")
date_gen = create_generator(fake_dates, "Dates")
date_and_day_gen = create_generator(date_and_day, "Date and Day")
time_range_gen = create_generator(time_range, "Time Ranges")
concise_time_range_gen = create_generator(concise_time_range, "Concise Time Ranges")
no_name_concise_time_range_gen = create_generator(no_name_concise_time_range, "No Name Concise Time Ranges")


def save_images(generator, label):
	saved_images = []
	for img, lbl in generator:
		# Compress the image by converting it to JPEG format with quality setting
		buffer = BytesIO()
		if img is not None:
			img.save(buffer, format="JPEG")  # Adjust quality to balance compression
			compressed_img = Image.open(buffer)

			# Append the compressed image and its label to the list
			saved_images.append((compressed_img, lbl))

			print(f"{label}: {len(saved_images)} images processed")
		if len(saved_images) >= NUM_IMAGES_TO_SAVE:
			break

	return saved_images


# Load images and labels
name_images = save_images(name_gen, "Names")
start_images = save_images(start_time_gen, "Start Times")
end_images = save_images(end_time_gen, "End Times")
location_images = save_images(location_gen, "Locations")
day_images = save_images(day_gen, "Days")
date_images = save_images(date_gen, "Dates")
date_and_day_images = save_images(date_and_day_gen, "Date and Day")
time_range_images = save_images(time_range_gen, "Time Ranges")
concise_time_range_images = save_images(concise_time_range_gen, "Concise Time Ranges")
no_name_concise_time_range_images = save_images(no_name_concise_time_range_gen, "No Name Concise Time Ranges")

images = []
labels = {}


def create_final_rota_image(name_images_with_labels, start_time_images_with_labels, end_time_images_with_labels,
                            location_images_with_labels, days_images_with_labels, dates_images_with_labels,
                            date_and_day_images_with_labels, time_range_images_with_labels,
                            concise_time_range_images_with_labels, no_name_concise_time_range_images_with_labels,
                            cell_width,
                            cell_height, rows, columns, table_type):
	table_dimensions = {
		2: {"cols": 6, "rows": rows},
		3: {"cols": columns, "rows": 5},
		4: {"cols": 5, "rows": rows},
		5: {"cols": 2 + columns, "rows": rows},
		6: {"cols": 1 + columns, "rows": rows + 1},
		"default": {"cols": 1 + (columns * 2) + 1, "rows": rows + 2}
	}
	collected_labels = []
	final_collected_labels = {}

	dims = table_dimensions.get(table_type, table_dimensions["default"])
	cols, rows = dims["cols"], dims["rows"]

	table_width = cell_width * cols
	table_height = cell_height * rows  # Adjust height for the weekday row

	background_colors = ["#f5f5f5", "#e0e0e0", "#ffffff", "#f0f8ff"]
	final_image = Image.new("RGB", (table_width, table_height), color=random.choice(background_colors))
	draw = ImageDraw.Draw(final_image)

	def draw_table_outline(draw_function, num_rows, num_cols, cell_width_function, cell_height_function, width, height):
		for num_rows_index in range(num_rows):
			draw_function.line(
				[(0, num_rows_index * cell_height_function), (width, num_rows_index * cell_height_function)],
				fill="black", width=20)
		for num_cols_index in range(num_cols):
			draw_function.line(
				[(num_cols_index * cell_width_function, 0), (num_cols_index * cell_width_function, height)],
				fill="black", width=20)

	def resize_and_paste(supplied_image, x, y):
		img_resized = supplied_image.resize((cell_width - 10, cell_height - 10))
		final_image.paste(img_resized, (x, y))

	def add_labels(label_name, label_day, label_date, label_location, label_start_time, label_end_time):
		if label_name not in final_collected_labels:
			final_collected_labels[label_name] = []
		if label_location is not None:
			final_collected_labels[label_name].append({
				"start_time": label_start_time,
				"end_time": label_end_time,
				"location": label_location,
				"day": label_day,
				"date": label_date
			})
		else:
			final_collected_labels[label_name].append({
				"start_time": label_start_time,
				"end_time": label_end_time,
				"day": label_day,
				"date": label_date
			})

	def get_random_image_label(image_label_list):
		return image_label_list[random.randint(0, len(image_label_list) - 1)]

	if table_type == 1:
		date_labels = []
		day_labels = []
		random.shuffle(dates_images_with_labels)
		random.shuffle(days_images_with_labels)

		# Draw the outline for the table
		draw_table_outline(draw, rows, cols, cell_width, cell_height, table_width, table_height)

		# Place the date images in the first row
		for index, (date_image, date_label) in enumerate(dates_images_with_labels[:columns]):
			resized_date_image = date_image.resize(
				(cell_width * 2 - 10, cell_height - 10))  # Resize to fit with padding
			final_image.paste(resized_date_image, (index * cell_width * 2 + cell_width, 0))  # Position date images
			date_labels.append(date_label)
			collected_labels.append(date_label)
		collected_labels.append("---")

		# Step 1: Place the weekday images in the second row
		for index, (day_image, day_label) in enumerate(days_images_with_labels[:columns]):
			resized_day_image = day_image.resize((cell_width * 2 - 10, cell_height - 10))  # Resize to fit with padding
			final_image.paste(resized_day_image,
			                  (index * cell_width * 2 + cell_width, cell_height))  # Position weekday images
			collected_labels.append(day_label)
			day_labels.append(day_label)
		collected_labels.append("---")

		# Step 2: Place each word image into its corresponding cell in the grid
		for row in range(rows - 2):
			name_image, name_label = get_random_image_label(name_images_with_labels)

			images_in_row = [name_image]

			location_image, location_label = get_random_image_label(location_images_with_labels)
			collected_labels.append(name_label)

			# Add start and end times for each day
			for index in range(columns):
				time_index = random.randint(0, len(start_time_images_with_labels) - 1)
				start_time_image, start_time_label = start_time_images_with_labels[time_index]
				end_time_image, end_time_label = end_time_images_with_labels[time_index]

				images_in_row.append(start_time_image)  # Start time for day i
				images_in_row.append(end_time_image)  # End time for day i

				collected_labels.append(start_time_label)
				collected_labels.append(end_time_label)
				add_labels(name_label, day_labels[index], date_labels[index], location_label, start_time_label,
				           end_time_label)

			images_in_row.append(location_image)  # Location column
			collected_labels.append(location_label)
			collected_labels.append("---")

			for col in range(cols):
				img = images_in_row[col]
				resize_and_paste(img, col * cell_width, (row + 2) * cell_height)


	elif table_type == 2:
		draw_table_outline(draw, rows, cols, cell_width, cell_height, table_width, table_height)

		for row in range(rows):
			name_image, name_label = get_random_image_label(name_images_with_labels)
			day_image, day_label = get_random_image_label(days_images_with_labels)
			date_image, date_label = get_random_image_label(dates_images_with_labels)
			location_image, location_label = get_random_image_label(location_images_with_labels)

			index = random.randint(0, len(start_time_images_with_labels) - 1)
			start_time_image, start_time_label = start_time_images_with_labels[index]
			end_time_image, end_time_label = end_time_images_with_labels[index]
			images_in_row = [date_image, day_image, name_image, location_image, start_time_image, end_time_image]
			collected_labels.append(date_label)
			collected_labels.append(day_label)
			collected_labels.append(name_label)
			collected_labels.append(location_label)
			collected_labels.append(start_time_label)
			collected_labels.append(end_time_label)
			collected_labels.append("---")

			add_labels(name_label, day_label, date_label, location_label, start_time_label, end_time_label)

			for col in range(cols):
				img = images_in_row[col]
				resize_and_paste(img, col * cell_width, row * cell_height)

	elif table_type == 3:
		random.shuffle(date_and_day_images_with_labels)
		names_labels = []
		location_labels = []
		start_time_labels = []
		end_time_labels = []
		days_labels = []
		dates_labels = []
		date_and_day_labels = []
		variables = [location_images_with_labels, start_time_images_with_labels, end_time_images_with_labels]
		images_in_row = []

		draw_table_outline(draw, rows, cols, cell_width, cell_height, table_width, table_height)

		# Place the date images in the first row
		for index, (date_and_day_image, date_and_day_label) in enumerate(date_and_day_images_with_labels[:cols]):
			resize_and_paste(date_and_day_image, index * cell_width, 0)
			date_and_day_labels.append(date_and_day_label)
			collected_labels.append(date_and_day_label)

		collected_labels.append("---")

		random.shuffle(name_images_with_labels)

		for index, (name_image, name_label) in enumerate(name_images_with_labels[:cols]):
			date_part = date_and_day_labels[index].split(' ')[0]  # Get the date part
			day_part = date_and_day_labels[index].split('(')[-1].strip(') ')  # Get the day part

			days_labels.append(day_part)
			dates_labels.append(date_part)

			resize_and_paste(name_image, index * cell_width, cell_height)

			names_labels.append(name_label)
			collected_labels.append(name_label)
		collected_labels.append("---")
		# Step 2: Place each word image into its corresponding cell in the grid
		for index in range(rows - 2):

			# Add start and end times for each day
			for j in range(cols):
				list_image, label = get_random_image_label(variables[index])

				images_in_row.append(list_image)
				collected_labels.append(label)

				if j == cols - 1:
					collected_labels.append("---")

				if index == 0:
					location_labels.append(label)
				elif index == 1:
					start_time_labels.append(label)
				elif index == 2:
					end_time_labels.append(label)

			for row in range(rows - 2):
				for col in range(cols):
					index = (row * cols) + col

					if index < len(images_in_row):  # Check if the index is within bounds
						img = images_in_row[(row * cols) + col]
						resize_and_paste(img, col * cell_width, (row + 2) * cell_height)
		for col in range(cols):
			add_labels(names_labels[col], days_labels[col], dates_labels[col], location_labels[col],
			           start_time_labels[col], end_time_labels[col])
	elif table_type == 4:
		draw_table_outline(draw, rows, cols, cell_width, cell_height, table_width, table_height)

		for row in range(rows):
			name_image, name_label = get_random_image_label(name_images_with_labels)
			day_image, day_label = get_random_image_label(days_images_with_labels)
			date_image, date_label = get_random_image_label(dates_images_with_labels)
			location_image, location_label = get_random_image_label(location_images_with_labels)
			time_range_image, time_range_label = get_random_image_label(time_range_images_with_labels)

			images_in_row = [date_image, day_image, name_image, location_image, time_range_image]

			start_time_str, end_time_str = time_range_label.split(" - ")
			collected_labels.append(date_label)
			collected_labels.append(day_label)
			collected_labels.append(name_label)
			collected_labels.append(location_label)
			collected_labels.append(time_range_label)
			collected_labels.append("---")
			add_labels(name_label, day_label, date_label, location_label, start_time_str, end_time_str)

			for col in range(cols):
				img = images_in_row[col]
				resize_and_paste(img, col * cell_width, row * cell_height)
	elif table_type == 5:
		date_labels = []
		day_labels = []

		draw_table_outline(draw, rows, cols, cell_width, cell_height, table_width, table_height)

		for row in range(rows):
			images_in_row = []

			day_image, day_label = get_random_image_label(days_images_with_labels)
			date_image, date_label = get_random_image_label(dates_images_with_labels)

			for index in range(cols):
				if index == 0:
					images_in_row.append(date_image)
					collected_labels.append(date_label)
					date_labels.append(date_label)
				elif index == 1:
					images_in_row.append(day_image)
					collected_labels.append(day_label)
					day_labels.append(day_label)
				else:
					concise_time_range_image, concise_time_range_label = get_random_image_label(
						concise_time_range_images_with_labels)
					images_in_row.append(concise_time_range_image)
					collected_labels.append(concise_time_range_label)

					name_part, time_range_part = concise_time_range_label.split(" (")
					time_range_part = time_range_part.strip(")")  # Remove the closing parenthesis

					# Split the time range to get start and end times
					start_time_labels, end_time_labels = time_range_part.split("-")

					# Strip any leading/trailing whitespace (if needed)
					name_real = name_part.strip()
					start_time_labels = start_time_labels.strip()
					end_time_labels = end_time_labels.strip()

					add_labels(name_real, day_labels[row], date_labels[row], None, start_time_labels, end_time_labels)
			collected_labels.append("---")
			for col in range(cols):
				img = images_in_row[col]
				resize_and_paste(img, col * cell_width, row * cell_height)
	elif table_type == 6:
		date_and_day_labels = []

		draw_table_outline(draw, rows, cols, cell_width, cell_height, table_width, table_height)
		random.shuffle(date_and_day_images_with_labels)

		# Place the date images in the first row
		for index, (date_and_day_image, date_and_day_label) in enumerate(date_and_day_images_with_labels[:cols - 1]):
			resize_and_paste(date_and_day_image, (index + 1) * cell_width, 0)
			collected_labels.append(date_and_day_label)
			date_and_day_labels.append(date_and_day_label)

		collected_labels.append("---")

		for row in range(1, rows):
			images_in_row = []

			name_image, name_label = get_random_image_label(name_images_with_labels)

			for index in range(cols):
				if index == 0:
					images_in_row.append(name_image)
					collected_labels.append(name_label)

				else:
					no_name_concise_time_range_image, no_name_concise_time_range_label = get_random_image_label(
						no_name_concise_time_range_images_with_labels)
					images_in_row.append(no_name_concise_time_range_image)
					collected_labels.append(no_name_concise_time_range_label)

					# Split the time range to get start and end times
					start_time_labels, end_time_labels = no_name_concise_time_range_label.strip("()").split("-")

					date_part = date_and_day_labels[index - 1].split(' ')[0]  # Get the date part
					day_part = date_and_day_labels[index - 1].split('(')[-1].strip(') ')  # Get the day part

					add_labels(name_label, day_part, date_part, None, start_time_labels, end_time_labels)

			collected_labels.append("---")
			for col in range(cols):
				img = images_in_row[col]
				resize_and_paste(img, col * cell_width, row * cell_height)

	return final_image, collected_labels, final_collected_labels


all_rota_data = []


def format_shifts(json_data):
	formatted_lines = []

	for name, shifts in json_data.items():
		# Construct the formatted line for each person
		shift_strings = [
			f"{shift['start_time']},{shift['end_time']},{shift['location']},{shift['day']},{shift['date']}"
			if 'location' in shift else f"{shift['start_time']},{shift['end_time']},{shift['day']},{shift['date']}"
			for shift in shifts
		]
		formatted_line = f"{name}: " + ", ".join(shift_strings)
		formatted_lines.append(formatted_line)
	formatted_lines.append("\n")

	# Join all formatted lines into one long line
	return " ".join(formatted_lines)


with open('../deep-text-recognition-benchmark/valid_output/labels.txt', "a") as label_file:
	current_index = len(os.listdir('../deep-text-recognition-benchmark/valid_output')) - 1
	for i in range(1, NUMBER_OF_ROTAS + 1):
		print("Image saved: ", current_index)
		print("Loop: ", i)
		if not SEPARATE:

			rota_img, labels, final_labels = create_final_rota_image(
				name_images, start_images, end_images, location_images, day_images, date_images, date_and_day_images,
				time_range_images, concise_time_range_images, no_name_concise_time_range_images,
				cell_width=randint(*CELL_WIDTH_RANGE), cell_height=randint(*CELL_HEIGHT_RANGE),
				rows=randint(*ROWS_RANGE),
				columns=randint(*COLUMNS_RANGE), table_type=randint(*TABLE_TYPE_RANGE)
			)
			filename = f'../deep-text-recognition-benchmark/valid_output/rotapicture_{current_index}.png'
			rota_img.save(filename, format="JPEG")

			formatted_labels = []

			entry = f'Filename: rotapicture_{current_index}.png\n'  # Filename on a new line

			# Format each label by splitting and joining with "|"
			for label in labels:
				if label == "---":
					# If the label is a row separator, just append it as is
					formatted_labels.append(label)
				else:
					formatted_labels.append(label)

			# Join all formatted labels into one string, separated by space and rows by "---"
			entry += "|".join(formatted_labels)  # Separate labels by space

			# Write to the label file
			label_file.write(entry.strip() + "\n")  # Ensure no trailing space or newline at the end

			current_index += 1

			with open('./out/final_labels.txt', "a") as final_label_file:
				final_label_file.write(format_shifts(final_labels))
		else:
			IMAGES = [name_images, start_images, end_images, location_images, day_images, date_images,
			          date_and_day_images,
			          time_range_images, concise_time_range_images, no_name_concise_time_range_images]

			filename = f'../deep-text-recognition-benchmark/valid_output/rotapicture_{current_index}.png'
			image, label = IMAGES[(i - 1) % 10][randint(0, (NUM_IMAGES_TO_SAVE - 1))]
			image.save(filename, format="JPEG")

			formatted_labels = []

			entry = f'Filename: rotapicture_{current_index}.png\n'  # Filename on a new line

			# Join all formatted labels into one string, separated by space and rows by "---"
			entry += label  # Separate labels by space

			# Write to the label file
			label_file.write(entry.strip() + "\n")  # Ensure no trailing space or newline at the end

			current_index += 1
