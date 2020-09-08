import os
import cv2
import shutil
import time
from os.path import join
from os import listdir
import random
import numpy as np

def get_images_from_video(video_path, dir_to_result_images,
                          class_prefix, every_n_image=1, file_extention='png',
                          pad_zeros=True, num_zeros=5):
  force_create_folder(dir_to_result_images)
  
  videocap = cv2.VideoCapture(video_path)
  tic = time.time()

  success, image = videocap.read()
  count = 0
  while success:
    count += 1
    if count % every_n_image == 0:
      if pad_zeros:
        zero_pad_count = str(count).zfill(num_zeros)
        cv2.imwrite(os.path.join(dir_to_result_images, '{}frame{}.{}'
        .format(class_prefix, zero_pad_count, file_extention)), image)
      else :
        cv2.imwrite(os.path.join(dir_to_result_images, '{}frame{}.{}'
      .format(class_prefix, count, file_extention)), image)
    success, image = videocap.read()
    
  print('time for video was ', time.time() - tic)

def copy_partial_files(src, dest, ratio):
  files_to_copy = [f for f in os.listdir(src) if random.uniform(0, 1) < ratio]
  for f in files_to_copy:
    shutil.copy2(join(src, f), join(dest, f))

def copy_labeled_images(src, dest, image_extention='jpg'):
  text_files = get_text_files_from_dir(src)
  for text_file_name in text_files:
    image_file_name = text_file_name[:-4] + '.' + image_extention
    shutil.copy2(join(src, text_file_name), join(dest, text_file_name))
    shutil.copy2(join(src, image_file_name), join(dest, image_file_name))

def copy_labels_to_dir(labels_dir, images_dir):
  text_files = get_text_files_from_dir(labels_dir)
  print(len(os.listdir(images_dir)))
  for text_file in text_files:
    shutil.copy2(join(labels_dir, text_file), join(images_dir, text_file))
  print(len(os.listdir(images_dir)))

def get_text_files_from_dir(dir_path):
  return [f for f in os.listdir(dir_path) if '.txt' in f]

def force_create_folder(dir_path):
  if os.path.exists(dir_path):
    shutil.rmtree(dir_path)
  os.mkdir(dir_path)

def train_test_split(image_src, label_src, train_dir, test_dir, test_ratio=0.2, image_extention='jpg'):
  force_create_folder(train_dir)
  force_create_folder(test_dir)
  
  force_create_folder(join(train_dir, 'labels'))
  force_create_folder(join(train_dir, 'images'))
  force_create_folder(join(test_dir, 'labels'))
  force_create_folder(join(test_dir, 'images'))
  
  text_files = get_text_files_from_dir(label_src)
  for text_file in text_files:
    image_file = text_file[:-4] + '.' + image_extention
    text_file_dest = None
    image_file_dest = None
    if random.uniform(0, 1) < test_ratio:
      text_file_dest = join(test_dir, 'labels', text_file)
      image_file_dest = join(test_dir, 'images', image_file)
    else :
      text_file_dest = join(train_dir, 'labels', text_file)
      image_file_dest = join(train_dir, 'images', image_file)

    shutil.copy2(join(label_src, text_file), text_file_dest)
    shutil.copy2(join(image_src, image_file), image_file_dest)
    
def change_class_labels(dir, new_class):
  all_text_files = [f for f in os.listdir(dir) if f.endswith('.txt')]
  for f in all_text_files:
    old_path = join(dir, f)
    new_path = join(dir, f + '.luka')
    with open(old_path, 'r') as reader:
      with open(new_path, 'w') as writer:
        for line in reader.readlines():
          line_list = line.split() # 5 entries
          line_list[0] = new_class
          new_line = ' '.join(line_list) + '\n'
          writer.write(new_line)
    os.remove(old_path)
    os.rename(new_path, old_path)

def copy_n_items(src, dest, n_items):
  # TEST
  print(len(os.listdir(dest)))
  
  for f in os.listdir(src)[:n_items]:
    shutil.copy2(join(src, f), join(dest, f))
  
  # TEST
  print(len(os.listdir(dest)))

def get_text_paths_for_class(path, cls):
  class_keyword = str(cls) + 'frame'
  return [join(path, filename) for filename in listdir(path)
   if filename.endswith('.txt') and class_keyword in filename]


def get_index_from_file_name(filename):
  number_index = filename.find('frame') + 5
  return filename[number_index:filename.find('.txt')]

def get_coordinates_with_file_index_from_textfiles(textfiles_list):
  coords_with_textfiles = []
  for textfile in textfiles_list:
    with open(textfile, 'r') as reader:
      curr_coords = []
      lines = reader.readlines()
      for line in lines:
        parts = line.split()
        curr_coords.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
      index = get_index_from_file_name(textfile)
      coords_with_textfiles.append((index, curr_coords))
  return coords_with_textfiles

def get_coordinates_from_textfiles(textfiles_list):
  coords = []
  for textfile in textfiles_list:
    with open(textfile, 'r') as reader:
      curr_coords = []
      for line in reader.readlines():
        parts = line.split()
        curr_coords.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
      coords.append(curr_coords)
  return coords

def get_image_path_from_text_path(f, old_parent_name, new_parent_name):
  return f[:-4].replace(old_parent_name, new_parent_name) + '.jpg'


# TRACKING
def calculate_euclidean_distances(x, y):
  x_squared = np.sum(x ** 2, axis=1)
  y_squared = np.sum(y ** 2, axis=1)
  diff = - 2 * np.dot(x, y.T)
  return (((diff + y_squared).T + x_squared) ** 0.5).T

def fill_rest_ids(num_nuts, ids_map, last_id):
  for i in range(num_nuts):
    if i not in ids_map:
      last_id += 1
      ids_map[i] = last_id
  return last_id

def give_ids(prev_coords, prev_ids, curr_coords, vy=0.02, last_id=0):
#  TODO might be extra to convert in numpy array curr coords
  pred_curr_coords = np.array(prev_coords)
  curr_coords = np.array(curr_coords)
  num_prev_nuts = pred_curr_coords.shape[0]
  num_curr_nuts = curr_coords.shape[0]
  curr_ids = {}

  if num_prev_nuts is 0:
    last_id = fill_rest_ids(num_curr_nuts, curr_ids, last_id)
    return curr_ids, last_id
    

  pred_curr_coords[:, 1] += vy

  #prev nuts row, curr nuts column
  dists = calculate_euclidean_distances(pred_curr_coords[:, :2], curr_coords[:, :2]) 
  # mit be done parrallel, uuid might make it even more parrallel
  for i in range(min(num_prev_nuts, num_curr_nuts)):
    curr_mins = np.min(dists, axis=1)
    prev_nut_index = np.argmin(curr_mins) # prev nut index
    curr_nut_index = np.argmin(dists[prev_nut_index])

    dists[prev_nut_index, :] = 100
    dists[:, curr_nut_index] = 100

    prev_id = prev_ids[prev_nut_index]
    if prev_id == -1:
      last_id += 1
      curr_ids[curr_nut_index] = last_id
    else:
      curr_ids[curr_nut_index] = prev_id

  if num_curr_nuts > num_prev_nuts:
    last_id = fill_rest_ids(num_curr_nuts, curr_ids, last_id)
  return curr_ids, last_id

def get_labeled_image(coords, ids_map, image_path):
  img = cv2.imread(image_path)
  for i, coord in enumerate(coords):
    x, y, w, h = coord
    x = int((x - w / 2) * 640)
    w = int(w * 640)
    y = int((y - h / 2) * 480)
    h = int(h * 480)
    cv2.rectangle(img, (x, y, w, h), (36,255,12), 1)
    cv2.putText(img, 'ID:' + str(ids_map[i]), (x - 55, int(y + h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
  return img

def record_tracking(image_path_list, coords, video_path, fps, vy = 0.02):
  if os.path.isfile(video_path):
    os.remove(video_path)
  prev_coords = []
  prev_ids = {}
  last_id = 0
  fourcc = 'mp4v'
  vid_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*fourcc), fps, (640, 480))
  for i in range(len(image_path_list)):
    curr_coords = coords[i]
    curr_ids, last_id = give_ids(prev_coords, prev_ids, curr_coords, vy, last_id)

    img = get_labeled_image(curr_coords, curr_ids, image_path_list[i])
    vid_writer.write(img)

    prev_ids = curr_ids
    prev_coords = curr_coords

    if i % 1000 == 0:
      print(i)
  vid_writer.release()

