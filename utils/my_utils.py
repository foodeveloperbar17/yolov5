import os
import cv2
import shutil
import time
from os.path import join
from os import listdir
import random
import numpy as np

def get_images_from_video(video_path, dir_to_result_images,
                          class_prefix, every_n_image=1, file_extention='jpg',
                          pad_zeros=True, num_zeros=6):
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

def copy_labels_to_dir(src, dest):
  text_files = get_text_files_from_dir(src)
  print(len(os.listdir(dest)))
  for text_file in text_files:
    shutil.copy2(join(src, text_file), join(dest, text_file))
  print(len(os.listdir(dest)))

def get_text_files_from_dir(dir_path):
  return [f for f in os.listdir(dir_path) if f.endswith('.txt')]

def force_create_folder(dir_path):
  if os.path.exists(dir_path):
    shutil.rmtree(dir_path)
  os.mkdir(dir_path)

def train_test_split(image_src, label_src, train_dir, valid_dir, test_dir, valid_ratio = 0.2, test_ratio=0, image_extention='jpg'):
  absent = 0
  force_create_folder(train_dir)
  force_create_folder(valid_dir)
  force_create_folder(test_dir)
  
  force_create_folder(join(train_dir, 'labels'))
  force_create_folder(join(train_dir, 'images'))
  force_create_folder(join(valid_dir, 'labels'))
  force_create_folder(join(valid_dir, 'images'))
  force_create_folder(join(test_dir, 'labels'))
  force_create_folder(join(test_dir, 'images'))
  
  text_files = get_text_files_from_dir(label_src)
  for text_file in text_files:
    image_file = text_file[:-4] + '.' + image_extention
    curr_image_src = join(image_src, image_file)
    if not os.path.exists(curr_image_src):
      absent += 1
      continue

    text_file_dest = None
    image_file_dest = None

    r = random.uniform(0, 1)
    if r < test_ratio:
      text_file_dest = join(test_dir, 'labels', text_file)
      image_file_dest = join(test_dir, 'images', image_file)
    elif r < test_ratio + valid_ratio:
      text_file_dest = join(valid_dir, 'labels', text_file)
      image_file_dest = join(valid_dir, 'images', image_file)
    else :
      text_file_dest = join(train_dir, 'labels', text_file)
      image_file_dest = join(train_dir, 'images', image_file)

    shutil.copy2(join(label_src, text_file), text_file_dest)
    shutil.copy2(curr_image_src, image_file_dest)
  print('images absent ' + str(absent))
    
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
        curr_coords.append([float(num) for num in line.split()[1:]])
      index = get_index_from_file_name(textfile)
      coords_with_textfiles.append((index, curr_coords))
  return coords_with_textfiles

def get_coordinates_from_textfiles(textfiles_list):
  coords = []
  for textfile in textfiles_list:
    with open(textfile, 'r') as reader:
      curr_coords = []
      for line in reader.readlines():
        curr_coords.append([float(num) for num in line.split()[1:]])
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

def give_ids(prev_coords, prev_ids, curr_coords, vy=0.02, limit_dist=0.1, last_id=0):
#  TODO might be extra to convert in numpy array curr coords
  pred_curr_coords = np.array(prev_coords)
  curr_coords = np.array(curr_coords)
  num_prev_nuts = pred_curr_coords.shape[0]
  num_curr_nuts = curr_coords.shape[0]
  curr_ids = {}

  if num_prev_nuts is not 0:
    pred_curr_coords[:, 1] += vy

    #prev nuts row, curr nuts column
    dists = calculate_euclidean_distances(pred_curr_coords[:, :2], curr_coords[:, :2]) 
    # mit be done parrallel, uuid might make it even more parrallel
    for _ in range(min(num_prev_nuts, num_curr_nuts)):
      curr_mins = np.min(dists, axis=1)
      prev_nut_index = np.argmin(curr_mins) # prev nut index
      curr_nut_index = np.argmin(dists[prev_nut_index])

      dist = dists[prev_nut_index, curr_nut_index]
      if dist > limit_dist:
        print('more than limit', dist)
        break

      dists[prev_nut_index, :] = 100
      dists[:, curr_nut_index] = 100

      prev_id = prev_ids[prev_nut_index]
      if prev_id == -1:
        last_id += 1
        curr_ids[curr_nut_index] = last_id
      else:
        curr_ids[curr_nut_index] = prev_id

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

def record_tracking(image_path_list, coords, video_path, fps, vy = 0.02, dist_limit=0.1):
  if os.path.isfile(video_path):
    os.remove(video_path)
  prev_coords = []
  prev_ids = {}
  last_id = 0
  fourcc = 'mp4v'
  vid_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*fourcc), fps, (640, 480))
  for i in range(len(image_path_list)):
    curr_coords = coords[i]
    curr_ids, last_id = give_ids(prev_coords, prev_ids, curr_coords, vy, dist_limit, last_id)

    img = get_labeled_image(curr_coords, curr_ids, image_path_list[i])
    vid_writer.write(img)

    prev_ids = curr_ids
    prev_coords = curr_coords

    if i % 1000 == 0:
      print(i)
  vid_writer.release()

def get_assigned_distances(coords, vy=0.02):
  prev_coords = np.array(coords[0])
  distances = []
  for i in range(1, len(coords)):
    curr_coords = coords[i]
    curr_coords = np.array(curr_coords)
    prev_coords[:, 1] += vy
    dists = calculate_euclidean_distances(prev_coords[:, :2], curr_coords[:, :2])
    
    curr_assigned_dists = []
    for i in range(min(prev_coords.shape[0], curr_coords.shape[0])):
      curr_mins = np.min(dists, axis=1)
      prev_nut_index = np.argmin(curr_mins)
      curr_nut_index = np.argmin(dists[prev_nut_index])

      curr_assigned_dists.append(dists[prev_nut_index, curr_nut_index])

      dists[prev_nut_index, :] = 100
      dists[:, curr_nut_index] = 100
    distances.append(curr_assigned_dists)

    prev_coords = curr_coords
  return distances

def video_list_to_images(video_path_class_mappings, images_base_dir, every_n_image=1):
  force_create_folder(images_base_dir)
  for video_path, class_prefix in video_path_class_mappings:
    images_folder = join(images_base_dir, class_prefix)
    os.mkdir(images_folder)
    get_images_from_video(video_path, images_folder, class_prefix, every_n_image, 'jpg', True, 6)

def filter_text_labels(src, dest, width, height):
    removed = 0
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.mkdir(dest)
    textfiles = [f for f in os.listdir(src) if f.endswith('.txt')]
    for f in textfiles:
        old_file_path = join(src, f)
        new_file_path = join(dest, f)
        with open(old_file_path, 'r') as reader:
            new_lines = []
            for line in reader.readlines():
                splitted = line.split()
                curr_width = float(splitted[3])
                curr_height = float(splitted[4])
                if curr_width > width and curr_height > height:
                    new_lines.append(line)
                else:
                    removed += 1
            if len(new_lines) != 0:
                with open(new_file_path, 'w') as writer:
                    for line in new_lines:
                        writer.write(line + '\n')
    print(removed)

def see_classes_distribution(labels_folder):
  distributions = [0, 0, 0, 0]
  for f in os.listdir(labels_folder):
    if f.endswith('.txt'):
      with open(join(labels_folder, f), 'r') as reader:
        for line in reader.readlines():
          if line.strip() == '':
            continue
          cls = int(line.split()[0])
          distributions[cls] += 1
  return distributions

def copy_image_with_label(image_src, label_src, dest, label_name, image_extention='jpg'):
  image_name = label_name[:-3] + image_extention
  image_src_path = join(image_src, image_name)
  label_src_path = join(label_src, label_name)
  if os.path.exists(image_src_path):
    image_dest_path = join(dest, image_name)
    label_dest_path = join(dest, label_name)
    shutil.copy2(image_src_path, image_dest_path)
    shutil.copy2(label_src_path, label_dest_path)
  else:
    print('cant find image in path {}'.format(image_src_path))

def distribute_evenly(src, dest, class_prefixes, num_instance):
  for class_prefix in class_prefixes:
    class_count = 0
    for class_text_file in [f for f in os.listdir(src) if f.endswith('.txt') and f[0] == class_prefix]:
      with open(join(src, class_text_file), 'r') as reader:
        class_count += len(reader.readlines())
      copy_image_with_label(src, src, dest, class_text_file)
      if class_count >= num_instance:
        break



def get_video_from_images(images_dir, video_path, fps):
  if os.path.exists(video_path):
    os.remove(video_path)
  fourcc = 'mp4v'
  video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*fourcc), fps, (640, 480))
  for img in [join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('jpg')]:
    video_writer.write(cv2.imread(img))
  video_writer.release()