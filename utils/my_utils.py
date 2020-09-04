import os
import cv2
import shutil
import time
from os.path import join
import random

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