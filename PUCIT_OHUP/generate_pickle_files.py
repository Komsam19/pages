### To generate pickle files of PUCIT_OHUL dataset###

import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
import cv2
import sys
import argparse
import os

##Only include while running on colab
running_on_colab = False
if running_on_colab == True:
  from google.colab import drive
  drive.mount("/gdrive", force_remount=True)
  dataset_folder = '/gdrive/My Drive/CALTex/dataset/PUCIT_OHUL/'
  data_folder= '/gdrive/My Drive/CALTex/data/PUCIT_OHUL/'
else:
  dataset_folder = '../../pages/PUCIT_OHUP/'
  data_folder= '../../pages/data/PUCIT_OHUP/'  

#This function makes a dictionary/vocabulary of all the unique characters in the labels file along with uniquely assigned numeric values to each different character.      
def create_vocabulary(labelfile):
    df = pd.read_excel(labelfile)
    lexicon = {}
    key = 1
    for i in df.index:  # Iteration over all labels/captions of training images.
        caption = df['Revised'][i]  # Label/caption of i-th image.
        slen = len(caption)
        j = 0
        while j < slen:
            ss = caption[j]  # Iteration over characters in the label/caption of i-th image. 
            if ss not in lexicon:
                lexicon[ss] = int(key)  # Set of unique characters with corresponding assigned labels. 
                
                # Print the Unicode (ASCII or UTF-8) value of the character.
                print(f"Character: {ss}, Unicode: {ord(ss)}")
                
                key = key + 1
            j = j + 1
        i = i + 1
    return lexicon


def save_vocabulary(worddicts):
    ## Save in txt format (letters)
    fp = open(dataset_folder + 'vocabulary.txt', 'w', encoding='utf-8')
    worddicts_r = [None] * (len(worddicts) + 1)
    
    ## Save in txt format (Unicode)
    fp_unicode = open(dataset_folder + 'vocabulary_unicode.txt', 'w', encoding='utf-8')

    ## Dictionary to store Unicode values
    unicode_dict = {}

    i = 1
    for kk, vv in worddicts.items():
        if i < len(worddicts) + 1:
            worddicts_r[vv] = kk
            fp.write(kk + '\t' + str(vv) + '\n')

            # Save Unicode values instead of characters
            unicode_val = ord(kk)
            fp_unicode.write(str(unicode_val) + '\t' + str(vv) + '\n')

            # Store in Unicode dictionary
            unicode_dict[unicode_val] = vv

        else:
            break
        i = i + 1

    fp.close()
    fp_unicode.close()

    ## Save Unicode dictionary in pickle format
    outFilePtr_unicode = open(data_folder + 'vocabulary_unicode.pkl', 'wb')
    pkl.dump(unicode_dict, outFilePtr_unicode)
    outFilePtr_unicode.close()



def partition(images, labels, valid_ind):
  train_labels=[]
  train_images={}
  valid_labels=[]
  valid_images={}
  data_part=len(images)-valid_ind
  for i in range(len(images)):
    if i<data_part:
      train_images[i]=images[i]
      train_labels.append(labels[i])
    else:
      valid_images[i-data_part]=images[i]
      valid_labels.append(labels[i])
  return train_images, train_labels, valid_images, valid_labels


#This function loads all the images from the imgfolder and corresponding labels of each image from labelfile.
#According to the dictionary, labels are converted into numeric sequence. 
def load_data(imgfolder, labelfile, dictionary, output_excel):
    labels = []
    images = {}
    count = 0
    ground_truth_dict = {}

    # Read the label file
    df = pd.read_excel(labelfile)

    # Extract the unique 'Person-Page' identifiers
    unique_identifiers = df['Person-Page-Line'].apply(lambda x: '-'.join(x.split('-')[:2])).unique()
    
    for identifier in unique_identifiers:
        # Construct the image file name (assuming all are PNG now)
        image_file = os.path.join(imgfolder, f"{identifier}.png")
        
        if os.path.exists(image_file):
            try:
                # Load the PNG image using OpenCV
                img = cv2.imread(image_file, -1)
                if img is None:
                    print(image_file + ' not available')
                    continue
                else:
                    print(f"Successfully loaded image: {image_file}")
                    print(f"before dimensions: {img.shape}")
                    # Resize the image (reduce both width and height by a factor of 5)
                    new_width = max(1, img.shape[1] // 5)
                    new_height = max(1, img.shape[0] // 5)
                    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    print(f"after dimensions: {img.shape}")

                    # Convert to grayscale if necessary
                    if len(img.shape) > 2:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Extract ground truth lines for the current identifier
                    ground_truth_lines = df[df['Person-Page-Line'].str.startswith(identifier)]['Revised']
                    #print(ground_truth_lines)
                    ground_truth_lines = ground_truth_lines.replace('\n', '', regex=True)
                    
                    # Reverse each line individually and then join
                    reversed_lines = [line[::-1] for line in ground_truth_lines] 
                    #print(reversed_lines)
                    full_ground_truth = '\n'.join(reversed_lines)

                    # Print image information and display using Matplotlib
                    print("----------------------------------------------")
                    print(image_file)
                    print("Ground-truth string: " + full_ground_truth)
                    
                    
                    # Add the image to the dictionary
                    images[count] = img
                    count += 1
                    # Keep newline (19) in numeric representation and handle reversing correctly
                    # Convert caption to numeric representation
                    w_list = []
                    for line in reversed_lines:
                        line_numeric = [dictionary[char] for char in line if char in dictionary]
                        w_list.extend(line_numeric[::-1])  # Reverse only each line separately
                        w_list.append(19)  # Add newline (19) at the end of each line
                    
                    labels.append(w_list)
                    print("Ground-truth in numeric representation: " + str(w_list).strip('[]'))
                    print("----------------------------------------------")
                    '''
                    plt.imshow(img, cmap="gray")
                    plt.title(f"Image ID: {count}\n{full_ground_truth}", color='b')
                    plt.axis('off')
                    plt.show()
                    '''

            except UnidentifiedImageError:
                print(f"Could not open file {image_file}. It may be corrupted or unsupported.")
                continue

    # Prepare the data for saving to an Excel file
    ground_truth_data = {
        'Person-Page-Line': list(ground_truth_dict.keys()),
        'Ground-Truth': list(ground_truth_dict.values())
    }

    # Create a DataFrame
    ground_truth_df = pd.DataFrame(ground_truth_data)

    # Save to an Excel file
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        ground_truth_df.to_excel(writer, sheet_name='Sheet1', index=False)
    print(f"Ground truth saved to {output_excel}")
    print(f"Number of images loaded: {len(images)}")  # Debug statement

    return images, labels



def main(args):
  train_images_path=dataset_folder + 'train_pages/'
  train_labels_path=dataset_folder + 'train_labels_v3.xlsx'
  test_images_path=dataset_folder +  'test_pages/'
  test_labels_path=dataset_folder + 'test_labels_v3.xlsx'
  output_labels_path=dataset_folder + "train_labels_v4.xlsx"
  output_test_labels_path=dataset_folder + "test_labels_v4.xlsx"

  #Load dictionary and data.
  #CAUTION: Dictionary/Vocabulary is always made from train_labels.xlsx.
  #Do not change this even when generating a pickle file for testing data.
  worddicts= create_vocabulary(train_labels_path)
  save_vocabulary(worddicts)
  #exit()
  
  images,labels = load_data(train_images_path,train_labels_path,worddicts, output_labels_path)
  test_images,test_labels = load_data(test_images_path,test_labels_path,worddicts, output_test_labels_path)

  if(int(args.valid_ind) > 0):
    train_images, train_labels, valid_images, valid_labels=partition(images, labels, int(args.valid_ind))
  else:
    train_images, train_labels, valid_images, valid_labels=partition(images, labels, (len(images)*15)/100)


  outFilePtr1 = open(data_folder + 'train_pages.pkl','wb')
  outFilePtr2 = open(data_folder + 'train_labels.pkl','wb')
  outFilePtr3 = open(data_folder + 'valid_pages.pkl','wb')
  outFilePtr4 = open(data_folder + 'valid_labels.pkl','wb')
  outFilePtr5 = open(data_folder + 'test_pages.pkl','wb')
  outFilePtr6 = open(data_folder + 'test_labels.pkl','wb')
  
  pkl.dump(train_images,outFilePtr1)
  pkl.dump(train_labels,outFilePtr2)
  pkl.dump(valid_images,outFilePtr3)
  pkl.dump(valid_labels,outFilePtr4)
  pkl.dump(test_images,outFilePtr5)
  pkl.dump(test_labels,outFilePtr6)

  outFilePtr1.close()
  outFilePtr2.close()
  outFilePtr3.close()
  outFilePtr4.close()
  outFilePtr5.close()
  outFilePtr6.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--valid_ind", type=int, default=0)
	(args, unknown) = parser.parse_known_args()
	main(args)    
