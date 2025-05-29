import os
import shutil

# Path to your test image folder
test_dir = r'C:\Users\kavya\Downloads\Sign-Language-Recognition-master\Sign-Language-Recognition-master\asl_alphabet_test\asl_alphabet_test'

# Target directory to hold sorted folders
sorted_dir = r'C:\Users\kavya\Downloads\Sign-Language-Recognition-master\Sign-Language-Recognition-master\asl_alphabet_test_sorted'

# Create target directory if it doesn't exist
if not os.path.exists(sorted_dir):
    os.makedirs(sorted_dir)

# Loop through all files in test_dir
for filename in os.listdir(test_dir):
    if filename.endswith('.jpg'):
        # Extract class name (e.g., 'A' from 'A_test.jpg')
        class_name = filename.split('_')[0]

        # Create class folder if not exists
        class_folder = os.path.join(sorted_dir, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # Move the file into the class folder
        src_path = os.path.join(test_dir, filename)
        dst_path = os.path.join(class_folder, filename)
        shutil.move(src_path, dst_path)
        print(f'Moved {filename} to {class_folder}')

print("✅ Sorting complete. Test images organized by class.")
