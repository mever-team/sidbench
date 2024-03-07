import os
import time 


input_dir = '/nas3-2/dekonstantinidou/coco-2017/train/data/'
output_dir = '/fssd8/user-data/manosetro/sid_bench/train/corvi2022/coco/0_real'

i = 0
with open('/fssd8/user-data/manosetro/sid_bench/train/corvi2022/real_coco.txt', 'r') as file:
    for line in file:
        i += 1
        line = line.strip()

        img_pth = input_dir + line
        cpd_file =  output_dir + '/' + line

        if os.path.exists(cpd_file):
            continue
        
        # check if image path exists, then copy
        if os.path.exists(img_pth):
            os.system(f'cp {img_pth} {cpd_file}')
            time.sleep(0.05)
            if i % 200 == 0:
                print(f'Copied {i} images')
                time.sleep(0.1)
        else:
            print(f'Path does not exist: {img_pth}')

print(f'Copied {i} images')