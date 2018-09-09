import os, sys
import numpy as np
import imageio
from time_left import pretty_time_left, pretty_print_time
import time

def crop_image( im, thresh=16 ):
    mask = (im[:,:,0]>thresh) * (im[:,:,1]>thresh) * (im[:,:,2]>thresh)
    lower_border = np.where(mask.sum(1)>0)[0][0]
    upper_border = np.where(mask.sum(1)>0)[0][-1]
    left_border = np.where(mask.sum(0)>0)[0][0]
    right_border = np.where(mask.sum(0)>0)[0][-1]
    im_cropped = im[lower_border:upper_border,left_border:right_border,:]
    mask_cropped = mask[lower_border:upper_border,left_border:right_border]
    return im_cropped, mask_cropped

def normalize_image( im, mask=True ):
    im = im.astype(np.float32)
    im[mask] -= im[mask].min()
    im[mask] /= im[mask].max()
    im[~mask] = im[mask].mean()
    return im

def process_folder( source, target ):
    file_list = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
    try:
        os.makedirs(target)
    except Exception:
        pass
    tic = time.time()
    for i, f in enumerate(file_list):
        im = imageio.imread(source+f)
        im, mask = crop_image( im, 16 )
        #im = normalize_image( im, mask )
        imageio.imwrite(target+f+'.cropped.png', np.array(im))
        print("{}/{} - {} left".format(i+1, len(file_list), pretty_time_left(tic, i+1, len(file_list) )))
        
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python', sys.argv[0], 'source_directory target_directory')
    else:
        process_folder( sys.argv[1], sys.argv[2] )