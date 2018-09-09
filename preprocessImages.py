import os, sys
import numpy as np
import imageio


def crop_image( im, thresh=6 ):
    mask = (im[:,:,0]>thresh) * (im[:,:,1]>thresh) * (im[:,:,2]>thresh)
    lower_border = np.where(mask.sum(1)>0)[0][0]
    upper_border = np.where(mask.sum(1)>0)[0][-1]
    left_border = np.where(mask.sum(0)>0)[0][0]
    right_border = np.where(mask.sum(0)>0)[0][-1]
    im_cropped = im[lower_border:upper_border,left_border:right_border,:].astype(np.float64)
    mask_cropped = mask[lower_border:upper_border,left_border:right_border]
    return im_cropped, mask_cropped

def normalize_image( im, mask=True ):
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
    for f in file_list:
        im = imageio.imread(source+f)
        im, mask = crop_image( im, 6 )
        im = normalize_image( im, mask )
        imageio.imwrite(target+f+'.cropped.png', np.array(im))
        
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python', sys.argv[0], 'source_directory target_directory')
    else:
        process_folder( sys.argv[1], sys.argv[2] )