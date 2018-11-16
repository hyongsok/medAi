import os, sys
import numpy as np
import imageio
from time_left import pretty_time_left
import time
import cv2
import warnings
import datetime

def crop_image( im, inner_box=False ):
    boxes = find_retina_boxes( im )
    if boxes is None:
        return None
    x, y, r_in, r_out = boxes
    img_y, img_x, _ = im.shape

    if inner_box:
        d = r_in
    else:
        d = r_out
    x_min = max(x - d, 0)
    x_max = min(x + d, img_x)
    y_min = max(y - d, 0)
    y_max = min(y + d, img_y)
    im_cropped = im[y_min:y_max+1,x_min:x_max+1,:]
    return im_cropped

def normalize_image( im, mask=True ):
    im = im.astype(np.float32)
    im[mask] -= im[mask].min()
    im[mask] /= im[mask].max()
    im[~mask] = im[mask].mean()
    return im

def find_retina_boxes( im, display = False, dp = 1.0, minDist = 500, param1=80, param2=30, minRadius=500, maxRadius=0 ):
    '''Finds the inner and outer box around the retina using openCV
    HoughCircles. Returns x,y coordinates of the box center, radius
    d of the inner box and radius r of the outer box as x, y, r_in, r_out. 
    If more than one circle is found returns the circle with the center
    closest to the image center. Returns None if no circle is found.
    If display = True then it returns the image with all circles drawn
    on it.
    All arguments after display are cv2.HoughCircles arguments
    '''
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    minRadius=int(gray.shape[0]/3)
    maxRadius=int(gray.shape[0]*1.5)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    # ensure at least some circles were found
    print(len(circles), 'circles found')
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        
        center_circle = 0
        if len(circles) > 1:
            cx, cy = np.array(im.shape[:2])/2
            dist = (circles[:,0]-cx)**2 + (circles[:,1]-cy)**2
            np.argmin(dist)
        
        if display:
            # loop over the (x, y) coordinates and radius of the circles
            output = im.copy()
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            x, y, r = circles[center_circle,:]
            cv2.circle(output, (x, y), r, (0, 255, 255), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (128, 128, 255), -1)
            return output
        
        x, y, r_out = circles[center_circle,:]
        r_in = int(np.sqrt((r_out**2)/2))
        return x, y, r_in, r_out
    else:
        warnings.warn('No circles found on image')
        if param1 > 40:
            print('Retry with param1=40.')
            return find_retina_boxes( im, display, dp=dp, minDist=minDist, param1=40, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        else:
            print('no luck, skipping image')

        return None


def process_folder( source, target, inner = False ):
    folder_list = [f for f in os.listdir(source) if os.path.isdir(os.path.join(source, f))]
    for f in folder_list:
        process_folder(os.path.join(source, f) , os.path.join(target, f), inner)

    file_list = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
    try:
        os.makedirs(target, exist_ok=True)
    except OSError:
        pass
    tic = time.time()
    for i, f in enumerate(file_list):
        print('Starting with', os.path.join(source, f))
        if os.path.exists(os.path.join(target, f)):
            print('skipping', os.path.join(target, f))
            continue
        try:
            im = imageio.imread(os.path.join(source, f))
        except ValueError as e:
            print('Error while reading file', os.path.join(source, f), '\n', e)
            continue
            
        im = crop_image( im, inner )
        #im = normalize_image( im, mask )
        if im is None:
            warn_message = 'Image {} could not be cropped.'.format(os.path.join(source, f))
            warnings.warn(warn_message)
            with open('warn.log', 'a') as fout:
                fout.write('{:%Y-%m-%d %H:%M}: {}\n'.format(datetime.datetime.now(), os.path.join(source, f)))
        try:
            imageio.imwrite(os.path.join(target, f), np.array(im))
        except ValueError as v:
            print('Could not write {}\nAdditional info:'.format(os.path.join(target, f)), v)
        print("{}: {}/{} - {} left".format(os.path.join(source, f), i+1, len(file_list), pretty_time_left(tic, i+1, len(file_list) )))
        
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python', sys.argv[0], 'source_directory target_directory [inner box (True/False)]')
    else:
        if len(sys.argv) > 3:
            inner_box_flag = bool(str(sys.argv[3]).lower() == 'true')
        else:
            inner_box_flag = False
        print('inner', inner_box_flag)
        process_folder( sys.argv[1], sys.argv[2], inner_box_flag )