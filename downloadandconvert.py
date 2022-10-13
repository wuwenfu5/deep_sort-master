# -*- coding:utf-8 -*-
"""
@author:Mr.Five
@email:wuwenfu5@qq.com
@num&WeChat:+86-15241192213
@file:downloadandconvert.py
@time:18-5-9下午9:24
https://github.com/hamidb/pedestrian_detection.git
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import urllib
from scipy.io import loadmat
import itertools
from io import StringIO

import tarfile, zipfile
import glob
import sys
import os.path
import fnmatch
import shutil
import cv2 as cv
import numpy as np
import struct

data_dir = os.getcwd() + '/data'
save_dir = os.getcwd() + '/output'
web_url = 'http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA'
resize = True
display = False
# minimum allowable width and height to collect samples
MIN_WIDTH = 15
MIN_HEIGHT = 15
# sample width and height for training
P_WIDTH = 32
P_HEIGHT = 64
DEPTH = 3

"""
To read .seq file, some data formats are inspired from:
http://blog.csdn.net/a2008301610258/article/details/45873867#
Also some .vbb file formats are inspired from:
https://github.com/mitmul/caltech-pedestrian-dataset-converter/blob/master/scripts/convert_annotations.py
"""


def open_seq_file(file):
    global img_width
    global img_height
    # read .seq file, and save the images into the savepath
    with open(file, 'rb') as f:
        # get rid of seq header containing some info about the seq
        header = str(f.read(548))
        img_width = struct.unpack('@i', f.read(4))[0]
        img_height = struct.unpack('@i', f.read(4))[0]
        # get rid of the rest
        header = str(f.read(468))
        string = str(f.read())
        # each image's header
        img_header = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"
        # split .seq file into segment with the image header
        strlist = string.split(img_header)
    count = 0
    for img in strlist:
        # ignore the header
        if count > 0:
            # add image header to img data
            img = os.path.join(img_header[0:9], img)
            image = cv.imdecode(np.frombuffer(img, np.uint8), 1)
            yield image
        count += 1


def load_annotation(setname, seqname):
    filepath = 'data/annotations/' + setname + '/' + seqname.split('.')[0] + '.vbb'
    if not os.path.exists(filepath):
        print('Warning: annotation file for %s/%s is missing' % (setname, seqname))
        return [], [], -1
    vbbfile = loadmat(filepath)
    vbbdata = vbbfile['A'][0][0]
    objList = vbbdata[1][0]
    objLbl = vbbdata[4][0]
    return objList, objLbl, 0


def download_dataset():
    # Download the tarball from Caltech website
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_url = [web_url + '/set%02d.tar' % f for f in range(10 + 1)]
    data_url += [web_url + '/annotations.zip']
    for url in sorted(data_url):
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                                 float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    # Extracting tarballsif tarballs are not extracted
    print('Uncompressing...')
    for fn in sorted(data_url):
        filename = fn.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        # if tarballs are not extracted
        if not os.path.exists(filepath.split('.')[0]):
            if tarfile.is_tarfile(filepath):
                with tarfile.open(filepath) as tf:
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner=numeric_owner) 
                        
                    
                    safe_extract(tf, data_dir)
            if zipfile.is_zipfile(filepath):
                with zipfile.ZipFile(open(filepath)) as zf:
                    zf.extractall(data_dir)


def draw_body(img_draw, pos, color, thickness):
    pos = pos.astype(np.int32)
    cv.rectangle(img_draw, tuple(pos[0:2]),
                 tuple(x + y for x, y in izip(pos[0:2], pos[2:4])),
                 color, thickness)


def draw_label(img_draw, lbl, pos, color, thickness):
    pos = pos.astype(np.int32)
    cv.putText(img_draw, lbl, (pos[0], pos[1] - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv.LINE_AA)


def draw_objects(img_draw, pos, posv, occl):
    # draw the whole body
    draw_body(img_draw, pos, (0, 0, 255), 1)
    # draw visible parts if any
    draw_body(img_draw, posv, (0, 255, 0), 1) if occl else None
    # draw label
    draw_label(img_draw, lbl, pos, (0, 0, 255), 1)


def save_records(record_str, filename):
    # delete .bin file if exists
    os.remove(filename) if os.path.exists(filename) else None
    with open(filename, 'wb') as f:
        f.write(record_str)
    # image bytes + 1 byte for label
    record_bytes = DEPTH * P_WIDTH * P_HEIGHT + 1
    object_n = len(record_str) / record_bytes
    assert np.floor(object_n) - object_n == 0, 'number of objects is not an integer: %r' % object_n
    if object_n == 0:
        print('No object found for "%s"' % os.path.basename(filename).split('.')[0])
    else:
        print('Saved %d object (%d/%d bytes) to %s'
              % (object_n, len(record_str),
                 os.stat(filename).st_size,
                 os.path.basename(filename)))


if __name__ == "__main__":
    # download caltech dataset if it is not downloaded
    download_dataset()
    total_patches = total_frames = frame_n = patch_n = 0
    for setpath in sorted(glob.glob(data_dir + "/set*")):
        setname = os.path.basename(setpath)
        record_str = StringIO()
        for parent, dirnames, filenames in os.walk(setpath):
            for filename in sorted(filenames):
                # check .seq file with suffix
                if fnmatch.fnmatch(filename, '*.seq'):
                    # get path of each .seq file
                    filepath = os.path.join(parent, filename)
                    # create saving directory for each seq file
                    save_seq_dir = ''.join([save_dir, '/', setname, '/', filename.split('.')[0]])
                    if not os.path.exists(save_seq_dir):
                        os.makedirs(save_seq_dir)
                    objList, objLbl, err = load_annotation(setname, filename)
                    sys.stdout.write('\rProcessing "%s/%s" ...' % (setname, filename))
                    sys.stdout.flush()
                    # check if the annotation file exists
                    if err:
                        print('Obj is empty with code %d' % err)
                        continue
                    total_patches += patch_n
                    total_frames += frame_n
                    frame_n = patch_n = 0
                    for img in open_seq_file(filepath):
                        if display:
                            img_draw = img.copy()
                        # check if the frame contains object
                        if len(objList[frame_n]):
                            # retrieve annotated objects
                            objects = objList[frame_n][0]
                            for id, pos, posv, occl in zip(objects['id'], objects['pos'],
                                                           objects['posv'], objects['occl']):
                                id = int(id[0] - 1)
                                lbl = str(objLbl[id][0])
                                if display:
                                    draw_objects(img_draw, pos[0], posv[0], occl)
                                pos = posv[0].astype(np.int32) if occl else pos[0].astype(np.int32)
                                # filter out too small patches
                                if lbl == 'person' and pos[2] >= MIN_WIDTH and pos[3] >= MIN_HEIGHT:
                                    # boundary gaurd
                                    pos[0] = 0 if pos[0] < 0 else pos[0]
                                    pos[1] = 0 if pos[1] < 0 else pos[1]
                                    pos[2] = img_width - pos[0] if pos[0] + pos[2] > img_width else pos[2]
                                    pos[3] = img_height - pos[1] if pos[1] + pos[3] > img_height else pos[3]
                                    # crop person from the image
                                    patch = img[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]]
                                    if resize:
                                        patch = cv.resize(patch, (P_WIDTH, P_HEIGHT), cv.INTER_CUBIC)
                                    save_name = os.path.join(save_seq_dir, "%04d.jpg" % patch_n)
                                    # seperate record string with '1' as a label for 'person'
                                    record_str.write('1')
                                    record_str.write(patch.tostring())
                                    cv.imwrite(save_name, patch)
                                    patch_n += 1
                        if display:
                            cv.imshow('Caltech-Ped dataset', img_draw)
                            key = cv.waitKey(1)
                            # press q to exit
                            if key == ord('q'):
                                cv.destroyAllWindows()
                                exit()
                        frame_n += 1
            save_records(record_str.getvalue(), ''.join([save_dir, '/', setname, '.bin']))
    print('Total %d positive samples are extracted '
          'from %d frames.' % (total_patches, total_frames))
