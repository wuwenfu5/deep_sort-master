# -*- coding:utf-8 -*-
"""
@author:Mr.Five
@email:wuwenfu5@qq.com
@num&WeChat:+86-15241192213
@file:generate_img.py
@time:18-5-7下午4:12
"""

import cv2
import time

t1 = time.perf_counter()

cap = cv2.VideoCapture(r'/media/wuwenfu5/Win_Ubuntu_Swap/Python_/Material/MyMOT/shitang5.AVI')

if cap.isOpened() is False:
    print('capture is not opened')
else:
    print('capture is opened')

frame_count = 0
while cap.isOpened():

    ret, frame = cap.read()

    if ret == True:
        keycode = cv2.waitKey(1) & 0xff
        if keycode != 0xff:
            print(keycode, '\n', chr(keycode))
            break
        frame_count += 1
        cv2.imwrite(str('./logs/%06d.jpg' % frame_count), frame)
        print('write %06d.jpg' % frame_count)
    else:
        break

cap.release()
print('\n %f s' % (time.perf_counter() - t1))
