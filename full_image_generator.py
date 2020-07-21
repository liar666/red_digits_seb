#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 17:09:11 2020

@author: gmuller
"""

from PIL import Image, ImageDraw
import random as rd

rd.seed(42)


DIGIT_IMG_DIR="/home/gmuller/Python/RedDigitsSeb/red_digits_seb/Data/Images/numbers_cleaned/"
OUT_IMG_DIR="./test_images/"
CLASSES=[0,1,2,3,4,5,6,7,8,9,"E","H"]
DEBUG=False

split_dirs = ['v', 'h']


def draw_digit(image, top_left, bottom_right):
    number_val = rd.choices(CLASSES, k=1)[0]
    number_img = Image.open(DIGIT_IMG_DIR+str(number_val)+".png")
    if DEBUG:
        print(">>> draw_digit::top_left="+str(top_left)+" / bottom_right="+str(bottom_right))
        print(">>> draw_digit::number="+str(number_val))
    
    scale_x = (bottom_right[0] - top_left[0]) / number_img.width
    scale_y = (bottom_right[1] - top_left[1]) / number_img.height
    scale = min(scale_x, scale_y)
    scale = (rd.random()*.5+.5)*scale  #TODO: not 0!
    if DEBUG:
        print(">>> draw_digit::x.scale="+str(scale_x))
        print(">>> draw_digit::y.scale="+str(scale_y))
        print(">>> draw_digit::min.scale="+str(scale))
        print(">>> draw_digit::rd.scale="+str(scale))
    if scale < .01:
        print("===== > scale too small!")
    else:
        resize_x = int(number_img.width*scale)
        resize_y = int(number_img.height*scale)
        number_img = number_img.resize((resize_x, resize_y))
        pos_x = rd.randint(top_left[0], bottom_right[0]-number_img.width)
        pos_y = rd.randint(top_left[1], bottom_right[1]-number_img.height)
        image.paste(number_img, (pos_x,pos_y), number_img)
        if DEBUG:
            print("resize_x="+str(resize_x)+"  /  resize_y="+str(resize_y))
            print(">>>> draw_digit::top_left[0]="+str(top_left[0])+"  /  bottom_right[0]="+str(bottom_right[0])+"  /  pos_x="+str(pos_x))
            print(">>>> draw_digit::top_left[1]="+str(top_left[1])+"  /  bottom_right[1]="+str(bottom_right[1])+"  /  pos_y="+str(pos_y))


def generate_full_image(image, top_left, bottom_right, nb_digits, split_dir):
    if DEBUG:
        print("gene_full_img::split_dir="+split_dir)
        print("gene_full_img::top_left="+str(top_left)+"  /  bottom_right="+str(bottom_right))
        draw = ImageDraw.Draw(image)
    if nb_digits < 2 :
        if DEBUG:
            print("************ Will stop at this iteration! ************")
        draw_digit(image, top_left, bottom_right)
    else:
        x_top_left  = top_left[0]
        y_top_left  = top_left[1]
        x_bot_right = bottom_right[0]
        y_bot_right = bottom_right[1]
        next_split_dir = split_dirs[rd.randint(0, 1)]
        if split_dir == 'v':
            new_x_digit = int(x_top_left+(x_bot_right-x_top_left)/2)
            new_y_digit = int(y_bot_right)
            new_x_nondigit = new_x_digit
            new_y_nondigit = y_top_left
        else: #elif split_dir == 'h':
            new_x_digit = int(x_bot_right)
            new_y_digit = int(y_top_left+(y_bot_right-y_top_left)/2)
            new_x_nondigit = x_top_left
            new_y_nondigit = new_y_digit
        if DEBUG:
            draw.rectangle((top_left[0], top_left[1], new_x_digit, new_y_digit), fill=None, outline=(0, 0, 255), width=2)
            draw.rectangle((new_x_nondigit, new_y_nondigit, bottom_right[0], bottom_right[1]), fill=None, outline=(0, 255, 0), width=2)
        draw_digit(image, top_left, (new_x_digit, new_y_digit))
        if DEBUG:
            print(">> Continuing ("+next_split_dir+")")
        generate_full_image(image, (new_x_nondigit,new_y_nondigit), bottom_right, nb_digits-1, next_split_dir)



if __name__ == "__main__":
    nb_images  = 50
    image_sizes = list(range(200,2000,100))
    max_digits = 5
    bg_color   = (255, 255, 255, 255)
    for nbi in range(nb_images):
        img_size = rd.choices(image_sizes, k=1)[0]
        background = Image.new('RGBA', (img_size,img_size), bg_color)
        nb_digits = rd.randint(1, max_digits+1)
        print("\n=========== Generating image "+str(nbi)+" ==============")
        print("nb_digits="+str(nb_digits))
        split_dir = split_dirs[rd.randint(0, 1)]
        generate_full_image(background, (0,0), (img_size, img_size), nb_digits, split_dir)
        background.save(OUT_IMG_DIR+"/img_"+str(nbi)+".png", "PNG")

