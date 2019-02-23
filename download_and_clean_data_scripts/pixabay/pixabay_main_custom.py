# -*- coding: utf-8 -*-
# @Time    : 2018/8/16 10:59
# @Author  : 陈子昂
import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import sys
from utils import save_img, path_processor, img_name_processor


def pexels(keyword):
    img_cnt = 0
    if not keyword: sys.exit('程序退出：未输入关键字！')
    for page in tqdm(range(1, 50)):
        print(f'\n-----[{keyword}]正在爬取第{page}页-----')
        pexels_url = "https://www.pexels.com/search/%s/?page=%s" % (keyword, page)

        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}
        res = requests.get(pexels_url,headers=headers,verify=False)

        # print(res.text)
        if 'Sorry, no pictures found!' in res.text:
            print('-*--*--*-爬取完毕-*--*--*-')
            sys.exit(0)

        soup = BeautifulSoup(res.text, 'lxml')
        # print(soup)
        articles = soup.find_all('article')
        # print(len(articles))
        for article in articles:
            src = article.img.attrs['src']
            print(src)
            path = rf'D://人脸相关的图片//pexels//{keyword}'
            if not os.path.exists(path):
                os.makedirs(path)
            filename = img_name_processor(src)
            file = os.path.join(path, filename)
            save_img(file=file, src=src)


if __name__ == "__main__":
    
    categories = ['male', 'old', 'vintage', 'dog', 'cat', 'building', 'nature', 'castle', 'water', 'ocean', 'cities', 'body', 'hands', 'people', 'culture', 'religion', 'color', 'patterns', 'houses', 'vintage', 'river', 'landscape', 'lights', 'animals', 'wallpaper', 'texture', 'current events', 'architecture', 'business', 'work', 'travel', 'fashion', 'food', 'drink', 'spirituality', 'experimental', 'health', 'arts', 'culture', 'children', 'people', 'events', 'trees', 'green', 'yellow', 'pink', 'blue', 'red', 'minimal', 'hands', 'head', 'eyes', 'mouth', 'eating', 'playing', 'sports']
    for i in categories:
        pexels(i)
