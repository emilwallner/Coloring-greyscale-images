"""
parser.py: A basic parser for the YFCC100M dataset.

author: Frank Liu - frank.zijie@gmail.com
last modified: 05/30/2015

Copyright (c) 2015, Frank Liu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Frank Liu (fzliu) nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Frank Liu (fzliu) BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from io import BytesIO
import random
import os
import sys
import time


import re
from random import randint
import uuid
from multiprocessing import Pool


# library imports (install with pip)
import numpy as np
from PIL import Image
import requests

# directory which contains the tab-separated YFCC100M data
# more info @ download at http://labs.yahoo.com/news/yfcc100m/
YFCC100M_DIR = "yfcc100m_dataset"

# keys for the YFCC100M data
YFCC100M_KEYS = [
    "photo_id",
    "identifier",
    "hash",
    "user_id",
    "username",
    "date_taken",
    "upload_time",
    "camera_type",
    "title",
    "description",
    "user_tags",
    "machine_tags",
    "longitude",
    "latitude",
    "accuracy",
    "page_url",
    "download_url",
    "license",
    "license_url",
    "server",
    "farm",
    "secret",
    "original",
    "extension",
    "image_or_video"
]


def image_from_url(url):
    """
        Downloads an image in numpy array format, given a URL.
    """

    # loop until the image is successfully downloaded
    status = None
    while status != 200:
        response = requests.get(url)
        status = response.status_code
    pimg = Image.open(BytesIO(response.content))
    pimg = pimg.convert("RGB")
    
    pimg.save('/home/ubuntu/storage/yahoo/yfcc100m-tools/peoplenet/' + str(uuid.uuid4()) + '.jpg')

    return True

def download_images(line):
        try:

            # fit the data into a dictionary
            values = [item.strip() for item in line.split("\t")]
            data = dict(zip(YFCC100M_KEYS, values))
            
            people = False
            if bool(re.search("people", data["machine_tags"])) or bool(re.search("people", data["user_tags"])):
                people = True

            if data["image_or_video"] == "0" and people:
                image_from_url(data["download_url"])
        
        except IOError:
            print('Error!')
        
        
if __name__ == "__main__":
    YFCC100M_DIR = '/home/ubuntu/storage/yahoo/yfcc100m-tools/parts/'
    
    for part in os.listdir(YFCC100M_DIR):
        fh =  open(os.path.join(YFCC100M_DIR, part), "r").readlines()
        
        print(part)
        pool = Pool(processes=16) 
        pool.map(download_images, fh)

    print("Done!")