import os
from PIL import Image
from multiprocessing import Pool

size = 224, 224

def resize_image(path_and_file):
	with Image.open(path_and_file[0]) as im:
		im.thumbnail(size)
		im.save('/home/userai/jobs/drawing2logo/data/screenshots_224/' + path_and_file[1], "PNG")

if __name__ == "__main__":
	img_dir = r"./screenshots_382/"
	images = []
	
	for filename in os.listdir(img_dir):
		filepath = os.path.join(img_dir, filename)
		images.append([filepath, filename])
	
	pool = Pool(processes=125) 
	pool.map(resize_image, images)

	print("Done!") 