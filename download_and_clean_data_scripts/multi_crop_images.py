import os
from PIL import Image
from multiprocessing import Pool


def crop_image(path_and_file):
	with Image.open(path_and_file[0]) as im:
		x, y = im.size
		im.crop((0, 18, x - 18, y)).save('/home/userai/jobs/drawing2logo/data/screenshots_382/' + path_and_file[1], "PNG")

if __name__ == "__main__":
	img_dir = r"./screenshots/"
	images = []
	
	for filename in os.listdir(img_dir):
		filepath = os.path.join(img_dir, filename)
		images.append([filepath, filename])
	
	pool = Pool(processes=120) 
	pool.map(crop_image, images)

	print("Done!") 