# Yahoo Flickr Creative Commons 100M

- Get access here: https://webscope.sandbox.yahoo.com/catalog.php?datatype=i&did=67
- Once you register you will get an instant email how to download the image urls (check your spam folder)
- Once you have the text file split it with 1M lines in each: ```split -l 1000000 mybigfile.txt```
- Edit the 'parse100m.py' file and choose which keywords to download and how many CPUs you have
- Run the script
- Known bug: I had a memory problem after downloading 150K images
- The downloading speed is high ~10-20M images in a day

# Pixabay

- ```pip install beautifulsoup4 tqdm```
- Edit the 'pixabay_main_custom.py' file to decide which key words to download
- The downloading speed is slow ~30K images in a day
- Either create a multiprocess version or run several scripts with different keywords using Tmux

