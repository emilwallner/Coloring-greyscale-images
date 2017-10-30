# Coloring B&W portraits with neural networks.

This is the code for my article ["Coloring B&W portraits with neural networks"](https://blog.floydhub.com/colorizing-b&w-photos-with-neural-networks/)

Earlier this year, Amir Avni used neural networks to [troll the subreddit](http://www.whatimade.today/our-frst-reddit-bot-coloring-b-2/) [/r/Colorization](https://www.reddit.com/r/Colorization/) - a community where people colorize historical black and white images manually using Photoshop. They were astonished with Amir’s deep learning bot - what could take up to a month of manual labour could now be done in just a few seconds.

I was fascinated by Amir’s neural network, so I reproduced it and documented the process. Read the article to understand the context of the code.  

![Fusion Layer](fusion_layer.png)





## **Deploying the code on FloydHub**

If you are new to FloydHub, do their [2-min installation](https://www.floydhub.com/), check my [5-min video tutorial](https://www.youtube.com/watch?v=byLQ9kgjTdQ&t=6s) or my [step-to-step guide](https://blog.floydhub.com/my-first-weekend-of-deep-learning/) - it’s the best (and easiest) way to train deep learning models on cloud GPUs.

Once FloydHub is installed, use the following commands: 


    git clone https://github.com/emilwallner/Coloring-greyscale-images-in-Keras

Open the folder and initiate FloydHub.


    cd Coloring-greyscale-images-in-Keras/floydhub
    floyd init colornet

The FloydHub web dashboard will open in your browser, and you will be prompted to create a new FloydHub project called `colornet`. Once that's done, go back to your terminal and run the same `init` command.


    floyd init colornet

Okay, let's run our job:


    floyd run --data emilwallner/datasets/colornet/2:data --mode jupyter --tensorboard

Some quick notes about our job: 


- We mounted a public dataset on FloydHub (which I've already uploaded) at the `data` directory with `--data` `emilwallner/datasets/colornet/2:data`. You can explore and use this dataset (and many other public datasets) by viewing it on [FloydHub](https://www.floydhub.com/emilwallner/datasets/cifar-10/1)
- We enabled Tensorboard with `--tensorboard`
- We ran the job in Jupyter Notebook mode with `--mode jupyter`
- If you have GPU credit, you can also add the GPU flag `--gpu` to your command - this will make it ~50x faster

Go to your the Jupyter notebook under the Jobs tab on the FloydHub website, click on the Jupyter Notebook link, and navigate to this file: `floydhub/Alpha version/alpha_version.ipynb`. Open it and click shift+enter on all the cells. It's the same process for the beta_version.ipynb and the full_version.ipynb.

Gradually increase the epoch value to get a feel for how the neural network learns.  
