# neural-style-transfer

The style of an image is transfered to an another image by optimizing the generated image (initially a noise image or the content image, which in this case the content image) using gradient descent. To optimize the generated image with the gradient descent, the loss is calculated by calculating the unweighted correlations in the extracted features from the style and content image. A pretrained VGG-19 CNN is used for feature extraction(Note: The network is not optimized, instead the generated image is optimized with respect to the content and style loss).
