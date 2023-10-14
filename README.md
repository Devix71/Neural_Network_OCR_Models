# License Plate OCR Experiment

The purpose of this experiment was to explore and evaluate various approaches and model architectures towards license plate number recognition. 

At the first, a general text OCR pyTorch model is implemented and analyzed, namely Deepayan Das's "Adapting OCR with limited supervision" framework [Adapting-OCR](https://github.com/Deepayan137/Adapting-OCR). This way better insights into the principles behind unsupervised and semi-supervised OCR can be gathered. This model exhibits great performance in regards to its OCR capabilities, recognizing 97.97% of the characters and 86% of the sequence of characters used in its testing.
>**Note<sub>1</sub>:** the model used a synthetic dataset, where the text images were clearly visible and had uniform sizes, nowhere near the conditions of a real-life dataset, thus its performance is not accurate of a real world implementation.


The second step was to integrate a real-world dataset into the model. The chosen dataset was Francesco Pettini's License Plate Characters dataset, available on [Kaggle](https://www.kaggle.com/datasets/francescopettini/license-plate-characters-detection-ocr). This way the model was exposed to images of license plates from various countries, not just general text.

As expected, the model did not perform better than when it used the synthetic data, but what was unexpected was the degree of how much worse it performed. It exhibited a performance drop of almost 85% in character recognition, the model predicting only around 15.5% of the characters in the testing set, and not just that, the model was unable to predict any full sequence of characters corresponding to a license plate number. These results are supported by the low Precision, Recall and F1 scores of `0.18`, `0.10` and `0.12` respectively, which without a doubt indicate a very poor performing model.

The reasons for such a poor performance may be due to the low size of the dataset (only 209 images in total compared to the several thousand in the synthetic dataset) but also due to the inherent more complex nature of the real-world images.
>**Note<sub>2</sub>:** an adaptive pooling layer was used in the model's architecture to make sure that the model's H dimension would always be 1, the impact on the architecture's performance is minimal, it is just one way to ensure compatibility between the input data and the model.


Next, measures were taken to improve the model, this was done through the use of dataset augmentation and a change in the model's architecture.
The dataset was augmented utilizing the information from the corresponding `.XML`, which provided bounding boxes coordinates for the individual characters in each license plate image. Using this information, I managed to tweak parameters of those characters such as rotation, contrast and color, in order to create a new image based on the the original image, this way effectively doubling the available dataset.

Besides this, the model's CRNN architecture was tweaked as well. As mentioned previously, the new images were more complex and had a greater size as opposed to the synthetic dataset. This meant that the model should be scaled up in complexity (deepened) as well. I have achieved this by adding 2 extra convolutional layers, with the express aim of collecting more information from the images (which are also a higher resolution as compared to the original synthetic dataset 128px vs 64px height resolution). This changed was complemented by the addition of a Residual Block layer, which has the purpose of skipping layers and providing shortcut connections, thus reducing the increased vanishing gradient due to adding 2 extra layers. Finally, a low-value dropout was added to the model to prevent overfitting the model on a particular sequence of characters.

The effect of these changes was an increased model performance overall, with the mention that it still did not achieve any degree of an acceptable performance. The new model character accuracy score was of `20.45`, showing a 5 percentage point increase over the previous architecture. Not only that but the other performance metrics Precision, Recall, and F1 score have also improved visibly (albeit slightly) with values of `0.21`, `0.13` and `0.14` respectively, showing that the new model is better suited to this type of dataset.


Lastly, I have adapted the improved model to the template of a larger dataset, namely [LicensePlatesOverview ](https://www.kaggle.com/code/trainingdatapro/licenseplatesoverview) created by Training Data Pro. Unthankfully the full dataset is paywalled, thus unavailable for free, as explicitly stated by the author, available in full [here](https://trainingdata.pro/data-market), but the free sample dataset was enough for me to adapt the dataset and collator class to work with the full dataset as well.

This new dataset came with a new set of challenges in implementing the data loading stage, as this time, the images were not of just of license plates, but of cars in various real-life situations, with bounding box coordinates being offered for the license plates. This meant that the license plate images had to be cropped from the raw images automatically, based on the offered bounding boxes, before being utilized within the model.

Once the images were cropped and the corresponding license plate numbers were stored in `.txt` files with the same name as the respective image, the dataset could then be loaded into the model. One downside of the cropping is that the new images have a lower resolution compared to the dedicated images from the previous dataset (just 64 pixels for the height dimension), this translates to a lower amount of base information that the model could use, not just that but there were only 142 total images of license plates available after cropping, an even lower amount that what was previously used.

Suffice to say, the model's performance on this data sample is extremely poor, despite using all the previous improvements, both architecture wise and augmentation wise, the resulting character accuracy was only `4.6`, this way only 4.6% of the characters were correctly identified by the model and the Precision, Recall and F1 score reflect this poor performance with values of `0.08`, `0.02` and `0.03`.

Despite this seemingly poor performance, I have no doubt that if the full dataset were to be used with this iteration of the model, the overall performance would be significantly higher and thanks to this datasample, using the full dataset wouldn't be difficult at all.


> More detailed information and explanations on each individual model can be found within their respective notebook files
