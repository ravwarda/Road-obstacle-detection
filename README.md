# Road obstacle detection

This project focuses on detecting road obstacles using image segmentation and object detection techniques. The project utilizes the [\[Nvidia Segformer model\]](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) for semantic segmentation and a custom CNN model for object detection.

## Project Structure

- `main.py`: Main script to train models, test on images, and process videos.
- `segformer_train.py`: Contains functions for training the Segformer model.
- `image_processing.py`: Contains functions for image processing and visualization.
- `cnn.py`: Contains functions for training the CNN model and performing object detection.

## Datasets
[\[RDD2020\]](https://data.mendeley.com/datasets/5ty2wb6gvg/1)
[\[BDD100k\]](https://www.vis.xyz/bdd100k/)
 
