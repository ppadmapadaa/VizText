# VizText - Multimodal Sentiment Analysis

## Overview
VizText is a multimodal sentiment analysis system that uses both text and images to predict sentiment (positive, neutral, negative).

## Model
- Text Model: RoBERTa
- Image Model: ResNet101
- Fusion: Early Fusion + PCNN

## Dataset
- MVSA-Single dataset

## How to Run
1. Install dependencies:
   pip install transformers torch torchvision scikit-learn pillow tqdm

2. Run:
   python VizText.py

## Results
The fusion model achieved higher accuracy compared to individual text/image models.

## Notes
- Requires dataset path setup
- Best run on GPU (Colab recommended)
