# get\_dataset.md

This document explains how to download and prepare the dataset required for running this project, specifically the **Kenian Cuisine Image Clasifier** data from the Kaggle competition.

## Prerequisites

1. **Kaggle account**: You need a Kaggle account to access the competition data. If you don't have one, register at [https://www.kaggle.com](https://www.kaggle.com).
2. **Kaggle API token**: Generate an API token:

   * Go to your Kaggle account settings ([https://www.kaggle.com/\`](https://www.kaggle.com/`)<your-username>\`/account).
   * Scroll to **API** and click **Create New API Token**.
   * A file named `kaggle.json` will be downloaded.
3. **Kaggle CLI**: Install the Kaggle command-line interface:

   ```bash
   pip install kaggle
   ```
4. **Place the token**: Move `kaggle.json` to the configuration directory:

   ```bash
   mkdir -p ~/.kaggle
   mv /path/to/downloaded/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

## Downloading the Dataset

### Using Kaggle CLI

**Download competition data**:

   ```bash
   mkdir dataset/
   kaggle competitions download \
     -c opencv-pytorch-project-2-classification-round-3 \
     -p ./dataset/
   ```

   This will download a zip file (e.g., `opencv-pytorch-project-2-classification-round-3.zip`) into the project's root folder.

### Alternative Download (Without Kaggle CLI)

1. Open your web browser and navigate to the competition page:
   [https://www.kaggle.com/competitions/opencv-pytorch-project-2-classification-round-3](https://www.kaggle.com/competitions/opencv-pytorch-project-2-classification-round-3)
2. Log in with your Kaggle account and accept the competition rules.
3. On the **Data** tab, click **Download All** to download the ZIP file.
4. Place the downloaded ZIP into project's root directory:

   ```bash
   mkdir dataset/
   mv ~/Downloads/opencv-pytorch-project-2-classification-round-3.zip ./dataset
   ```
