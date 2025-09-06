🗓️ **Last updated:** September 6, 2025 

# **GraphCSVAE: Graph Categorical Structured Variational Autoencoder for Spatiotemporal Auditing of Physical Vulnerability Towards Sustainable Post-Disaster Risk Reduction**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


## **1. Installation**

This code depends on [MATLAB R2025b](https://uk.mathworks.com/), [QGIS 3.40.5-Brastislava](https://www.qgis.org/en/site/forusers/download.html), or any newer versions. The MATLAB toolboxes for [Mapping](https://uk.mathworks.com/products/mapping.html), [Financial](https://uk.mathworks.com/products/finance.html), [Statistics and Machine Learning](https://uk.mathworks.com/help/stats/getting-started-12.html), and [Deep Learning](https://uk.mathworks.com/help/deeplearning/ug/deep-learning-in-matlab.html) must also be installed to enable the data import and export of GeoTIFF files (*.tif) and perform the deep learning training. Educational license is available for schools and universities.

## **2. Data**

| Data Repository  | Link |
| ------------- | ------------- |
| A Local-Scale Dataset of Annual Spatiotemporal Maps of Physical Vulnerability in the Cyclone-Impacted Coastal Khurushkul Community (Bangladesh) and Mudslide-Affected Freetown (Sierra Leone) (2016–2023) via Graph Variational State-Space Model (GraphVSSM)  | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16656471.svg)](https://doi.org/10.5281/zenodo.16656471)  |

## **3. Code**

1. In `code\main\`:
   - `KhurushkulBGD.m` - a script for training and inference for the case study in Khurushkul, BGD.
   - `FreetownSLE.m` - a script for training and inference for the case study in Freetown, SLE.
2. In `code\models\OV\`:
   - `decoderOV.m` - a function for decoder network
   - `encoderOV.m` - a function for encoder network
   - `initializePOV.m` - a function that initializes the encoder-decoder architecture
3. In `code\helper\`
   - `trainingFigs\` - a set of functions to initialize figures for training
   - `utility\` - a set of functions for various purposes like construction of grid-based adjacency matrices
4. In `gnn\`: contains the trained GraphCSVAE models
5. In `plot\`: contains the corresponding loss plots

## **4. Repository Structure**
```
.
├── LICENSE
├── code
│   ├── helper
│   │   ├── trainingFigs
│   │   │   └── initializeOVfig.m
│   │   └── utility
│   │       ├── initializeGlorot.m
│   │       ├── modelOVlossSHOCK.m
│   │       └── sparse_dilate_ultra.m
│   ├── main
│   │   ├── FreetownSLE.m
│   │   └── KhurushkulBGD.m
│   └── models
│       └── OV
│           ├── decoderOV.m
│           ├── encoderOV.m
│           ├── initializePOV.m
│           └── modelOVloss.m
├── gnn
│   ├── FreetownSLE
│   │   └── GraphVSSM_OVparameters.mat
│   └── KhurushkulBGD
│       └── GraphVSSM_OVparameters.mat
├── plot
│   ├── FreetownSLE
│   │   ├── OV_KL.fig
│   │   └── OV_ReconLoss.fig
│   └── KhurushkulBGD
│       ├── OV_KL.fig
│       └── OV_ReconLoss.fig
└── README.md
```

## **5. Computing Infrastructure**
We performed all experiments using a MacBook Pro (Apple M3 Max) with 48GB memory. Fortunately, our experiments did not need to use GPU. Due to ease of implementation
and our familiarity, we used the deep learning and mapping toolboxes of MATLAB. However, other software libraries and frameworks can be used to reproduce our results.

## **6. Acknowledgment**
This work is funded by the UKRI Centre for Doctoral Training in Application of Artificial Intelligence to the study of Environmental Risks (AI4ER) (EP/S022961/1).

## **7. Citation**

```
@conference{example_conference,
  title        = {GraphCSVAE: Graph Categorical Structured Variational Autoencoder for Spatiotemporal Auditing of Physical Vulnerability Towards Sustainable Post-Disaster Risk Reduction},
  author       = {Dimasaka, Joshua and Gei{\ss}, Christian and Muir-Wood, Robert and So, Emily},
  year         = 2025,
  month        = {October},
  booktitle    = {Proceedings of the 8th International Disaster and Risk Conference},
  address      = {Nicosia, Cyprus}
}
```