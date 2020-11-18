# REQUIREMENTS

## Python 3.6
* GDAL
* numpy
* h5py
* pandas
* pydtw
* scikit-image
* scikit-learn

To install GDAL on Ubuntu
```
# if installing into virtual environment
sudo apt-get install python3.6-dev
```
```
sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
sudo apt-get install gdal-bin
sudo apt-get install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
pip install GDAL
```

To install remaining packages from the `requirements.txt` file
```
pip install -r requirements.txt
```

## R 4.0.0
Install the following packages from CRAN repositories

* doParallel
* rhdf5
* lubridate
* zoo
* sandwich
* xts
* raster

```
install.packages("package_name")
```

Install `bfast` and `strucchange` packages from [here](https://github.com/bfast2/bfast) with

```
library(devtools)
install_github("bfast2/strucchange")
install_github("bfast2/bfast")
```


# USAGE

Before running any of the modules, update the `config.ini` file with the correct information regarding data, paths and module parameters

## Module 1 - Feature Extraction
```
python main.py -c /path/to/config.ini -m1
```

Input: Sentinel-2 (L2A) tile data acquired over N years <br />
Output: multi-band NDI image for each acquisition date

Extracted features: `(B1-B2)/(B1+B2)`
1. SWIR2 and BLUE
2. BLUE and RED
3. SWIR1 and GREEN
4. BLUE and SWIR1
5. SWIR2 and NIR
6. SWIR1 and RED
7. SWIR2 and RED
8. NIR and RED

## Module 2 - Time Series Reconstruction
```
python main.py -c /path/to/config.ini -m2
```

Input: multi-band NDI images <br />
Output: `ts.h5` generated for each NDI and year, containing for each pixel of the image the reconstructed time series values

pixel | LC class | NDI | NDI annual values
:---: | :---: | :---: | :---: 
0 | 4 | 1 | 13.4 15.8 18.9 ... 
1 | 3 | 1 | 44.2 35.7 23.8 ...
2 | 3 | 1 | 18.9 23.9 31.3 ...

## Module 3 - BFAST Trend Analysis
```
python main.py -c /path/to/config.ini -m3
```

Input: NDI time series `ts.h5` files for the N years <br />
Output: two-band image for each NDI, providing at each pixel position the year of the potential change detected by BFAST and a confidence value

In the configuration file, select `frequency = 52` to reduce the computational time by aggregating the annual time series into weekly ones before applying BFAST

## Module 4 - Land Cover Training
```
python main.py -c /path/to/config.ini -m4
```

Input: NDI time series `ts.h5` files <br />
Output: class models

In the configuration file, select `weekly = True` to reduce the computational time by aggregating the annual time series into weekly ones before computing the DTW

## Module 5 - LC Classification and Change Detection
```
python main.py -c /path/to/config.ini -m5
```

Input: NDI time series `ts.h5` files for the N years and class models <br />
Output: two-band image, providing at each pixel position the year of the potential change and an accuracy value


# HW RESOURCES
Required hardware resources depend on the dimension of the analyzed tile images (total number of pixels). In order to adapt the software to different types of hardware, the user can adjust for each module the block/batch of pixels loaded to memory and processed

In particular:
* Module 2 `blocksize` refers to blocks of pixels within the original MxM tile image
* Module 3 `batchsize` refers to batches of pixels in the time series `ts.h5` file structure
* Module 4 `blocksize` refers to blocks of pixels in the DTW M^2xM^2 matrix