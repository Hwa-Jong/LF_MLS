# LF_MLS

## Official Code

----------------
## Environment
- Python 3.6.9
- torch 1.9.0+cu111
- torchvision 0.10.0+cu111
- cv2 4.5.3
- PIL 8.3.1
- skimage 0.17.2
----------------
## Dataset URL
###### HCI_old : http://lightfield-analysis.net/hci_database/download_lightfields.sh
###### HCI_new : https://lightfield-analysis.uni-konstanz.de/
###### EPFL : http://plenodb.jpeg.org/lf/epfl/
###### Stanford : http://lightfields.stanford.edu/LF2016.html
###### INRIA : http://clim.inria.fr/research/LowRank2/datasets/datasets.html
###### Kalantari : https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/SIGASIA16/

----------------
## Usage

### Make dataset
#### Make Train dataset
> ```
> python3 make_train_data.py 
> ```

#### Make Test dataset
###### You have to change 'test_dataset' in make_test_data.py (default:'HCI_new') [HCI_old HCI_new EPFL Stanford INRIA]
> ```
> python3 make_test_data.py 
> ```

### Train
> ```
> python3 train.py --device "cuda:0"
> ```

### Test
> ```
> python3 test.py --dataset_name HCI_new --device "cuda:0"
> ```

