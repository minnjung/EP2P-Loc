# EP2P-Loc: End-to-End 3D Point to 2D Pixel Localization for Large-Scale Visual Localization (ICCV 2023)
Official repository of ["EP2P-Loc: End-to-End 3D Point to 2D Pixel Localization for Large-Scale Visual Localization"](https://arxiv.org/pdf/2309.07471.pdf).

![EP2P-Loc model](images/network.png)

We propose **EP2P-Loc**, a novel large-scale visual localization method that mitigates such appearance discrepancy and enables end-to-end training for pose estimation. This repository is built upon the foundations of [Swin-Transformer](https://github.com/SwinTransformer/Transformer-SSL), [Fast Point Transformer](https://github.com/POSTECH-CVLab/FastPointTransformer), and [EPro-PnP](https://github.com/tjiiv-cprg/EPro-PnP).

**Updates**
- Aug 18, 2023: Release benchmark datasets
- Jul 18, 2023: Initial commit


## Requirements
* Ubuntu 16.04
* Python 3.6
```
conda create -n ep2ploc python=3.6
conda activate ep2ploc

pip install -r requirements.txt
conda install -c sirokujira python-pcl --channel conda-forge
```


## Dataset
### Download datasets
* [2D-3D-S and (Aligned) S3DIS](http://buildingparser.stanford.edu/dataset.html)
* [KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)


### Preprocessing
```
cd datasets

# 2D-3D-S
python preprocess_2d3ds.py --data_path <2D-3D-S_path> --s3dis_path <S3DIS_path> --cache_path <cache_path(optional)> --save_path <save_path>

# KITTI
python preprocess_kitti.py --data_path <KITTI_path> --save_path <save_path>
```


## Training and Testing
TBU


## Citation
```
@INPROCEEDINGS{EP2PLoc2023ICCV,
  author = {Kim, Minjung and Koo, Junseo and Kim, Gunhee},
  title = {EP2P-Loc: End-to-End 3D Point to 2D Pixel Localization for Large-Scale Visual Localization},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year = {2023}
}
```
