[![PyPI Versions](https://img.shields.io/pypi/pyversions/av2)](https://pypi.org/project/av2/)
![CI Status](https://github.com/argoai/argoverse2-api/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

# Argoverse 2 API

Official GitHub repository for the [Argoverse 2](https://www.argoverse.org) family of datasets.

If you have any questions or run into any problems with either the data or API, please feel free to open a [GitHub issue](https://github.com/argoai/argoverse2-api/issues)!

## Overview

- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing-automation)
- [Contributing](#contributing)
- [Citing](#citing)
- [License](#license)

## Getting Started

### Setup

The easiest way to install the API is via [pip](https://pypi.org/project/av2/) by running the following command:

```bash
pip install av2
```

### Downloading the datasets
Please see [the Download README](DOWNLOAD.md) for detailed instructions on how to download each dataset.

### Argoverse 2 Sensor Dataset

<p align="center">
  <img src="https://user-images.githubusercontent.com/29715011/158742778-557f31a4-569d-44aa-a032-99836094dc97.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158742776-069501c4-8dd4-4f9d-ac8c-f0421f855607.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158739736-fe876299-23da-46ed-98ce-173f938d1702.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158739767-886e1c2f-4613-495d-9204-a7b4813af16d.gif" height="150">
</p>

Please refer to the [sensor dataset README](src/av2/datasets/sensor/README.md) for additional details.

### Argoverse 2 Lidar Dataset

<p align="center">
  <img src="https://user-images.githubusercontent.com/29715011/158715494-472339d1-a5d5-4d33-8fcf-3455c0d78d27.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158715496-f439ccad-71af-4880-8b43-ade7b6c8f333.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158715498-23d7a11f-12a1-4aeb-b9af-dbced217b340.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158715497-d1603423-c32f-4cf0-ab1e-6bbc9c458535.gif" height="150">
</p>

Please refer to the [lidar dataset README](src/av2/datasets/lidar/README.md) for additional details.

### Argoverse 2 Motion Forecasting Dataset

<p align="center">
  <img src="https://user-images.githubusercontent.com/29715011/158486284-1a0df794-ee0a-4ae6-a320-0dd0d1daad06.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158486286-e734e654-b879-4994-a129-9957cc591af4.gif" height="150">
  <img src="https://user-images.githubusercontent.com/29715011/158486288-5e7c0971-de0c-4ff5-bea7-76f7922dd1e0.gif" height="150">
</p>


Please refer to the [motion forecasting dataset README](src/av2/datasets/motion_forecasting/README.md) for additional details.

### Argoverse 2 Map Change Dataset

Please refer to the map change dataset (known as the **Trust, but Verify Dataset**) [README](src/av2/datasets/tbv/README.md) for additional details.

### Map API

Please refer to the [map README](src/av2/map/README.md) for additional details about the common format for vector and
raster maps that we employ across all AV2 datasets.

## Compatibility Matrix

| `Python Version`   |     `linux`        |     `macOS`        |    `windows`       |
| -------------      | -----------------  | ------------------ | ------------------ |
| `3.8`              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| `3.9`              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| `3.10`             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

## Testing

All incoming pull requests are tested using [nox](https://nox.thea.codes/en/stable/) as
part of the CI process. This ensures that the latest version of the API is always stable on all supported platforms. You
can run the full suite of automated checks and tests locally using the following command:

```bash
nox -r
```

## Contributing

Have a cool feature you'd like to add? Found an unhandled corner case? The Argoverse team welcomes contributions from
the open source community - please open a PR using the following [template](.github/pull_request_template.md)!

## Citing

Please use the following citation when referencing the [Argoverse 2](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/4734ba6f3de83d861c3176a6273cac6d-Paper-round2.pdf) Sensor, Lidar, or Motion Forecasting Datasets:

```BibTeX
@INPROCEEDINGS { Argoverse2,
  author = {Benjamin Wilson and William Qi and Tanmay Agarwal and John Lambert and Jagjeet Singh and Siddhesh Khandelwal and Bowen Pan and Ratnesh Kumar and Andrew Hartnett and Jhony Kaesemodel Pontes and Deva Ramanan and Peter Carr and James Hays},
  title = {Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks 2021)},
  year = {2021}
}
```

Use the following citation when referencing the [Argoverse 2](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/6f4922f45568161a8cdf4ad2299f6d23-Paper-round2.pdf) Map Change Dataset:
```BibTeX
@INPROCEEDINGS { TrustButVerify,
  author = {John Lambert and James Hays},
  title = {Trust, but Verify: Cross-Modality Fusion for HD Map Change Detection},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks 2021)},
  year = {2021}
}
```

## License

All code provided within this repository is released under the MIT license and bound by the Argoverse terms of use,
please see [LICENSE](LICENSE) and [NOTICE](NOTICE) for additional details.
