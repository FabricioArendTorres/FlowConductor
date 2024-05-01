"""
https://zenodo.org/records/1161203#.Wmtf_XVl8eN
https://doi.org/10.5281/zenodo.1161203

"""
import os
import platform
import subprocess

import logging
from pathlib import Path

url_data = "https://zenodo.org/records/1161203/files/data.tar.gz?download=1"

# not used in this stub but often useful for finding various files
uci_dir = Path(__file__).resolve().parents[0]
uci_tar_path = uci_dir / "uci_data.tar.gz"

os.makedirs(uci_dir, exist_ok=True)


def preprocess_uci_data() -> None:
    from flowcon.datasets.uci import gas, power, miniboone, hepmass

    gas.save_splits()
    power.save_splits()
    miniboone.save_splits()
    hepmass.save_splits()


def download_uci_data() -> None:
    # res = urllib.request.urlretrieve(url_data_2018, path_data_raw / "vertical_profiles.json")
    os.makedirs(uci_dir, exist_ok=True)
    is_empty_dir = os.listdir(uci_dir)
    if platform.system() != 'Linux':
        logging.info(f"This script only supports Linux, as it uses wget to download the data. "
                     f"If you are not on Linux, please manually download the zip from the following URL: \n {url_data}  ")
        exit()
    # download zip
    if not uci_tar_path.exists():
        logging.info("Downloading UCI data from Zenodo. ")
        res = subprocess.call(["wget", "-O", uci_tar_path, url_data])
    # extract zip
    logging.info("Extracting vertical profiles from .zip file. ")
    # res = subprocess.call(["unzip", uci_zipped_path, "-d", uci_data_dir])
    res = subprocess.call(["tar", "-xf", uci_tar_path, "-C", uci_dir])
    res = subprocess.call(["rm", uci_tar_path])

def download_and_preprocess_uci_data():
    download_uci_data()
    preprocess_uci_data()

if __name__ == '__main__':
    download_uci_data()
    preprocess_uci_data()
