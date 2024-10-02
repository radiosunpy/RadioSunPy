from radiosunpy.utils.download_utils import (fetch_one_url,
                                             fetch_many_urls,
                                             write_data_to_csv,
                                             fetch_ratan_url,
                                             fetch_ratan_urls_parallel)
import pytest
import numpy as np
import csv
import os


@pytest.fixture
def url_list():
    with open('../../GAO/SolarMonitorUrls.txt') as file:
        url_list = [line.rstrip() for line in file]
    url_list = list(map(str, np.unique(url_list)))
    return url_list

@pytest.fixture
def ratan_url_list():
    with open('../../GAO/RatanUrls.txt') as file:
        ratan_url_list = [line.rstrip() for line in file]

    return ratan_url_list


def test_fetch_one_url(url_list):
    url = url_list[0]
    data = fetch_one_url(url)
    assert isinstance(data, list)


def test_fetch_many_urls(url_list):
    data = fetch_many_urls(url_list[:4])
    assert isinstance(data, list)
    assert len(data[0]) == 12

def test_write_data_to_csv(url_list):

    file_path = '../../GAO/SolarMonitorPredsTest.csv'
    data = fetch_many_urls(url_list[:4])
    write_data_to_csv(data, file_path)
    assert os.path.exists(file_path)


def test_fetch_ratan_url(ratan_url_list):
    url = ratan_url_list[-1]
    save_path = '../../GAO/RatanFits'
    fetch_ratan_url(url, save_path)
    assert os.path.exists(save_path)

def test_fetch_ratan_urls_parallel(ratan_url_list):
    save_path = '../../GAO/RatanFits'
    fetch_ratan_urls_parallel(ratan_url_list[-3:], save_path)
    assert os.path.exists(save_path)

