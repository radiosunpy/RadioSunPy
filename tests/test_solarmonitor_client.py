import pytest
import numpy as np
from datetime import timedelta
from pathlib import Path
from radiosunpy.time import TimeRange
from radiosunpy.client.solarmonitor_client import SolarMonitorClient


@pytest.fixture
def tr():
    """
    Fixture that provides two Time objects that are more than 1 nanosecond apart.
    """
    tr = TimeRange('2017-09-03', '2017-09-03')
    return tr


@pytest.fixture
def url():
    return "https://solarmonitor.org/data/2010/01/02/meta/arm_forecast_20100102.txt"


@pytest.fixture
def url_many():
    return "https://solarmonitor.org/data/2024/08/28/meta/arm_forecast_20240828.txt"


@pytest.fixture
def smc():
    """
    Fixture that provides two Time objects that are more than 1 nanosecond apart.
    """
    smc = SolarMonitorClient()
    return smc


class TestSolarMonitorClient:

    def test_acquire_data(self, tr, smc):
        file_urls = smc.acquire_data(tr)
        assert len(file_urls) == 1
        assert isinstance(file_urls[0], str)

    def test_parse_date_from_url(self, url, smc):
        dates = smc.parse_date_from_url(url)
        assert len(dates) == 1
        assert isinstance(dates, str)

    def test_parse_line(self, smc):
        line = '11024 Dso 20(40) 4(5) 0(1)'
        pred_info = smc.parse_line(line)
        assert len(pred_info) == 1


    def test_get_data_from_empty(self, smc):
        url = 'https://solarmonitor.org/data/2011/03/15/meta/arm_forecast_20110315.txt'
        data = smc.get_data_from_url(url)
        assert len(data) == 0

    def test_get_data_from_url(self, url, smc):
        data = smc.get_data_from_url(url)
        assert isinstance(data, list)


    def test_get_data_from_url_many_events(self, url_many, smc):
        data = smc.get_data_from_url(url_many)
        assert isinstance(data, list)
