import pytest
from pathlib import Path
from radiosun.time.timerange import TimeRange
from radiosun.client.client import SRSClient, RATANClient
from urllib.request import urlopen
from astropy.table import  Table
from astropy.io import fits


@pytest.fixture
def tr():
    """
    Fixture that provides two Time objects that are more than 1 nanosecond apart.
    """
    tr = TimeRange('2017-09-03', '2017-09-03')
    return tr

@pytest.fixture
def srs():
    """
    Fixture that provides two Time objects that are more than 1 nanosecond apart.
    """
    srs = SRSClient()
    return srs

@pytest.fixture
def ratan_client():
    return RATANClient()

@pytest.fixture
def r_fits_url():
    ratan_client = RATANClient()
    tr = TimeRange('2017-09-03', '2017-09-03')
    url = ratan_client.acquire_data(tr)[0]
    return url


class TestSRSClient:

    def test_acquire_data(self, tr, srs):
        file_urls = srs.acquire_data(tr)
        assert len(file_urls) == 1
        assert isinstance(file_urls[0], str)

    def test_extract_lines(self, tr,srs):

        file_url = srs.acquire_data(tr)[0]
        with urlopen(file_url) as response:
            content = response.read().decode('utf-8').split('\n')
            header, section_lines, supplementary_lines = srs.extract_lines(content)
        assert len(header) == 11
        assert len(section_lines) == 3

    def test_form_data(self, tr,srs):
        file_url = srs.acquire_data(tr)
        data = srs.form_data(file_url)
        assert len(data) == 4
        assert isinstance(data, Table)

    def test_get_data(self, tr, srs):
        data = srs.get_data(tr)
        assert len(data) == 4
        assert isinstance(data, Table)


class TestRATANClient:

    def test_acquire_data(self, tr, ratan_client):
        urls = ratan_client.acquire_data(tr)
        assert len(urls) == 1
        assert isinstance(urls[0], str)

    def test_get_scans(self, tr, ratan_client):
        tables = ratan_client.get_scans(tr)
        assert len(tables) == 1
        assert isinstance(tables[0], Table)

    def test_process_fits_data(self, ratan_client, r_fits_url):

        raw, processed = ratan_client.process_fits_data(r_fits_url,
                                                   save_path=None,
                                                   save_with_original=False)
        assert isinstance(processed, fits.hdu.hdulist.HDUList)
        assert isinstance(raw, fits.hdu.hdulist.HDUList)
        assert processed[1].header['NAXIS'] == 2
        assert processed[1].name == 'I'
        assert processed[2].header['NAXIS'] == 2
        assert processed[2].name == 'V'

    def test_process_fits_data_with_saving(self, ratan_client, r_fits_url):
        save_path = Path(__file__).absolute().parents[1]/'data'
        raw, processed = ratan_client.process_fits_data(r_fits_url,
                                                       save_path=save_path,
                                                       save_with_original=False,
                                                       save_raw=True
                                                    )
        assert isinstance(processed, fits.hdu.hdulist.HDUList)

    def test_process_fits_data_from_disk(self, ratan_client):
        save_path = Path(__file__).absolute().parents[1]/'data'
        data_path = Path(__file__).absolute().parents[1]/'data'/'20170903_121257_sun+0_out.fits'
        raw, processed = ratan_client.process_fits_data(data_path,
                                                       save_path=save_path,
                                                       save_with_original=False,
                                                       save_raw=False
                                                    )
        assert isinstance(processed, fits.hdu.hdulist.HDUList)
