from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm
from radiosunpy.time import TimeRange
from astropy.time import Time, TimeDelta
from radiosunpy.client import BaseClient
import pandas as pd
import numpy as np
from radiosunpy.scrapper import Scrapper
from urllib.request import urlopen
from urllib.parse import urlparse
import re
from astropy.table import Table

__all__ = ['SolarMonitorClient']


class SolarMonitorClient(BaseClient):

    base_url = 'https://solarmonitor.org/data/%Y/%m/%d/meta/arm_forecast_%Y%m%d.txt'
    regex_pattern = '(arm_forecast_(\d{8})\.txt)'

    def acquire_data(self, timerange: TimeRange) -> list[str]:
        scrapper = Scrapper(self.base_url, self.regex_pattern)
        return scrapper.form_fileslist(timerange)

    def parse_date_from_url(self, url):
        path = urlparse(url).path
        match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', path)
        if match:
            year, month, day = match.groups()
            return f'{year}-{month}-{day}'
        return None

    def parse_field(self, field):
        if field == '...':
            return None
        return int(field)

    def parse_line(self, line):
        parts = line.split()
        noaa_number = parts[0]
        mcintosh_class = parts[1]

        # C-class fields
        c_class_fields = parts[2].split('(')
        c_class_mcevol = self.parse_field(c_class_fields[0])
        c_class_mcstat = self.parse_field(c_class_fields[1].rstrip(')')) if len(c_class_fields) > 1 else None
        if len(c_class_fields) == 3:
            c_class_noaa = self.parse_field(c_class_fields[2].rstrip(')')) if len(c_class_fields) > 1 else None
        else:
            c_class_noaa = 'none'

        # M-class fields
        m_class_fields = parts[3].split('(')
        m_class_mcevol = self.parse_field(m_class_fields[0])
        m_class_mcstat = self.parse_field(m_class_fields[1].rstrip(')')) if len(m_class_fields) > 1 else None
        if len(m_class_fields) == 3:
            m_class_noaa = self.parse_field(m_class_fields[2].rstrip(')')) if len(m_class_fields) > 1 else None
        else:
            m_class_noaa = 'none'

        # X-class fields
        x_class_fields = parts[4].split('(')
        x_class_mcevol = self.parse_field(x_class_fields[0])
        x_class_mcstat = self.parse_field(x_class_fields[1].rstrip(')')) if len(x_class_fields) > 1 else None
        if len(m_class_fields) == 3:
            x_class_noaa = self.parse_field(x_class_fields[2].rstrip(')')) if len(x_class_fields) > 1 else None
        else:
            x_class_noaa = 'none'

        return [
            noaa_number,
            mcintosh_class,
            c_class_mcevol,
            c_class_mcstat,
            c_class_noaa,
            m_class_mcevol,
            m_class_mcstat,
            m_class_noaa,
            x_class_mcevol,
            x_class_mcstat,
            x_class_noaa
        ]

    def get_data_from_url(self, url: str):

        with urlopen(url) as response:
            content = response.read().decode('utf-8').split('\n')
            date_text = self.parse_date_from_url(url)
            table_data = [[date_text] + self.parse_line(line.strip()) for line in content if line.strip()]
            return table_data

    def get_data(self, url_list: list[str]):
        data = []
        refused_urls = []

        # Using ThreadPoolExecutor for parallel URL fetching
        with ThreadPoolExecutor() as executor:
            # Submitting tasks for each time_date in the list
            futures = {executor.submit(self.get_data_from_url, url): url for url in url_list}

        # Processing results as they complete
        for future in tqdm(as_completed(futures), total=len(url_list)):
            try:
                # Extend the url_list with the result of the future
                data.extend(future.result())  # Extend the url_list with the result
            except Exception as e:
                # If an error occurs, log it and continue
                url = futures[future]  # Retrieve the associated time_date
                refused_urls.append(url)
                print(f"Error fetching URLs for {url}: {e}")

        return data, refused_urls


    def form_data(self):
        pass
