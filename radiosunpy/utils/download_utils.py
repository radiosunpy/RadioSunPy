from radiosunpy.client.solarmonitor_client import SolarMonitorClient
from radiosunpy.client import RATANClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import csv


# Function to process a single time_date and return the URLs for
def fetch_one_url(url):
    smc = SolarMonitorClient()
    return smc.get_data_from_url(url)


def fetch_many_urls(url_list):
    data = []
    refused_urls = []

    # Using ThreadPoolExecutor for parallel URL fetching
    with ThreadPoolExecutor() as executor:
        # Submitting tasks for each time_date in the list
        futures = {executor.submit(fetch_one_url, url): url for url in url_list}

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


def write_data_to_csv(data, file_path):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file did not exist, we can write a header (optional)
        if not file_exists:
            writer.writerow(['Data',
                             'noaa_number',
                             'mcintosh_class',
                             'c_class_mcevol',
                             'c_class_mcstat',
                             'c_class_noaa',
                             'm_class_mcevol',
                             'm_class_mcstat',
                             'm_class_noaa',
                             'x_class_mcevol',
                             'x_class_mcstat',
                             'x_class_noaa'])

        writer.writerows(data)


def fetch_ratan_url(url, save_path):
    #save_path = '../../GAO/RatanFits'
    file_path = save_path + '/' + url.split('/')[-1].split('.')[0] + '_processed.fits'
    if os.path.exists(file_path):
        print(f'File {url} already exists')
        return
    else:
        ratan_client = RATANClient()
        _, _ = ratan_client.process_fits_data(
            url,
            save_path=save_path,
            save_with_original=False,
            save_raw=False)
        return


def fetch_ratan_urls_parallel(ratan_url_list, save_path):
    # Using ThreadPoolExecutor to parallelize the fetching
    with ThreadPoolExecutor() as executor:
        # Submitting tasks for each time_date in the list
        futures = {executor.submit(fetch_ratan_url, url, save_path): url for url in ratan_url_list}

    # Processing results as they complete
    for future in tqdm(as_completed(futures), total=len(ratan_url_list)):
        try:
            # Extend the url_list with the result of the future
            future.result()
            # Extend the url_list with the result
        except Exception as e:
            # If an error occurs, log it and continue
            url = futures[future]  # Retrieve the associated time_date
            print(f"Error fetching URLs for {url}: {e}")

    return