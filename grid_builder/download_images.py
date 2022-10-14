import os.path
import threading
import urllib.request

from flickr_search_images import read_cvs_file, read_excluded_file


def thread_download(urls, data_dir):
    cnt = 1
    total = len(urls)

    for url in urls:
        name = url.split('/')[-1]
        file_name = data_dir + '/' + name
        if os.path.isfile(file_name):
            print(f'ERROR Already existing{file_name}')
            raise RuntimeError(f'ERROR Already existing{file_name}')

        try:
            urllib.request.urlretrieve(url, file_name)
            print(f'Downloaded {cnt}/{total} {file_name}')
        except urllib.error.HTTPError as e:
            print(f'Downloading {cnt}/{total} {name} failed: {e.code} {e.msg}')

        cnt += 1


def concurrent_down_load_images(num_threads: int, dataset_name: str, data_dir: str):

    if not os.path.isdir(data_dir) or not data_dir.endswith(dataset_name):
        raise RuntimeError(f'Illegal datadir {data_dir}')

    images = read_cvs_file(f'input/{dataset_name}.csv')
    excluded = read_excluded_file(f'input/{dataset_name}_excluded.csv')

    url_list = []

    for id in images:

        if id in excluded:
            continue

        url = images[id]['url']
        name = url.split('/')[-1]

        if not os.path.isfile(data_dir + '/' + name):
            url_list.append(url)




    threads = list()
    thread_len = int(len(url_list) / num_threads) + num_threads

    start = 0
    total = 0
    for idx in range(num_threads):
        end = min(start + thread_len, len(url_list))

        if start == end:
            break

        thread_list = url_list[start:end]
        start = end
        thread = threading.Thread(target=thread_download, args=(thread_list, data_dir,))
        print(f'Thread {idx} downloading {len(thread_list)}/{len(url_list)}')
        threads.append(thread)
        total += len(thread_list)

    assert total == len(url_list)


    for thread in threads:
        thread.start()

    for index, thread in enumerate(threads):
        thread.join()




def down_load_images(dataset_name: str, data_dir: str):

    if not os.path.isdir(data_dir) or not data_dir.endswith(dataset_name):
        raise RuntimeError(f'Illegal datadir {data_dir}')

    images = read_cvs_file(f'input/{dataset_name}.csv')
    excluded = read_excluded_file(f'input/{dataset_name}_excluded.csv')

    total = len(images) - len(excluded)
    cnt = 1

    # for id in reversed(list(images.keys())):
    for id in images:

        if id in excluded:
            continue

        url = images[id]['url']
        name = url.split('/')[-1]

        if os.path.isfile(data_dir+'/'+name):
            print(f'Already existing {cnt}/{total} {name}')
        else:
            try:
                urllib.request.urlretrieve(url, data_dir+'/'+name)
                print(f'Downloaded {cnt}/{total} {name}')
            except urllib.error.HTTPError as e:
                print(f'Downloading {cnt}/{total} {name} failed: {e.code} {e.msg}')

        cnt += 1


def main():
    dataset_name = 'flickr_images'
    #data_dir = f'/home/hacke/projects/data/geolocation_classifier/{dataset_name}'
    data_dir = f'/mnt/store/geolocation_classifier/datadir/{dataset_name}'

    concurrent_down_load_images(20, dataset_name, data_dir)
    #down_load_images(dataset_name, data_dir)


if __name__ == '__main__':
    main()
