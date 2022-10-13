import os.path
import urllib.request

from flickr_search_images import read_cvs_file, read_excluded_file


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
    dataset_name = 'geotags_reconstructed'
    #data_dir = f'/home/hacke/projects/data/geolocation_classifier/{dataset_name}'
    data_dir = f'/mnt/store/geolocation_classifier/datadir/{dataset_name}'

    down_load_images(dataset_name, data_dir)


if __name__ == '__main__':
    main()
