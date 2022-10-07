import os.path
import urllib.request

from flickr_search_images import read_cvs_file


def down_load_images(file_name: str, data_dir: str):

    if not os.path.isdir(data_dir):
        raise RuntimeError(f'Illegal datadir {data_dir}')

    images = read_cvs_file(file_name)

    total = len(images)
    cnt = 1
    for id in images:
        url = images[id]['url']
        name = url.split('/')[-1]

        if os.path.isfile(data_dir+'/'+name):
            print(f'Already existing {cnt}/{total} {name}')
        else:
            print(f'Downloading {cnt}/{total} {name}')
            urllib.request.urlretrieve(url, data_dir+'/'+name)

        cnt += 1


def main():
    #data_dir = '/home/hacke/projects/data/geolocation_classifier'
    data_dir = '/mnt/store/geolocation_classifier/datadir'
    file_name = 'input/flickr_images.csv'
    down_load_images(file_name, data_dir)


if __name__ == '__main__':
    main()
