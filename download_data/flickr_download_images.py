import os.path
import csv

import flickrapi


CSV_HEADER = ['# id', 'url', 'longitude', 'latitude']


def initialize_flickr(authorize: bool=False):
    # user hartmut keil
    api_key = u'200a473edeb894b7adc966692cf96f65'
    api_secret = u'dce0219c24f27acc'

    flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

    # see https://stuvel.eu/flickrapi-doc/3-auth.html
    # Only do this if we don't have a valid token already
    if not flickr.token_valid(perms='read') and authorize:
        # Get a request token and an url for out-of-bound
        flickr.get_request_token(oauth_callback='oob')
        authorize_url = flickr.auth_url(perms='read')

        print('No authentication token found')
        print(f'1. Please open {authorize_url} with a browser')
        print(f'1. Enter the displayed verifier code here')
        verifier = str(input())


        # Trade the request token for an access token
        flickr.get_access_token(verifier)
    return flickr




def read_file(images: map):
    file_name = 'flickr_images.csv'

    # open file for reading
    if not os.path.isfile(file_name):
        with open(file_name, 'x', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADER)

    with open(file_name, 'r', encoding='UTF8', newline='') as file:
        reader = csv.reader(file)

        for tokens in reader:
            key = tokens[0]

            if key.startswith('#'):
                continue

            if key in images:
                msg = f'ERROR: duplicated image id {key}'
                print(msg)
                raise RuntimeError(msg)

            images[key] = {CSV_HEADER[1]: tokens[1], CSV_HEADER[2]: tokens[2], CSV_HEADER[3]: tokens[3]}

    # open for writing/appending
    file = open(file_name, 'a', encoding='UTF8', newline='')

    return file




def search_images_within_switzerland(flickr, images: dict, file):
    # coordinates bounding boy switzerland according https://giswiki.hsr.ch/Bounding_Box
    min_latitude = 45.6755
    max_latitude = 47.9163
    min_longitude = 5.7349
    max_longitude = 10.6677

    # bbox=minimum_longitude, minimum_latitude, maximum_longitude, maximum_latitude.
    bounding_box = f'{min_longitude},{min_latitude},{max_longitude},{max_latitude}'

    all_photos = []

    writer = csv.writer(file)
    page = 1
    cnt = 1
    while True:
        #photos = flickr.photos.search(text='switzerland', bbox=bounding_box, extras='url_c,license', page=page)
        photos = flickr.photos.search(tags='switzerland', bbox=bounding_box, extras='url_c,license', page=page)

        total = photos['photos']['total']
        pages = photos['photos']['pages']
        photo_list = photos['photos']['photo']
        lines = []

        print(f'Page {page}/{pages}, element {cnt}/{total}')

        for elem in photo_list:

            id = elem.get('id')
            url = elem.get('url_c')
            title = elem.get('title')

            if id is None or url is None:
                print(f'ERROR incomplete data: id {id} url {url} title {title}')
                continue

            if id in images:
                print(f'Id {id} already found (title {title})')
                cnt += 1
                continue

            try:
                location = flickr.photos.geo.getLocation(photo_id=id)
                latitude = location['photo']['location']['latitude']
                longitude = location['photo']['location']['longitude']

                print(f'Photo {cnt}/{total} : id={id} title={title} longitude={longitude} latitude={latitude}')

                lines.append([id, url, longitude, latitude])
                images[id] = {CSV_HEADER[1]: url, CSV_HEADER[2]: longitude, CSV_HEADER[3]: latitude}
                cnt += 1
            except:
                print(f'ERROR photo {cnt}/{total} : id={id} title={title} no coordinates found')


        print(f'Page {page}/{pages} found {len(lines)}/{len(photo_list)} new files')

        if len(lines) > 0:
            writer.writerows(lines)
            file.flush()

        page += 1


def main():
    flickr = initialize_flickr(False)
    images = {}
    with read_file(images) as file:
        search_images_within_switzerland(flickr, images, file)


if __name__ == '__main__':
    main()
