import datetime
import os.path
import csv
import re
import time

import flickrapi


# coordinates bounding boy switzerland according https://giswiki.hsr.ch/Bounding_Box
MIN_LATITUDE_SWITZERLAND = 45.6755
MAX_LATITUDE_SWITZERLAND = 47.9163
MIN_LONGITUDE_SWITZERLAND = 5.7349
MAX_LONGITUDE_SWITZERLAND = 10.6677

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


def read_excluded_file(file_name: str):
    ids = {}

    with open(file_name, 'r', encoding='UTF8', newline='') as file:
        reader = csv.reader(file)

        for tokens in reader:
            if len(tokens) == 0 or tokens[0].startswith('#'):
                continue

            key = tokens[0]
            if key in ids:
                msg = f'ERROR: duplicated image id {key}'
                print(msg)
                raise RuntimeError(msg)

            ids[key] = tokens[1]

        return ids


def read_labels_file(file_name: str):

    labels = {}
    with open(file_name, 'r', encoding='UTF8', newline='') as file:
        reader = csv.reader(file)

        for tokens in reader:

            if len(tokens) == 0 or tokens[0].startswith('#'):
                continue

            key = tokens[0]

            if key in labels:
                msg = f'ERROR: duplicated image id {key}'
                print(msg)
                raise RuntimeError(msg)

            labels[key] = tokens[1]

    return labels



def read_cvs_file(file_name: str):

    images = {}

    # open file for reading
    if not os.path.isfile(file_name):
        with open(file_name, 'x', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADER)

    with open(file_name, 'r', encoding='UTF8', newline='') as file:
        reader = csv.reader(file)

        for tokens in reader:

            if len(tokens) == 0 or tokens[0].startswith('#'):
                continue

            key = tokens[0]

            if key in images:
                msg = f'ERROR: duplicated image id {key}'
                print(msg)
                raise RuntimeError(msg)

            images[key] = {CSV_HEADER[1]: tokens[1], CSV_HEADER[2]: tokens[2], CSV_HEADER[3]: tokens[3]}

    return images




def search_images_within_switzerland(flickr, images: dict, file):

    # bbox=minimum_longitude, minimum_latitude, maximum_longitude, maximum_latitude.
    bounding_box = f'{MIN_LONGITUDE_SWITZERLAND},{MIN_LATITUDE_SWITZERLAND},{MAX_LONGITUDE_SWITZERLAND},{MAX_LATITUDE_SWITZERLAND}'

    all_photos = []
    tags = 'Suisse,Schweiz,Switzerland,Svizzera,Swiss'

    writer = csv.writer(file)

    # flickr was founded Februar 2004 by Yahoo
    start_date = datetime.date(year=2005, month=1, day=1)

    # 14.10.2022: we queried until 2013-10-28
    start_date = datetime.date(year=2012, month=0, day=27)


    end_date = datetime.date(year=2022, month=12, day=30)

    current_date = start_date
    while current_date < end_date:
        min_date = current_date
        max_date = min_date + datetime.timedelta(days=1)

        unix_time_min = time.mktime(min_date.timetuple())
        unix_time_max = time.mktime(max_date.timetuple())
        current_date = max_date
        page = 1

        print(f'Query date {current_date}')

        while True:
            try:
                photos = flickr.photos.search(min_upload_date=unix_time_min, max_upload_date=unix_time_max, tag_mode='any',
                                          tags=tags, bbox=bounding_box, extras='url_c,license', page=page)
            except Exception as e:
                print(f'ERROR call search, {page}/{pages} date {current_date}: {e}')
                continue


            total = int(photos['photos']['total'])

            if total <= 0:
                break

            pages = int(photos['photos']['pages'])
            photo_list = photos['photos']['photo']

            print(f'Page {page}/{pages} date {current_date} total {total}')

            new_cnt = 0
            for idx, elem in enumerate(photo_list):

                id = elem.get('id')
                url = elem.get('url_c')
                title = elem.get('title')

                if id is None or url is None:
                    print(f'ERROR incomplete data: id {id} url {url} title {title}')
                    continue

                if id in images:
                    print(f'Id {id} already found (title {title})')
                    continue

                try:
                    location = flickr.photos.geo.getLocation(photo_id=id)
                    latitude = location['photo']['location']['latitude']
                    longitude = location['photo']['location']['longitude']

                    print(f'   Photo {idx}/{len(photo_list)} : id={id} title={title} longitude={longitude} latitude={latitude}')

                    images[id] = {CSV_HEADER[1]: url, CSV_HEADER[2]: longitude, CSV_HEADER[3]: latitude}

                    writer.writerow([id, url, longitude, latitude])
                    file.flush()

                    new_cnt += 1
                except:
                    print(f'   ERROR photo {idx}/{len(photo_list)} : id={id} title={title} no coordinates found')

            # end for loop
            print(f'Page {page}/{pages} found {new_cnt}/{total} new files')

            page += 1
            if page >= pages:
                break




def search_images():
    file_name = 'input/flickr_images.csv'
    images = read_cvs_file(file_name)
    flickr = initialize_flickr(False)

    with open(file_name, 'a', encoding='UTF8', newline='') as file:
        search_images_within_switzerland(flickr, images, file)

def read_swiss_keywords():

    key_words = set()
    with open(f'input/swiss_keywords.txt', encoding='UTF8') as file:
        lines = file.readlines()
        for line in lines:
            tokens = line.split()
            for token in tokens:
                key_words.add(token.lower())

    return key_words

def create_dataset_from_Ids(dataset_name: str, image_ids):

    key_words = read_swiss_keywords()

    flickr = initialize_flickr(False)

    # create files
    if not os.path.isfile(f'input/{dataset_name}_excluded.csv'):
        with open(f'input/{dataset_name}_excluded.csv', 'w', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['# id', 'url'])
            file.flush()

    if not os.path.isfile(f'input/{dataset_name}.csv'):
        with open(f'input/{dataset_name}.csv', 'w', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADER)
            file.flush()

    images = read_cvs_file(f'input/{dataset_name}.csv')
    excluded = read_excluded_file(f'input/{dataset_name}_excluded.csv')

    with open(f'input/{dataset_name}.csv', 'a', encoding='UTF8', newline='') as file_valid, \
            open(f'input/{dataset_name}_excluded.csv', 'w', encoding='UTF8', newline='') as file_excluded:

        writer_valid = csv.writer(file_valid)
        writer_excluded = csv.writer(file_excluded)


        cnt = 0
        for idx, id in enumerate(image_ids):
            id = id.split('_')[0]

            if id in images or id in excluded:
                continue

            try:
                info = flickr.photos.getInfo(photo_id=id)
                photo = info['photo']

                if 'location' not in photo:
                    writer_excluded.writerow([id, 'no location tag'])
                    file_excluded.flush()
                    continue

                location = photo['location']
                if 'latitude' not in location or 'longitude' not in location:
                    writer_excluded.writerow([id, 'no location tags 2'])
                    file_excluded.flush()
                    continue

                longitude = float(location['longitude'])
                latitude = float(location['latitude'])

                # check bounding box
                if latitude < MIN_LATITUDE_SWITZERLAND or latitude > MAX_LATITUDE_SWITZERLAND or longitude < MIN_LONGITUDE_SWITZERLAND or longitude > MAX_LONGITUDE_SWITZERLAND:
                    print(f'{idx}: photo id={id} longitude {longitude} latitude {latitude} not in bounding box')
                    writer_excluded.writerow([id, 'not in bounding box'])
                    file_excluded.flush()
                    continue

                # check tags
                tags = photo['tags']
                valid_tag = False
                all_tokens = []
                for tag in tags['tag']:
                    tokens = re.split(';,: ',tag['raw'])
                    for token in tokens:
                        if token.lower() in key_words:
                            valid_tag = True
                            break
                        all_tokens.append(token)
                    if valid_tag:
                        break

                # check title
                title = ''
                if not valid_tag and photo['title'] and photo['title']['_content']:
                    title = photo['title']['_content']
                    tokens = re.split(';,: ',title)
                    for token in tokens:
                        if token.lower() in key_words:
                            valid_tag = True
                            break

                # check description
                description = ''
                if not valid_tag and photo['description'] and photo['description']['_content']:
                    description = photo['description']['_content']
                    tokens = re.split(';,: ',description)
                    for token in tokens:
                        if token.lower() in key_words:
                            valid_tag = True
                            break

                if not valid_tag:
                    print(f'{idx}: photo id={id} not valid: tags {all_tokens} title {title} description {description}')
                    writer_excluded.writerow([id, 'no valid tags'])
                    file_excluded.flush()
                    continue

                longitude = location['longitude']
                latitude = location['latitude']
                secret = photo['secret']
                server = photo['server']
                farm = photo['farm']

                # example http://farm4.staticflickr.com/3109/2824840348_3c684b1d2c.jpg
                url = f'https://farm{farm}.staticflickr.com/{server}/{id}_{secret}.jpg'
                print(f'{idx}: {cnt} using photo id={id} longitude {longitude} latitude {latitude} url={url}')

                writer_valid.writerow([id, url, longitude, latitude])
                file_valid.flush()

                cnt += 1
            except Exception as e:
                writer_excluded.writerow([id, 'exception'])
                file_excluded.flush()
                print()
                print(f'{idx} ERROR photo id={id}: {e}')

    print('-'*10)
    print(f'Found {cnt} training data pint')
    print('-'*10)






if __name__ == '__main__':
    search_images()
    #images = read_cvs_file('input/geotags_185K.csv')
    #ids = images.keys()
    #create_dataset_from_Ids('geotags_reconstructed', ids)
