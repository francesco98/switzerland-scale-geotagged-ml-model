import os.path
import webbrowser

import flickrapi


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
    file_name = 'flickr_images.txt'

    # open file for reading
    if not os.path.isfile(file_name):
        file = open(file_name, 'x')
        file.writelines(['# image-id:url,longitude,latitude\n'])
        file.close()

    file = open(file_name, 'r')

    Lines = file.readlines()
    for line in Lines:
        line = line.strip()
        if len(line) < 1 or line.startswith('#'):
            continue
        tokens = line.split(sep=',')
        key = tokens[0]

        if key in images:
            msg = f'ERROR: duplicated image id {key}'
            print(msg)
            raise RuntimeError(msg)

        images[key] = {'url': tokens[1], 'long': tokens[2], 'lat': tokens[3]}

    file.close()


    # open for writting/appendimng
    file = open(file_name, 'a')

    return file




def search_images_within_switzerland(images: dict, file):
    # coordinates bounding boy switzerland according https://giswiki.hsr.ch/Bounding_Box
    min_latitude = 45.6755
    max_latitude = 47.9163
    min_longitude = 5.7349
    max_longitude = 10.6677

    # bbox=minimum_longitude, minimum_latitude, maximum_longitude, maximum_latitude.
    bounding_box = f'{min_longitude},{min_latitude},{max_longitude},{max_latitude}'

    all_photos = []

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

            id = elem['id']
            url = elem['url_c']
            title = elem['title']

            if id in images:
                print(f'Id {id} already found (title {title})')
                cnt += 1
                continue

            try:
                location = flickr.photos.geo.getLocation(photo_id=id)
                latitude = location['photo']['location']['latitude']
                longitude = location['photo']['location']['longitude']

                print(f'Photo {cnt}/{total} : id={id} title={title} longitude={longitude} latitude={latitude}')

                lines.append(f'{id},{url},{longitude},{latitude}\n')
                images[id] = {'url': url, 'long': longitude, 'lat': latitude}
                cnt += 1
            except:
                print(f'ERROR photo {cnt}/{total} : id={id} title={title} no coordinates found')


        print(f'Page {page}/{pages} found {len(lines)}/{len(photo_list)} new files')
        file.writelines(lines)
        file.flush()
        page += 1


flickr = initialize_flickr(False)


images = {}

file = read_file(images)
search_images_within_switzerland(images, file)


