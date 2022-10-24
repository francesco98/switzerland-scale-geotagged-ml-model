import socket

ENV_VMWARE_ADNLT903 = 'adnlt903-vm1'
ENV_ADNWSRTX01 = 'adnwsrtx01'


BASE_DIRS = {}
DATA_DIRS = {}


# adnwsrtx01
BASE_DIRS[ENV_ADNWSRTX01] = '/home/test-dev/projects/adncuba-geolocation-classifier/grid_builder'
DATA_DIRS[ENV_ADNWSRTX01] = '/mnt/store/geolocation_classifier/datadir'

# hacke vmware adnlt903
BASE_DIRS[ENV_VMWARE_ADNLT903] = '/home/hacke/projects/adncuba-geolocation-classifier/grid_builder'
DATA_DIRS[ENV_VMWARE_ADNLT903] = '/home/hacke/projects/data/geolocation_classifier'


def get_env():
    hostname = socket.gethostname()

    if hostname == ENV_VMWARE_ADNLT903:
        return ENV_VMWARE_ADNLT903

    if hostname == ENV_ADNWSRTX01:
        return ENV_ADNWSRTX01

    raise RuntimeError(f'Please add your hostname and your locations to this file')


def get_data_dir():
    return DATA_DIRS[get_env()]


def get_base_dir():
    return BASE_DIRS[get_env()]
