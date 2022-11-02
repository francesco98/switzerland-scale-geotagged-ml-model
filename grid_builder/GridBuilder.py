# if MacOS:
# import pywraps2 as s2
# else:
import s2geometry.pywraps2 as s2
import csv

# Class to represent coordinates (lat, lng)
from typing import List

from grid_builder.utility import read_cvs_file, read_validated_file, read_excluded_file

class Point:
    def __init__(self, lat_degrees: float, lng_degrees: float):
        self.latLng = s2.S2LatLng.FromDegrees(lat_degrees, lng_degrees).Normalized()
        self.cellId = s2.S2CellId(self.latLng)

        self.lat = lat_degrees
        self.lng = lng_degrees

    def is_lower_than(self, other: 'Point') -> bool:
        return self.lat <= other.lat and self.lng <= other.lng

    def is_greater_than(self, other: 'Point') -> bool:
        return self.lat >= other.lat and self.lng >= other.lng
    
    def __str__(self) -> str:
        return '{:.15f}'.format(self.lng) + " " + '{:.15f}'.format(self.lat)

# Class to represent a single image (id, url, lat, lng)
class Image:
    def __init__(self, id: int, url: str, coords: Point):
        self.id = id
        self.url = url
        self.coords = coords

        self.label = None

    def is_bounded(self, lowerBound: Point, upperBound: Point) -> bool:
        return self.coords.is_greater_than(lowerBound) and self.coords.is_lower_than(upperBound)

# Class to build the grid
class Grid:
    # Coordinates bounding boy switzerland according https://giswiki.hsr.ch/Bounding_Box
    LOWER_BOUND_SWITZERLAND = Point(45.6755, 5.7349)
    UPPER_BOUND_SWITZERLAND = Point(47.9163, 10.6677)

    def __init__(self, images: List[Image], dataset_name: str, lowerBound: Point, upperBound: Point, minPoints: int, maxPoints: int,maxLevel: int):
        for image in images:
            if not image.is_bounded(lowerBound, upperBound):
                raise ValueError('Your points are not inside the given boundary')

        if maxLevel < 1 or maxLevel > s2.S2CellId.kMaxLevel:
            raise ValueError(f'Illegal maxLevel {maxLevel}')

        self.__images = images
        self.__minPoints = minPoints
        self.__maxPoints = maxPoints
        self.__maxLevel = maxLevel
        self.__dataset_name = dataset_name

        self.__gridArray = None
    
    @property
    def gridArray(self) -> List[s2.S2CellId]:
        if self.__gridArray == None:
            raise ValueError("You have to build the grid before retrieving it")

        return self.__gridArray

    def __splitImagesByCell(self, cell: s2.S2CellId):
        outsideCellImages = []
        insideCellImages = []
        for image in self.__images:
            if cell.contains(image.coords.cellId):
                insideCellImages.append(image)
            else:
                outsideCellImages.append(image)
        
        return [insideCellImages, outsideCellImages]

    def __assignLabel(self, images: List[Image], label):
        for image in images:
            image.label = label

    def __processCurrentCell(self, currentCell: s2.S2CellId, nextCellsQueue: List[s2.S2CellId]):
        insideCellImages, outsideCellImages = self.__splitImagesByCell(currentCell)

        # Ignoring cells not having the minimum points
        if len(insideCellImages) > self.__minPoints:
            # If the cell contains at most the maximum number of points (or it is a leaf), than it is a class
            if len(insideCellImages) <= self.__maxPoints or currentCell.level() == self.__maxLevel:
                self.__gridArray.append(currentCell)
                self.__assignLabel(insideCellImages, len(self.__gridArray)-1)

                self.__images = outsideCellImages
            # Otherwise, add it to the queue to split it later
            else:
                nextCellsQueue.append(currentCell)
    


    def buildGrid(self):
        if self.__gridArray != None:
            raise ValueError("You cannot call the build method twice")
        
        self.__gridArray = []

        currentCell = s2.S2CellId.Begin(0) # Starting point (face 0 - level 0)
        hasNewCells = True
    
        nextCellsQueue = [] # Queue of cells to be analyzed (used by the algorithm)

        while hasNewCells:
            if currentCell.is_face(): # If we are at level 0, we have to analyze 6 cells (6 faces)
                limitChildren = 6
            else: # If we have at level >= 1, each cell is splitted in 4 smaller cells
                limitChildren = 4

            for _ in range(limitChildren):
                self.__processCurrentCell(currentCell, nextCellsQueue)
                currentCell = currentCell.next()
                
            print(f"--> #Classes: {len(self.__gridArray)} - #Remaining Images: {len(self.__images)}")
        
            # If we still have cells to split, let's pop children of the first element
            if len(nextCellsQueue) > 0:
                currentCell = nextCellsQueue.pop(0).child_begin()
            else:
                hasNewCells = False

    def toWkt(self, fileName=None):
        wktList = []
        for index, cellId in enumerate(self.__gridArray):
            latLngList = [s2.S2LatLng(s2.S2Cell(cellId).GetVertex(i%4)) for i in range(5)]
            points = [Point(latLng.lat().degrees(), latLng.lng().degrees()) for latLng in latLngList]
            
            wktList.append([index, "POLYGON ((" + ",".join([str(point) for point in points]) + "))"]) 

        if fileName:
            header = ['# Label', 'WKT']

            with open(fileName, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f'# Grid created from dataset {self.__dataset_name}'])
                writer.writerow(header)
                writer.writerows(wktList)
        else:
            return wktList


    def toCellIds(self,  fileName: str):
        labels = [[index, cellId.id()] for index, cellId in enumerate(self.__gridArray)]

        header = ['# Label', 'S2 cellId']

        with open(fileName, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'# Grid created from dataset {self.__dataset_name}'])
            writer.writerow(header)
            writer.writerows(labels)



def convert_images_for_grid(dataset_name: str) -> List[Image]:

    images = read_cvs_file(f'input/{dataset_name}.csv')
    excluded = read_excluded_file(f'input/{dataset_name}_excluded.csv')
    validated = read_validated_file(f'input/{dataset_name}_validated.csv')

    data_points = []

    for id in images:

        if id in excluded:
            continue

        if id not in validated:
            continue

        elem = images[id]
        url = elem['url']
        longitude = float(elem['longitude'])
        latitude = float(elem['latitude'])

        point = Point(latitude, longitude)
        data_points.append(Image(id, url, point))

    return data_points


if __name__ == '__main__':

    polygons_grid_file = 'output/grid_polygons.csv'
    cellId_grid_file = 'output/grid_cellIds.csv'


    MIN_POINTS = 0
    MAX_POINTS = 5000
    MAX_LEVEL = 12

    dataset_name = 'flickr_images'

    data_points = convert_images_for_grid(dataset_name)

    grid = Grid(data_points, dataset_name, Grid.LOWER_BOUND_SWITZERLAND, Grid.UPPER_BOUND_SWITZERLAND, MIN_POINTS, MAX_POINTS, MAX_LEVEL)
    grid.buildGrid()
    grid.toWkt(polygons_grid_file)
    grid.toCellIds(cellId_grid_file)
