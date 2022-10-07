# if MacOS:
# import pywraps2 as s2
# else:
import s2geometry.pywraps2 as s2
import csv

# Class to represent coordinates (lat, lng)
from typing import List


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
    def __init__(self, images: List[Image], lowerBound: Point, upperBound: Point, minPoints: int, maxPoints: int):
        for image in images:
            if not image.is_bounded(lowerBound, upperBound):
                raise ValueError('Your points are not inside the given boundary')
        
        self.__images = images
        self.__minPoints = minPoints
        self.__maxPoints = maxPoints

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
            if len(insideCellImages) <= self.__maxPoints or currentCell.level() == s2.S2CellId.kMaxLevel:
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
            header = ['# ID', 'WKT']

            with open(fileName, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(wktList)
        else:
            return wktList