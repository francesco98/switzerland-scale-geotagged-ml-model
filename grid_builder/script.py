# RUN USING DYLD_LIBRARY_PATH=/usr/local/lib

from os import read
import pywraps2 as s2
import csv
import json

# Model class

class levelWkt:
    def __init__(self, level: int, wkt: str):
        self.level = level
        self.wkt = wkt
    
    def toArray(self):
        return [self.level, self.wkt]

# Utility functions

def getPoint(s2Point: s2.S2Point):
    latLng = s2.S2LatLng(s2Point)
    return '{:.15f}'.format(latLng.lng().degrees()) + " " + '{:.15f}'.format(latLng.lat().degrees())

def getPolygon(s2CellId: s2.S2CellId):
    s2Cell = s2.S2Cell(s2CellId)
    points = ""
    for i in range(5): # We need to add the first point also as last point to close the polygon
        if i < 4:
            endChar = ","
        else:
            endChar = ""
        points += getPoint(s2Cell.GetVertex(i%4)) + endChar

    return "POLYGON ((" + points + "))"

def generateCsv(name, wkt: list[levelWkt]):
    header = ['Level', 'WKT']
    data = [row.toArray() for row in wkt]

    with open(name + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def readJson(filename: str):
    with open(filename) as f:
        data = json.load(f)
    
    return [s2.S2CellId(s2.S2LatLng.FromDegrees(row['latitude'], row['longitude']).Normalized()) for row in data]

def splitPointsByCell(points: list[s2.S2CellId], cell: s2.S2CellId):
    outsideCellPoints = []
    insideCellPoints = []
    for point in points:
        if cell.contains(point):
            insideCellPoints.append(point)
        else:
            outsideCellPoints.append(point)
    
    return [insideCellPoints, outsideCellPoints]

def savePoints(points: list[s2.S2CellId]):
    k = 0
    header = ['ID', 'Longitude', 'Latitude']
    data = []
    for point in points:
        s2LatLon = point.ToLatLng()
        lng = '{:.15f}'.format(s2LatLon.lng().degrees())
        lat = '{:.15f}'.format(s2LatLon.lat().degrees())

        data.append([k, lng, lat])
        k+=1
        if k > 1000:
            break

    with open('points.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def createGrid(points: list[s2.S2CellId]):
    minPoints = 100 # Minimum number of points contained by a cell
    maxPoints = 1000 # Maximum number of points contained by a cell

    selectedCells = [] # Selected cell to represent a class

    currentCell = s2.S2CellId.Begin(0) # Starting point (face 0 - level 0)
    hasNewCells = True
   
    nextLevelCells = [] # Queue of cells to be analyzed (used by the algorithm)

    while hasNewCells:
        
        # If we are at level 0, we have to analyze 6 cells (6 faces)
        if currentCell.is_face():
            limitChildren = 6
        # If we have at level >= 1, each cell is splitted in 4 smaller cells
        else: 
            limitChildren = 4

        for _ in range(limitChildren):
            # print(f"Level: {currentCell.level()} - Face: {currentCell.face()} - #classes: {len(selectedCells)} - #points: {len(points)}")

            insideCellPoints, outsideCellPoints = splitPointsByCell(points, currentCell)

            # Ignoring cells not having the minimum points
            if len(insideCellPoints) > minPoints:
                # If the cell contains at most the maximum number of points (or it is a leaf), than it is a class
                if len(insideCellPoints) <= maxPoints or currentCell.is_leaf():
                    selectedCells.append(currentCell)
                    points = outsideCellPoints
                # Otherwise, add it to the queue to split it later
                else:
                    nextLevelCells.append(currentCell)

            currentCell = currentCell.next()
            
        print(f"--> #classes: {len(selectedCells)} - #remaining_points: {len(points)}")
       
        # If we still have cells to split, let's pop children of the first element
        if len(nextLevelCells) > 0:
            currentCell = nextLevelCells.pop(0).child_begin()
        else:
            hasNewCells = False
     
    return [levelWkt(index, getPolygon(cellId)) for index, cellId in enumerate(selectedCells)]

# Main function

def main():
    points = readJson("geotagsCH.json")
    data = createGrid(points)
    generateCsv("grid-wws-ch", data)
    
main()