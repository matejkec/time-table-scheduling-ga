import numpy as np
import random
import copy
import math
import statistics
import pandas as pd
import matplotlib.pyplot as plt
 
# TEST DATA
groups = [[0, 'g0'], [1, 'g1'], [2, 'g2']]
professors = [[0,'p0'],[1,'p1'],[2,'p2'],[3,'p3'],[4,'p4']]
rooms = [[0,'r0',50,0],[1,'r1',50,0],[2,'r2',40,0],[3,'r3',30,0],[4,'r4',40,1],[5,'r5',50,1],[6,'r6',50,2]]
courses = [[0,'c0',2,2,30,0,0],
           [1,'c1',2,0,50,1,1],
           [2,'c2',2,0,40,2,2],
           [3,'c3',2,1,30,0,3],
           [4,'c4',3,1,30,1,4],
           [5,'c5',3,1,30,2,0],
           [6,'c6',3,0,30,0,1],
           [7,'c7',3,0,30,1,2],
           [8,'c8',4,0,30,2,3],
           [9,'c9',4,1,30,0,4],
           [10,'c10',4,1,30,1,0],
           [11,'c11',4,1,30,2,1]]


# Parameters
POPULATION_SIZE = 100
MUTATION_PROBABILITY = 0.025
CROSSOVER_PROBABILITY = 1.0
MAX_NUMBER_OF_GENERATIONS = 1000
MAX_NO_PROGRESS_GENERATIONS = 150

days = 5
daysName = ['','Pon', 'Uto', 'Sri', 'Cet', 'Pet']
working_hours = 8
workingTimeFrom = 8

coursesLen = len(courses)
roomsLen = len(rooms)
totalTimeSlot = days*working_hours
totalRoomsTimeSlot = roomsLen*totalTimeSlot

def getInitialPopulation(POPULATION_SIZE):
    population = []
    populationFitness = []
    for i in range(POPULATION_SIZE):
        randomRoomTime = random.sample(range(0, totalRoomsTimeSlot), coursesLen)
        chromosome = np.zeros((coursesLen, 3), dtype=int)
        for i in range(len(chromosome)):
            chromosome[i][0] = i
            chromosome[i][1] = randomRoomTime[i]//(totalTimeSlot)
            chromosome[i][2] = randomRoomTime[i]-(randomRoomTime[i]//(totalTimeSlot))*(totalTimeSlot)
        population.append(chromosome)
        populationFitness.append(getFitness(chromosome))       
    return (list(t) for t in zip(*sorted(zip(populationFitness, population), key=lambda x: x[0])))  

def crossPopulation(population, population2, populationFitness, populationFitness2):
    newPop1, newPop2, newPop1F, newPop2F = [], [], [], []
    for i in range(len(population)):
        if np.random.uniform(0.0, 1.0) < 0.5:
            newPop1.append(copy.copy(population[i]))
            newPop1F.append(copy.copy(populationFitness[i]))
            newPop2.append(copy.copy(population2[i]))
            newPop2F.append(copy.copy(populationFitness2[i]))
        else:
            newPop1.append(copy.copy(population2[i]))
            newPop1F.append(copy.copy(populationFitness2[i]))
            newPop2.append(copy.copy(population[i]))
            newPop2F.append(copy.copy(populationFitness[i]))
    newPop1F, newPop1 = (list(t) for t in zip(*sorted(zip(newPop1F, newPop1), key=lambda x: x[0])))
    newPop2F, newPop2 = (list(t) for t in zip(*sorted(zip(newPop2F, newPop2), key=lambda x: x[0])))        
    
    return newPop1, newPop2, newPop1F, newPop2F

def rouletteWheelFitnessSelection(choices):
    max = sum(choices)
    pick = random.uniform(0, max)
    current = 0
    for key, value in enumerate(choices):
        current += value
        if current > pick:
            parent1 = key
            parent2 = key
            break
    while(parent1 == parent2):
        pick = random.uniform(0, max)
        current = 0
        for key, value in enumerate(choices):
            current += value
            if current > pick:
                parent2 = key
                break
    return parent1, parent2
        
def rouletteWheelRankSelection(choices):
    rank = []
    for i in range(len(choices)):
        if(i == 0):
            rank.append(1)
        else:
            if (choices[i] == choices[i-1]):
                rank.append(copy.copy(rank[-1]))
            else:
                rank.append(copy.copy(rank[-1])+1)
    max = sum(rank)
    pick = random.uniform(0, max)
    current = 0
    for key, value in enumerate(rank):
        current += value
        if current > pick:
            parent1 = key
            parent2 = key
            break
    while(parent1 == parent2):
        pick = random.uniform(0, max)
        current = 0
        for key, value in enumerate(rank):
            current += value
            if current > pick:
                parent2 = key
                break
    return parent1, parent2
        
def uniformCrossover(parent1, parent2):
    child = np.zeros((coursesLen, 3), dtype=int)
    for i in range(len(parent1)):
        if np.random.uniform(0.0, 1.0) < 0.5:
            child[i] = copy.copy(parent1[i])
        else:
            child[i] = copy.copy(parent2[i])
    return child

def mutate(chromosome, MUTATION_PROBABILITY):
    for i in range(len(chromosome)):
        if np.random.uniform(0.0, 1.0 ) < MUTATION_PROBABILITY:
            if np.random.random() < 0.5:
                randomRoom = random.randint(0, roomsLen-1)
                chromosome[i][1] = randomRoom
            else:
                randomTimeSlot = random.randint(0, totalTimeSlot-1)
                chromosome[i][2] = randomTimeSlot
    return chromosome  

def getFitness(chromosome):
    softConstraints = 0
    overtime = 0
    typeConflict = 0
    roomSizeConflict = 0
    groupConflict = 0
    courseOverlap = 0
    professorConflict = 0
    
    professor_time = np.zeros((len(professors), totalTimeSlot), dtype=int)
    group_time = np.zeros((len(groups), totalTimeSlot), dtype=int)
    room_time = np.zeros((roomsLen, totalTimeSlot), dtype=int)     
       
    for i in range(len(chromosome)):
        courseTime = courses[chromosome[i][0]][2]
        startingTime = chromosome[i][2] % working_hours
        endingTime = startingTime + courseTime      
        if (endingTime > working_hours):
            overtime += (endingTime-working_hours)
            courseTime = working_hours-startingTime
        for j in range(courseTime):
            if (room_time[chromosome[i][1]][chromosome[i][2]+j] == 1):
                courseOverlap += 1
            else:
                room_time[chromosome[i][1]][chromosome[i][2]+j] += 1
            if (professor_time[courses[chromosome[i][0]][6]][chromosome[i][2]+j] == 1):
                professorConflict += 1
            else:
                professor_time[courses[chromosome[i][0]][6]][chromosome[i][2]+j] += 1
            if (group_time[courses[chromosome[i][0]][5]][chromosome[i][2]+j] == 1):
                groupConflict += 1
            else:
                group_time[courses[chromosome[i][0]][5]][chromosome[i][2]+j] += 1
        if (courses[chromosome[i][0]][3] != rooms[chromosome[i][1]][3]):
            typeConflict += 1
        if (courses[chromosome[i][0]][4] > rooms[chromosome[i][1]][2]):
            roomSizeConflict += 1                                               
        
    for i in range(len(professor_time)):
        hoursInDay = []
        for j in range(days):
            hoursInARow = 0
            hoursInDay.append(0)
            ProfessorStartTime = -1
            ProfessorEndTime = -1
            for k in range(working_hours):           
                if (professor_time[i][j*working_hours + k] > 0):
                    hoursInDay[j] += 1
                    hoursInARow += 1
                    if (hoursInARow > 4):
                        softConstraints += 1
                    if (ProfessorStartTime == -1):
                        ProfessorStartTime = k
                    ProfessorEndTime = k
                else:
                    hoursInARow = 0
            if (ProfessorEndTime - ProfessorStartTime + 1 > 8):
                softConstraints += (ProfessorEndTime - ProfessorStartTime + 1 - 8)
        average = statistics.mean(hoursInDay)
        for j in range(days):
            if(hoursInDay[j] != int(average) and hoursInDay[j] != math.ceil(average)):
                softConstraints += min(abs(int(average)-hoursInDay[j]), abs(math.ceil(average)-hoursInDay[j]))      
    
    for i in range(len(group_time)):
        hoursInDay = []
        for j in range(days):
            emptySlot = 0
            longBreaks = 0
            hoursInARow = 0
            hoursInDay.append(0)
            for k in range(working_hours):
                if (k==0 or emptySlot > 0):
                    if (group_time[i][j*working_hours + k] == 0):
                        emptySlot += 1
                    else:
                        softConstraints += emptySlot
                        emptySlot = 0                   
                if (group_time[i][j*working_hours + k] > 0):
                    hoursInDay[j] += 1
                    hoursInARow += 1
                    if (hoursInARow > 4):
                        softConstraints += 1
                    if (longBreaks > 1):
                        softConstraints += longBreaks - 1
                    longBreaks = 0
                else:
                    hoursInARow = 0
                    longBreaks += 1
        softConstraints += emptySlot
        average = statistics.mean(hoursInDay)
        for j in range(days):
            if(hoursInDay[j] != int(average) and hoursInDay[j] != math.ceil(average)):
                softConstraints += min(abs(int(average)-hoursInDay[j]), abs(math.ceil(average)-hoursInDay[j]))      
        
    hardConstraints =  overtime + typeConflict + groupConflict + professorConflict + roomSizeConflict + courseOverlap
    fitness = hardConstraints*100 + softConstraints
    return 1/(1+fitness)

def getNewGeneration(population, populationFitness):
    newGeneration = [copy.copy(population[-1])]
    newGenerationFitness = [copy.copy(populationFitness[-1])]
    while(len(newGeneration) < POPULATION_SIZE):
        parent1, parent2 = rouletteWheelFitnessSelection(populationFitness)
        if (np.random.uniform(0.0, 1.0) < CROSSOVER_PROBABILITY):
            child = uniformCrossover(population[parent1], population[parent2])
        else:
            child = population[parent1]          
        child = mutate(child, MUTATION_PROBABILITY)

        isDuplicated = False
        for j in range(len(newGeneration)):
            if (child == newGeneration[j]).all():
                isDuplicated = True
                break
        if (not isDuplicated):
            childFitness = getFitness(child)
            newGeneration.append(copy.copy(child))
            newGenerationFitness.append(copy.copy(childFitness))
    return (list(t) for t in zip(*sorted(zip(newGenerationFitness, newGeneration), key=lambda x: x[0])))

def drawTimetable(arr, columnLabels, title):
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    plt.title(title)
    df = pd.DataFrame(arr, columns=columnLabels)   
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')   
    fig.tight_layout()
    plt.show()

def drawGroupTT(groupID, chromosome):    
    arr = []
    for i in range(working_hours):
        arr.append([])
        string = []
        for j in range(days+1):
            if (j==0):
                string.append(str(workingTimeFrom+i)+":00")
            else:
                string.append("")
        arr[i] = string    
    for i in range(len(chromosome)):
        courseTime = courses[chromosome[i][0]][2]
        if (groupID == courses[chromosome[i][0]][5]):
            d = chromosome[i][2] // working_hours
            t = chromosome[i][2]-(chromosome[i][2]//(working_hours))*(working_hours)
            for j in range(courseTime):
                arr[t+j][d+1] = str(courses[chromosome[i][0]][1]) + " - " + str(rooms[chromosome[i][1]][1])    
    drawTimetable(arr, daysName[:(days+1)], "Group: " + groups[groupID][1])  
       
def drawProfessorTT(professorID, chromosome):
    arr = []
    for i in range(working_hours):
        arr.append([])
        string = []
        for j in range(days+1):
            if (j==0):
                string.append(str(workingTimeFrom+i)+":00")
            else:
                string.append("")
        arr[i] = string  
    for i in range(len(chromosome)):
        courseTime = courses[chromosome[i][0]][2]
        if (professorID == courses[chromosome[i][0]][6]):
            d = chromosome[i][2] // working_hours
            t = chromosome[i][2]-(chromosome[i][2]//(working_hours))*(working_hours)
            for j in range(courseTime):
                arr[t+j][d+1] = str(courses[chromosome[i][0]][1]) + " - " + str(rooms[chromosome[i][1]][1])     
    drawTimetable(arr, daysName[:(days+1)], "Professor: " + professors[professorID][1])

def drawRoomTT(day, chromosome):
    arr = []
    roomsName = []
    for i in range(len(rooms)+1):
        if (i==0):
            roomsName.append('')
        else:
            roomsName.append(rooms[i-1][1])                
    for i in range(working_hours):
        arr.append([])
        string = []
        for j in range(len(rooms)+1):
            if (j==0):
                string.append(str(workingTimeFrom+i)+":00")
            else:
                string.append("")
        arr[i] = string  
    for i in range(len(chromosome)):
        courseTime = courses[chromosome[i][0]][2]
        if (day == (chromosome[i][2] // working_hours)):
            t = chromosome[i][2]-(chromosome[i][2]//(working_hours))*(working_hours)
            for j in range(courseTime):
                arr[t+j][chromosome[i][1]+1] = str(courses[chromosome[i][0]][1]) + " - " + str(groups[courses[chromosome[i][0]][5]][1])        
    drawTimetable(arr, roomsName, "Day: " + daysName[day+1])   

# Algorithm     
maxFit = []
stopCondition = False
noProgressGenerations = 0
numberOfGenerations = 0
maximumAtGen = 0
populationFitness, population = getInitialPopulation(POPULATION_SIZE)
while(not stopCondition):        
    newGenerationFitness, newGeneration = getNewGeneration(population, populationFitness)
    if(populationFitness[-1] == newGenerationFitness[-1]):
        noProgressGenerations += 1
    else:
        noProgressGenerations = 0
    population = copy.copy(newGeneration)
    populationFitness = copy.copy(newGenerationFitness)
    numberOfGenerations += 1    
    
    if(numberOfGenerations >= MAX_NUMBER_OF_GENERATIONS or populationFitness[-1] == 1.0 or noProgressGenerations > MAX_NO_PROGRESS_GENERATIONS):
        stopCondition = True            
    
    maxFit.append(populationFitness[-1])
    if(populationFitness[-1] < 0.01):
        maximumAtGen += 1
    print(populationFitness[-1])
print(populationFitness[-1])
print(maximumAtGen)
print("---------")

# Draw    
#if(max(populationFitness) > 0.01):    
#    for i in range(len(groups)):
#        drawGroupTT(i, population[np.argmax(populationFitness)])
#    for i in range(len(professors)):
#        drawProfessorTT(i, population[np.argmax(populationFitness)])
#    for i in range(days):
#        drawRoomTT(i, population[np.argmax(populationFitness)])
