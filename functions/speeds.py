
import math 

def distanciasPuntos(vectorX, vectorY, frameRate):
    vx = vectorX
    vy = vectorY
    
    distXY = []
    velocidades = []
    
    lvx = len(vx)
    lvy = len(vy)
    for i in range(lvx):
        if i <= lvx-2:
            difX = vx[i+1] - vx[i]
            dify = vy[i+1] - vy[i]
            diferencia =((pow(difX, 2)) + (pow(dify, 2)))
            distancia = math.sqrt(diferencia)
            distXY.append(distancia) 
    
    for j in distXY:
        vel = (j/frameRate)
        velocidades.append(vel)
        
    
    return velocidades

def speeds_Dict(dictPuntos, frameRate):
    diccionarioXY = {}

    for i in range(len(dictPuntos)):
        x = []
        y = []
        xy = []
        for j in range(len(dictPuntos[i])):
            x.append(dictPuntos[i][j][0])
            y.append(dictPuntos[i][j][1])
        xy.append(x)
        xy.append(y)
        diccionarioXY.update({i:xy})

    diccionarioVeloc = {}
    
    for j in range(len(diccionarioXY)):
        vv = distanciasPuntos(diccionarioXY[j][0], diccionarioXY[j][1], frameRate)
        diccionarioVeloc.update({j: vv})
    return diccionarioVeloc
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        