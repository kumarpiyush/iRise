import math
import operator

def length(l) :
    i=0
    val=0
    while i<128 :
        val=val+l[i]*l[i]
        i=i+1
    return math.sqrt(val)
def toNums(l) :
    ret=[]
    i=0
    while i<128 :
        ret=ret+[float(l[i])]
        i=i+1
    return ret

lines=open("out1").readlines()
vectors=[]
for line in lines :
    vec=line.split(' ')
    vectors=vectors+[toNums(vec)]

vecLenDiffs=[]
for i in range(0,len(vectors)) :
    for j in range(i+1,len(vectors)) :
        vecLenDiffs=vecLenDiffs+[length(map(operator.sub,vectors[i],vectors[j]))]
        
min=vecLenDiffs[0]
for i in range(0,len(vecLenDiffs)) :
    if vecLenDiffs[i]<min :
        min=vecLenDiffs[i]

alpha=open("out2","w")
for i in range(0,len(vecLenDiffs)) :
    vecLenDiffs[i]=vecLenDiffs[i]/min
    alpha.write(`vecLenDiffs[i]`)
    alpha.write('\n')
alpha.close()