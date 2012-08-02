import os
for i in range(1,20) :
    os.system("./MySIFT imagesDone/" + `i` + ".png > vectors/" + `i` + ".txt")
