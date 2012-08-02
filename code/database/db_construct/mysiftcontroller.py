import os
for i in range(1,56) :
    os.system("./MySIFT imagesDone/" + `i` + ".png > vectors/" + `i` + ".txt")
    print i
