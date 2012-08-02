import commands
a=commands.getoutput("ls imagesDone/")
b=a.split('\n')
f=open('images.txt','w')
for i in range(0,len(b)):
    #f.write('images/')
    f.write(b[i])
    f.write('\n')
f.close()
