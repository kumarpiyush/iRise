import commands
a=commands.getoutput("ls ../images/")
b=a.split('\n')
f=open('names','w')
for i in range(0,len(b)):
    f.write('../images/')
    f.write(b[i])
    f.write('\0')
    f.write('\n')
f.close()
