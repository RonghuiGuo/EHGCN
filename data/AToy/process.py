newlines = []
for line in open("/home/lixunkai/RS-Code/myCode/data/AToy/train.txt"):   
    # newline = []
    # print(line)
    line = line.strip('\n')
    newline = list(map(int, line.split(" ")))
    print(newline)
    temp = filter(lambda x: x<30000,newline)
    newline = list(temp)
    print(newline)