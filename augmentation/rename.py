from os import rename, listdir

fnames = listdir('C:\Users\Alon Melamud\Desktop\ready_to_train\train')
c=0
n=1104
for fname in fnames:
    rename(str('C:\Users\Alon Melamud\Desktop\ready_to_train\train'+str(n)+'.BMP'), 'C:\Users\Alon Melamud\Desktop\ready_to_train\train'+str(c)+'.BMP')
    rename(str('C:\Users\Alon Melamud\Desktop\ready_to_train\train'+str(n)+'.xml'), 'C:\Users\Alon Melamud\Desktop\ready_to_train\train'+str(c)+'.xml')
    c+=1
    n+=1
