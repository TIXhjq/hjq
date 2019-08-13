a={}

a[1]=2
a[2]=3

b={}

b[4]=a

print(b)

f=open('test.txt','w')
f.write(str(b))
f.close()

f=open('test.txt','r')
a=f.read()
dict_name=eval(a)
f.close()
print(dict_name)

