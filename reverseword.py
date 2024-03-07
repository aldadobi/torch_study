def reversewords(s : str) -> str:
      new_s = ' '.join(s.split(' '))[::-1].strip(' ')

      ret = []

      for i, c in enumerate(new_s):
            if i < len(new_s) - 1 and \
            new_s[i] == new_s[i+1] == ' ':
                  i += 1
            else:
                  ret.append(c)
                  i += 1

      return ''.join(ret)

a = " hi   there   "

#ans = reversewords(a)

#print(ans)

new_a = ' '.join(a.split(' ')[::-1]).strip(' ')
new_b = ' '.join(a.split(' ')[::-1]).strip(' ')
ret = []

for i, c in enumerate(new_a):
      if i < len(new_a) - 1 and \
      new_a[i] == new_a[i+1] == ' ':
            i += 1
            print(c)
      else:
            ret.append(c)
            print(c)

print(a)
print(new_a)
print(''.join(ret))
'''
x = ['a', 'b', 'c']
y = ''.join(x)
z = '_'.join(x)
print(y)
print(z)



y_1 = a.split(' ')
x_1 = ' '.join(y_1)
print(x_1)
print(y_1)
   
'''