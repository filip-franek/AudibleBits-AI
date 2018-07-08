#%%
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(2,1,1) # two rows, one column, first plot. subplot is subclass of axes
t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2*np.pi*t)
line, = ax.plot(t, s, color='blue') # creates a Line2D instance and adds it to the Axes.lines list
line, = ax.plot(t, s*2, color='red')
del ax.lines[0] # deletes the first plot
line, = ax.plot(t, s*3, color='green')

#%%
import matplotlib.pyplot as plt
fig = plt.figure(1)
ax2 = fig.add_subplot(1,1,1)
n, bins, patches = ax2.hist(np.random.randn(1000), 50)
plt.getp(fig.patch)
#%%
import matplotlib.pyplot as plt
fig = plt.figure()
plt.subplot(4,2,1)
plt.subplot(4,2,(2,4))
plt.subplot(4,2,3)
plt.subplot(4,2,(5,6))
plt.subplot(4,1,4)

#%%
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(211)
ax1.set_ylabel('volts')
ax1.set_title('a sine wave')

t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2*np.pi*t)
line, = ax1.plot(t, s, color='blue', lw=2)

# Fixing random state for reproducibility
np.random.seed(19680801)

ax2 = fig.add_axes([0.15, 0.1, 0.7, 0.3])
n, bins, patches = ax2.hist(np.random.randn(1000), 50,
                            facecolor='yellow', edgecolor='yellow')
ax2.set_xlabel('time (s)')

plt.show()
#%%
string_build = ""
for data in container:
    string_build += str(data)
#%% 
    # Python program to illustrate 
# *args 
def testify(arg1, *argv):
    print("first argument :", arg1)
    for arg in argv:
        print("Next argument through *argv :", arg)
 
testify('Hello', 'this', 'to', 'GeeksforGeeks')

#%%

def test(arg1, *args):
    print(arg1)
    [print(arg) for arg in args]
test('a','b','c')

#%%

def fun(foo,bar):
    print(foo, bar) 
#fun('a', 'b')
fun(2,7)
fun(foo=2,bar=7)
fun(bar=7,foo=2)
#%% 
def fnSum(*args):
    tmp = 0;
    for x in args:
        tmp+=x
    print(tmp)
fnSum(1,1,1,1)

#%%
def fnPrint(*args):
    for i in args: print(i)
lsVar = [0, 'a', 2.5, "bc"]
fnPrint(*lsVar)

#%%
def capital_cities(**kwargs):
    # initialize an empty list to store the result
    result = []
    for key, value in kwargs.items():
        result.append("The capital city of {} is {}".format(key,value))
    return result
        
print(capital_cities(China = "Beijing",Cairo = "Egypt",Rome = "Italy"))