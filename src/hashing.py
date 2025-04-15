#creating a basic hashmap 

class HashMap:
    def __init__(self):
        self.size=10
        self.table=[[] for _ in range(self.size)]

    def _hash(self,key):
        return hash(key) % self.size
    
    def put(self, key,value):
        idx= self._hash(key)
        for i,(k,v) in enumerate(self.table[idx]):
            if k==key:
                self.table[idx][i]=(key,value)
                return
        self.table[idx].append((key,value))

    def get(self,key):
        idx=self._hash(key)
        for (k,v) in self.table[idx]:
            if k==key:
                return v
        return None

    def remove(self,key):
        idx=self._hash(key)
        self.table[idx]=[pair for pair in self.table[idx] if pair[0] != key]


#=====================================================================
#practice: checck for duplocates in an array

#normal method:

def duplicated_arr(arr):
    return bool(len(arr)!=set(arr))

#count frequencies of elememts
def freq(arr):
    dict={}
    for num in arr:
        if num in dict:
            dict[num]+=1
        else:
            dict[num]=1
        
    return dict

#------------------------------------------------------------------
#ginf the first repeating element
def repeated_ele(nums):
    dict={}
    for num in nums:
        if num in dict:
            return (f"{num} repeats first")
        else:
            dict[num]=1

#method 2:
def first_appear(arr):
    seen=set()
    for num in arr:
        if num in seen:
            return (f"{num} repeats first")
        seen.add(num)        

#-----------------------------------------------------------------
#checck if two arrays  are equual:
def equal_array(num1,num2):
    if len(num1)!=len(num2):
        return "Different Size, Not Equal"
    
    freq1={} 
    freq2={}
    
    for num in num1:
        if num in freq1:
            freq1[num]+=1
        else:
            freq1[num]=1

    for num in num2:
        if num in freq2:
            freq2[num]+=1
        else:
            freq2[num]=1

    if freq1==freq2:
        return "Equal"
    else: 
        return "Same Size but Not Equal"

#-----------------------------------------------------

#Two sum:










