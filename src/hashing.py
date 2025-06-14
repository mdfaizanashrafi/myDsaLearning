
"""

# 🧠 Hashing in Programming: Complete Guide

---

## 📌 1. Introduction to Hashing

**Hashing** is a **data processing technique** used to **map data of arbitrary size to data of fixed size**, typically through a function called a **hash function**. It enables **fast data lookup, insertion, and deletion**, especially in data structures like **hash tables**.

> 🔍 Think of hashing like assigning unique ID tags to luggage at an airport. No matter the bag’s size, each is tagged with a short, unique identifier for fast retrieval.

---

### ✅ Key Characteristics:

* Uses a **hash function** to convert input (key) into an **index**.
* Efficient for **search**, **insert**, and **delete** operations: **O(1)** average time complexity.
* Handles **collisions**, which occur when two keys map to the same index.

---

## 🧬 2. How Hashing Works

### Step-by-step:

1. Input (key) is passed to a **hash function**.
2. The hash function returns an **index** (hash code).
3. This index is used to **store or retrieve** the value in a hash table.
4. In case multiple values map to the same index, a **collision resolution** strategy is applied.

---

## 🔢 3. Types of Hashing

| Type                       | Description                                                              |
| -------------------------- | ------------------------------------------------------------------------ |
| **Direct Hashing**         | Key itself is the address (used when the key is within a small range).   |
| **Modular Hashing**        | `index = key % table_size`; most commonly used simple hash function.     |
| **Multiplicative Hashing** | Uses multiplication and fractional part extraction for index generation. |
| **Universal Hashing**      | Randomized method to reduce collisions in worst cases.                   |
| **Cryptographic Hashing**  | One-way functions used for security (e.g., SHA-256, MD5, bcrypt).        |

---

## 💥 4. Collision Resolution Techniques

| Technique             | Description                                                     |
| --------------------- | --------------------------------------------------------------- |
| **Chaining**          | Store collided keys in a linked list or list at the same index. |
| **Open Addressing**   | Probe next empty slot (linear, quadratic, or double hashing).   |
| **Linear Probing**    | Check next slots sequentially.                                  |
| **Quadratic Probing** | Skip slots quadratically until an empty one is found.           |
| **Double Hashing**    | Use a second hash function to determine step size for probing.  |

---

## 🌍 5. Real-World Use Cases of Hashing

| Use Case                     | Description                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------- |
| **Hash Maps / Hash Tables**  | Used in dictionaries, caches, sets, and object stores for constant-time access. |
| **Cryptography**             | Secure data transmission, password hashing, digital signatures.                 |
| **Database Indexing**        | Rapid lookups of records using hashed keys.                                     |
| **Compiler Symbol Tables**   | Fast variable/function lookup during parsing and compilation.                   |
| **Blockchain & Bitcoin**     | SHA-256 used in proof-of-work and block identification.                         |
| **Load Balancing**           | Hashing user IDs or IPs to route requests across servers.                       |
| **File/Media Deduplication** | Identify duplicate files using hash signatures.                                 |

---

## ⚡ 6. Advantages

* **Fast lookup**: O(1) on average for search, insert, and delete.
* **Efficient memory use** with good hash functions and load balancing.
* **Flexible key types**: Can hash strings, numbers, etc.

---

## ⚠️ 7. Limitations

* **Collisions** can degrade performance.
* Poor hash functions lead to **clustering**.
* Difficult to sort data stored in hash tables.
* Memory overhead due to open slots or chaining.

---

## 🔚 8. Conclusion

Hashing is a **core computer science concept** powering everything from **password managers** and **blockchains** to
 **dictionaries** in programming languages. Understanding its principles and pitfalls is crucial for designing efficient 
 algorithms and systems.

"""

#====================================================================
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


#==============================================================












