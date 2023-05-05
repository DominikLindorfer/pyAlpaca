def binary_search(arr, target): 
    left = 0
    right = len(arr)-1
  
    while left<=right: 
  
        mid = (left+right)//2
  
        # Check if the element is present at the middle itself 
        if arr[mid] == target: 
            return mid 
  
        # If the element is smaller than mid, it can only  
        # be present in left subarray 
        elif arr[mid] > target: 
            right = mid - 1 
  
        # Else the element can only be present in right  
        # subarray 
        else: 
            left = mid + 1
    return -1


# Driver code 
arr = [3, 7, 8, 12] 
target = 9
print("Index of the element is", binary_search(arr, target)) # prints 2 as index of 9 is 2nd element in the array. 


import numpy as np

def generate_random_array(x, y):
    return [np.random.randint(1,100000) for i in range(x,y)]

random_array = generate_random_array(10, 20)

import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a*x**2 + b*x + c

x_data = np.array([1, 2, 3, 4, 5])
y_data = func(x_data, 2, 3, 5)

popt, pcov = curve_fit(func, x_data, y_data)

print("Best-fit parameters:", popt)
print("Parameter uncertainties:", pcov)


def delete_threes(arr):
    for i in range(len(arr)):
        if arr[i] == 3:
            arr.pop(i)
            i -= 1
    return arr

delete_threes([1, 2, 3, 4, 3, 4, 3, 4])

def delete_number(arr, num):
    if num in arr:
        arr.remove(num)
    return arr

arr = [1, 2, 3, 4, 5]
num = 3

print(delete_number(arr, num)) # Output: [1, 2, 4, 5]


def binary_search(arr, x):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1

    return -1  # not found

arr = [1, 2, 3, 4, 5]
x = 3

result = binary_search(arr, x)
if result != -1:
    print("Element is at index", result)
else:
    print("Element is not in array")


def insertion_sort(list):
    for i in range(1, len(list)):
        key = list[i]
        j = i-1
        while j >= 0 and list[j] > key:
            list[j+1] = list[j]
            j -= 1
        list[j+1] = key
    return list

if __name__ == '__main__':
    list = [7, 3, 1, 9, 0]
    print(insertion_sort(list)) # Output: [0, 1, 3, 7, 9]


Describe quantization of datatypes in machine learning.

### Quantization of data

Quantization is the process of reducing the amount of data needed to represent a signal or image. It involves mapping the original data onto a lower-dimensional space while retaining the essential information. In machine learning, this can be used to reduce the amount of data needed for training, or to reduce the complexity of the model.

### Quantization of datatypes

In machine learning, datatypes are often quantized to reduce the amount of data needed for training, or to reduce the complexity of the model. For example, images can be represented using fewer colors or less resolution, and audio can be represented using fewer frequencies or less time resolution. Additionally, integers can be represented using fewer bits, and floats can be represented using fewer decimal places.

### Benefits of quantization

Quantization can reduce the amount of data needed for training, making it easier to store and process data. Additionally, it can reduce the complexity of models, making them faster and more efficient. Finally, it can also reduce the amount of computation needed, making it easier to deploy models on limited hardware systems. 

### Drawbacks of quantization

The main drawback of quantization is the loss of information. By reducing the amount of data, we are also reducing the amount of information that can be stored or processed. This can lead to reduced accuracy, especially in tasks such as image or speech recognition. Additionally, some models may not be able to handle the reduced data, and may require more data for training. Finally, there may be some cases where the reduced accuracy is not acceptable, and the full amount of data may be needed. 

## Examples of quantization

* Image quantization: Reducing the number of colors or resolution in an image can reduce the amount of data needed for storage and processing.
* Audio quantization: Reducing the number of frequencies or time resolution in an audio signal can reduce the amount of data needed for storage and processing.
* Integer quantization: Reducing the number of bits used to store an integer can reduce the amount of data needed for storage and processing.
* Float quantization: Reducing the number of decimal places used to store a float can reduce the amount of data needed for storage and processing.
* Model quantization: Reducing the complexity of a model can reduce the amount of data needed for training, as well as the amount of computation and storage needed.
* Data quantization: Reducing the amount of data available can reduce the complexity of a model, as well as the amount of computation and storage needed. This can be done by randomly selecting a subset of data points, or by using techniques such as feature hashing.

## See also

* Data compression
* Dimension reduction
* Discretization
* Encoding
* Decoding
* Information loss
* Model complexity
* Model reduction
* Quantization error
* Reductionism
* Simplification
* Sparse data
* Subset selection
* Vectorization
* Wavelet transform
* Zipf's law