/**
 * @file
 * @author Ethan Gaebel <egaebel>
 * @date   26-Feb-2022
 * @brief Modified from https://github.com/egaebel/Array-List--Cplusplus
*/

/*
The MIT License (MIT) Copyright (c) 2016 Ethan Gaebel

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef LIST_H
#define LIST_H

#include "utils/cuda_utils.cuh"

template <typename T> class List
{
  public:
    //The array that is the basis for this List.
    T *_array;
    //The current number of elements in this List.
    int _size;
    //The current capacity of the array that is backing this.
    int _capacity;

    bool _sorted;

    //~Constructors-----------------------------------------------
    __hd__ List()
    {
        this->_array = new T[20];
        this->_capacity = 20;
        this->_size = 0;

        _sorted = false;
    }

    __hd__ List(int size)
    {
        this->_array = new T[this->_size];
        this->_capacity = this->_size;
        this->_size = 0;

        _sorted = false;
    }

    __hd__ List(const List<T> &list)
    {
        this->_array = new T[list.size() * 2];
        this->_capacity = list.size() * 2;
        this->_size = list.size();

        _sorted = false;

        for (int i = 0; i < list.size(); i++) {
            this->_array[i] = *list.get(i);
        }
    }

    __hd__ ~List()
    {
        delete[] this->_array;
    }

    __hd__ void reallocate()
    {
        this->_capacity *= 2;
        T *temp = new T[this->_capacity];

        for (int i = 0; i < this->_size; i++) {
            temp[i] = this->_array[i];
        }

        delete[] this->_array;

        this->_array = temp;
    }

    __hd__ void reallocate(int newSize)
    {
        this->_capacity = newSize;
        T *temp = new T[newSize];

        for (int i = 0; i < this->_size; i++) {
            temp[i] = this->_array[i];

            delete[] this->_array;

            this->_array = temp;
        }
    }

    //~Methods---------------------------------------------------
    __hd__ void add(const T &element)
    {
        if ((this->_size - 1) == this->_capacity) {
            reallocate();
        }

        this->_array[this->_size] = element;
        this->_size++;
    }

    __hd__ void insert(const T &element, int index)
    {
        if (index >= 0 && index <= this->_size) {
            //Reallocate if necessary.
            if (index >= this->_capacity || this->_size == this->_capacity) {
                int multiplicand = (index >= this->_capacity) ? index : this->_size;
                reallocate(multiplicand * 2);
            }

            //Shift elements to the right.
            for (int i = this->_size; i > index; i--) {
                this->_array[i] = this->_array[i - 1];
            }

            this->_array[index] = element;
            this->_size++;

            _sorted = false;
        }
    }

    __hd__ void extend(const List<T> &list)
    {
        for (int i = 0; i < list.size(); i++) {
            add(list.get(i));
        }
    }

    __hd__ void extend_arr(T *array, int size)
    {
        for (int i = 0; i < size; i++) {
            add(array[i]);
        }
    }

    __hd__ bool remove(const T &element)
    {
        int index = binarySearch(element);

        if (index >= 0 && index < this->_size) {
            remove(index);

            return true;
        } else {
            return false;
        }
    }

    __hd__ const T *removeAt(int index)
    {
        T *removed = NULL;

        if (index < this->_size && index >= 0) {
            for (int i = index; i < this->_size; i++) {
                this->_array[i] = this->_array[i + 1];
            }

            removed = this->_array + index;
            this->_size--;
        }

        return removed;
    }

    __hd__ void clear()
    {
        delete[] this->_array;
        this->_array = new T[this->_capacity];

        _sorted = false;
        this->_size = 0;
    }

    __hd__ bool swap(int index1, int index2)
    {
        if (index1 >= 0 && index2 >= 0 && index1 < this->_size && index2 < this->_size) {
            T temp = this->_array[index1];
            this->_array[index1] = this->_array[index2];
            this->_array[index2] = temp;

            _sorted = false;

            return true;
        }

        return false;
    }

    __hd__ T *get(int index) const
    {
        return (index >= 0 && index < this->_size) ? (this->_array + index) : NULL;
    }

    __hd__ T back()
    {
        return *get(_size - 1);
    }

    __hd__ bool set(const T &newValue, int index)
    {
        if (index >= 0 && index < this->_size) {
            this->_array[index] = newValue;
            return true;
        }

        return false;
    }

    //~Extra Basic Functions-----------------------------------
    __hd__ const List<T> &subList(int index1, int index2)
    {
        if (index1 >= 0 && index1 < this->_size && index2 >= 0 && index2 <= this->_size &&
            index1 < index2) {
            List<T> returnList = new List<T>(index2 - index1);

            //Loop through all elements, so there isn't just a reference
            //copy performed.
            for (int i = index1; i < index2; i++) {
                returnList.add(this->get(i));
            }

            return returnList;
        }

        return NULL;
    }

    __hd__ T *getArray()
    {
        return this->_array;
    }

    __hd__ T *getSubArray(int index1, int index2) const
    {
        if (index1 >= 0 && index1 < this->_size && index2 >= 0 && index2 <= this->_size &&
            index1 < index2) {
            T *returnList = new T[index2 - index1];

            //Loop through all elements, so there isn't just a reference
            //copy performed.
            for (int i = index1; i < index2; i++) {
                returnList[i] = this->get(i);
            }

            return returnList;
        }

        return NULL;
    }

    //~Binary Search-------------------------------------------
    __hd__ int find(const T &element)
    {
        if (_sorted) {
            return binarySearch(element, 0, this->_size - 1);
        } else {
            for (int i = 0; i < this->_size; i++) {
                if (this->_array[i] == element) {
                    return i;
                }
            }
            return -1;
        }
    }

    __hd__ int binarySearch(const T &element, int index1, int index2) const
    {
        if (index1 < index2) {
            int pivot = (index1 + index2) / 2;

            if (element < this->_array[pivot]) {
                return binarySearch(element, index1, pivot - 1);
            } else if (element > this->_array[pivot]) {
                return binarySearch(element, pivot + 1, index2);
            } else {
                return pivot;
            }
        } else if (index1 == index2) {
            return index1;
        } else {
            return -1;
        }
    }

    //~Information Getting methods
    __hd__ int size() const
    {
        return this->_size;
    }

    __hd__ int capacity()
    {
        return this->_capacity;
    }
};

#endif
