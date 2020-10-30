# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 16:40:26 2020

@author: angel
"""
# skrypt do zmiany nazwy w folderze 'no' -> N + nr i 'yes' -> Y + nr

# necessary library:
import os

# zmiana nazwy na NO:______________________________________________________
pathh = r"C:\Users\angel\OneDrive\Pulpit\sieci\Projekt\INPUT\no" 
filenames = os.listdir(pathh)
# print(filenames)

# create a list from 0 to amount of filenames - 1
# I want to create an iterator which iter step by step over files 
order = [x for x in range(len(filenames))]
iterator = iter(order)

for filename in filenames:
    try:   # ponizej kluczowa linijka
        os.rename(os.path.join(pathh, filename), # <- old name, below new name
                  os.path.join(pathh, "N" + str(next(iterator)) + ".jpg"))
    except:
        pass

# zmiana nazwy YES:________________________________________________________
pathYES = r"C:\Users\angel\OneDrive\Pulpit\sieci\Projekt\INPUT\yes" 
filenames = os.listdir(pathYES)
# print(filenames)

# create a list from 0 to amount of filenames - 1
# I want to create an iterator which iter step by step over files 
order = [x for x in range(len(filenames))]
iterator = iter(order)

for filename in filenames:
    try:   # ponizej kluczowa linijka
        os.rename(os.path.join(pathYES, filename), # <- old name, below new name
                  os.path.join(pathYES, "Y" + str(next(iterator)) + ".jpg"))
    except:
        pass    