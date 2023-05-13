from bs4 import BeautifulSoup
import re
import numpy as np
from datetime import datetime
import xml.etree.ElementTree as ET

# tree = ET.parse(r'C:/Users/user/Downloads/Approximate_for_phase_wm/Step2.xml')
# root = tree.getroot()
# list_value=[]
# for neighbor in root.iter('value'):
#     list_value.append(neighbor.text)
# print(len(list_value))


def read_xml_file(xml_file, element):
    """
    Parse the xml file to xml.etree.cElementTree
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    number_of_element = len(root.findall(element))
    return '{:,.0f}'.format(number_of_element)
# context = ET.iterparse('C:/Users/user/Downloads/Approximate_for_phase_wm/Step1.xml', events=("start", "end"))


def binarize(num):
    str_bin = bin(num)[2:]
    return np.array(list((10 - len(str_bin)) * '0' + str_bin), dtype=int)

# print(arr)


with open('C:/Users/user/Downloads/Approximate_for_phase_wm/Step1.xml', 'r') as f:
    data = f.read()

def binary_repr_ar(A, W):
    tmp1 = A[:, None]
    tmp2 = (1 << np.arange(W - 1, -1, -1))
    tmp = tmp1 & tmp2
    p = ((tmp != 0).view('u1'))
    return p.astype('S1').view('S' + str(W)).ravel()


def equals_func(x, id):
    for i in range(10):
        if x.GetID()[i] != id[i]:
            return False
    return True


def get_max_length(lst):
    return len(max(lst, key=len))


def Getting(variables_, ):
    shape = variables_[0].shape
    i_ = 0
    bit_var = []

    start = datetime.now()
    # преобразуем переменные в массив массивов - в битовый вид
    for i in range(len(variables_)):
        variables_[i] = variables_[i].flatten()
        b = np.array(list(map(binarize, variables_[i])))
        bit_var.append(b)

    id = np.full(b.shape, int(-1))

    while i_ < 10:
        id[:, i_] = 0
        for j in range(len(variables_)):
            id[:, i_] += bit_var[j][:, i_] * np.power(2, j)
        i_ += 1

    # for i in range(10):
    #     search_id[i] = id[i]
    #     flag = search_id in list_id
    #
    #     if flag:
    #         break

    #

    compar = [(list_id.index(list(id[i,:]))) for i in range(id.shape[0])]
    val_lst = np.array([data[dict_list[i]] for i in compar])
    enter_matr = np.reshape(val_lst, shape)

    # ind = list_id.index(id)1
    # value = data[ind]
    # print("Getting ends",value)
    print("time of method", datetime.now() - start)
    return enter_matr


# Passing the stored data inside
# the beautifulsoup parser, storing
# the returned object
strt= datetime.now()
Bs_data = BeautifulSoup(data, "lxml")
print("BSSoup",datetime.now()-strt)
# Finding all instances of tag
# `unique`
b_unique = Bs_data.find_all('value')
b_id = Bs_data.find_all('id')
list_id =([x for x in Bs_data.find_all('id')])
print("min pars",datetime.now()-strt)
for i in range(len(list_id)):
    tmp_list = []
    cu = list(list_id[i])

    for j in range(len(cu)):

        if j % 2 == 1:
            if str(cu[j])[5:6] == "-":
                tmp_list.append(int(str(cu[j])[5:7]))
            else:
                tmp_list.append(int(str(cu[j])[5:6]))
    list_id[i] = tmp_list
print("to numpy",datetime.now()-strt)

dict_list = [i for i in range(0, len(list_id))]
print("create dict",datetime.now()-strt)

need_digit = [0, 1, 2, 3]
for i in range(len(list_id)):
    if -1 in list_id[i]:
        ind = list_id[i].index(-1)
        temp = list_id[i].copy()
        for j in range(1, 10 - ind + 1):
            for dig in need_digit:
                temp[-j] = dig

                list_id.append(temp.copy())
                dict_list.append(i)
print("add els without 1",datetime.now()-strt)
# print(type(list_id),type(list_id[0]))

data = [float(x.text.strip()) for x in Bs_data.find_all('value')]

print("ooppoi")

tmp = np.full((1000,1000),int(255))
tmp2 = np.full((1000,1000),int(0))
tmp3 = np.full((1000,1000),int(0))

Getting([tmp, tmp2,tmp3])
