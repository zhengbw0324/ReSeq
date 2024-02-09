import os
from collections import defaultdict

import numpy as np
import pandas as pd
import os
import time


def get_timestamp(timeStr):
    timeArray = time.strptime(timeStr, '%Y-%m-%d')
    timeStamp = int(time.mktime(timeArray))
    return timeStamp


root = "./"
dataname = "askubuntu"
# dataname = "stackoverflow"
dir = os.path.join(root, dataname)
file = "sx-"+ dataname+ "-a2q.txt"

fpath = os.path.join(root, dataname, file)
ti = 1298908800 #2011-3-1
# ti = 1388505600   #2014-1-1
# ti = 1420041600   #2015-1-1
# ti=1443628800 #2015-10-1
# 1433088000  2015-6-1
# 1443628800  2015-10-1

suc_set = set()
aset = set()
qset = set()
f = open(fpath, 'r')
for line in f:
    line = line.strip().split(" ")
    a,q,t = line
    if int(t)>=ti:
        suc_set.add((a,q,t))
        aset.add(a)
        qset.add(q)

f.close()

print("inter_num",len(list(suc_set)))
print("a_num",len(list(aset)))
print("q_num",len(list(qset)))

a_suc = defaultdict(list)
q_suc = defaultdict(list)

for s in suc_set:
    a, q = s[0],s[1]
    a_suc[a].append(q)
    q_suc[q].append(a)

while True:
    illegal_a = set()
    illegal_q = set()

    for a in a_suc.keys():
        if len(a_suc[a]) < 5:
            illegal_a.add(a)

    for q in q_suc.keys():
        if len(q_suc[q]) < 5:
            illegal_q.add(q)

    if len(illegal_q) == 0 and len(illegal_q) == 0:
        break

    for a in illegal_a:
        for q in a_suc[a]:
            q_suc[q].remove(a)
        del a_suc[a]

    for q in illegal_q:
        for a in q_suc[q]:
            a_suc[a].remove(q)
        del q_suc[q]
tlist = []
filtered_suc_set = set()
for s in suc_set:
    a, q = s[0],s[1]
    if a in a_suc.keys() and q in q_suc.keys():
        filtered_suc_set.add(s)
        tlist.append(int(s[2]))


print("after 5-core：")

print("a_num",len(a_suc.keys()))
print("q_num",len(q_suc.keys()))
print("inter_num",len(filtered_suc_set))



all_q = ["[PAD]"] + sorted(list(set(q_suc.keys())))
all_a = ["[PAD]"] + sorted(list(set(a_suc.keys())))
ump = np.array(all_q)
utoken2id = {t: i for i, t in enumerate(ump)}
imp = np.array(all_a)
itoken2id = {t: i for i, t in enumerate(imp)}

user_token = {"id2token": ump, "token2id": utoken2id}
item_token = {"id2token": imp, "token2id": itoken2id}

with open(os.path.join(dir, "user-token.pkl"), 'wb') as f:
    pickle.dump(user_token, f)


with open(os.path.join(dir, "item-token.pkl"), 'wb') as f:
    pickle.dump(item_token, f)



f_target = open(os.path.join(dir, dataname+'.inter'), 'w')
head = '\t'.join(["q_id:token","a_id:token","direct:token","timestamp:float","label:float"]) + '\n'
f_target.write(head)
for s in filtered_suc_set:
    a, q ,t = s
    line = '\t'.join([q, a, '0', t, '1']) + '\n'
    f_target.write(line)

f_target.close()
