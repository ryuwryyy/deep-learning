from sklearn import svm

#学習器にXORデータを与える[入力１,入力2,出力]・・・①

xor_data=[
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

#データとラベルに分ける・・・②

data=[]
label=[]

for row in xor_data:
    p=row[0]
    q=row[1]
    r=row[2]
    data.append([p,q])
    label.append(r)

#学習・・・③

clf=svm.SVC()
clf.fit(data,label)


#予測・・・④

pre=clf.predict(data)
print("予測結果:",pre)


#結果を確認・・・⑤

ok=0
total=0

for idx,answer in enumerate(label):
    p=pre[idx]
    if p==answer :
        ok=ok+1
        total=total+1

print("正解率:",ok,"/","total","-",ok/total)
