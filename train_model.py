"""Train a small MLP cat/dog classifier from the repository's image archive.
This is a newly trained portfolio model, not a preexisting checkpoint.
"""
from pathlib import Path
from io import BytesIO
import json
import random
import zipfile

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report

ZIP=Path("cats_and_dogs.zip")
OUT=Path("models")
SIZE=(32,32)
rng=random.Random(42)

def vectorize(data):
    with Image.open(BytesIO(data)) as im:
        im=ImageOps.fit(im.convert("RGB"),SIZE,method=Image.Resampling.BILINEAR)
        return np.asarray(im,dtype=np.float32).reshape(-1)/255.0

with zipfile.ZipFile(ZIP) as z:
    groups={}
    for label,folder in ((0,"cats"),(1,"dogs")):
        names=[n for n in z.namelist() if n.startswith("cats_and_dogs/train/"+folder+"/") and n.lower().endswith((".jpg",".jpeg",".png"))]
        rng.shuffle(names)
        groups[label]=names[:1500]
        print(folder,"images available",len(names),"using",len(groups[label]),flush=True)
    X=[];y=[]
    for label,names in groups.items():
        for path in names:
            try:
                X.append(vectorize(z.read(path)))
                y.append(label)
            except Exception as exc:
                print("Skipped invalid image:",path,str(exc)[:120])
X=np.asarray(X,dtype=np.float32)
y=np.asarray(y,dtype=np.int32)
print("Training on images",len(X),flush=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
model=MLPClassifier(hidden_layer_sizes=(64,24),activation="relu",solver="adam",
                    batch_size=128,learning_rate_init=.001,alpha=.003,
                    max_iter=36,early_stopping=True,n_iter_no_change=6,
                    random_state=42)
model.fit(Xtr,ytr)
pred=model.predict(Xte)
acc=float(accuracy_score(yte,pred))
print("HOLDOUT ACCURACY",acc,"EPOCHS",model.n_iter_,flush=True)
print(classification_report(yte,pred,target_names=["Cat","Dog"]),flush=True)
OUT.mkdir(exist_ok=True)
joblib.dump({"estimator":model,"image_size":SIZE,"class_names":["Cat","Dog"],
            "holdout_accuracy":acc,"train_samples":len(Xtr),"test_samples":len(Xte)},
            OUT/"cats_dogs_mlp.joblib",compress=3)
(OUT/"model_info.json").write_text(json.dumps({
    "model":"MLPClassifier","image_size":list(SIZE),"train_samples":len(Xtr),
    "test_samples":len(Xte),"holdout_accuracy":acc,"class_names":["Cat","Dog"],
    "note":"New portfolio MLP trained on the repository image archive."
},indent=2),encoding="utf-8")
print("SAVED", (OUT/"cats_dogs_mlp.joblib").stat().st_size,flush=True)
