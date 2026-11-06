"""Cats and Dogs — interactive image classification powered by a trained MLP."""
from pathlib import Path
from io import BytesIO
import zipfile
import random
import numpy as np
import pandas as pd
import joblib
from PIL import Image,ImageOps
import altair as alt
import streamlit as st

BASE=Path(__file__).resolve().parent
MODEL=BASE/"models"/"cats_dogs_mlp.joblib"
st.set_page_config(page_title="Cat vs Dog | Image Classification",page_icon="🐾",layout="wide",initial_sidebar_state="collapsed")
css=BASE/"vision_theme.css"
if css.exists():
    st.markdown("<style>"+css.read_text(encoding="utf-8")+"</style>",unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL.exists():return None
    return joblib.load(MODEL)

def classify(im,artifact):
    arr=np.asarray(ImageOps.fit(im.convert("RGB"),tuple(artifact["image_size"]),method=Image.Resampling.BILINEAR),dtype=np.float32).reshape(1,-1)/255.
    scores=artifact["estimator"].predict_proba(arr)[0]
    return sorted([(label,float(prob)) for label,prob in zip(artifact["class_names"],scores)],key=lambda p:p[1],reverse=True)

st.markdown("""<section class="hero"><div><div class="eyebrow">NEURAL NETWORKS / COMPUTER VISION</div><h1>Cat or Dog? <em>Ask the model.</em></h1><p>Upload an animal photo and explore an image classifier trained on the project's own dataset.</p><div class="chips"><span>Neural networks</span><span>Image classification</span><span>Interactive inference</span></div></div><div class="hero-decor"><span>🐈</span><strong>VS</strong><span>🐕</span></div></section>""",unsafe_allow_html=True)

model=load_model()
if model is None:
    st.warning("The training workflow has not yet published a model artifact. Prediction will become available only when the trained model is present.")
    st.link_button("Inspect GitHub training workflow","https://github.com/ayoubennadifb-spec/Cats-dogs-Neural-network-classification/actions")
    st.stop()

tabs=st.tabs(["◈  Classify an Image","▦  Dataset Samples","ⓘ  About the Model"])
with tabs[0]:
    left,right=st.columns([1,1],gap="large")
    with left:
        with st.container(border=True):
            st.markdown("### Upload a pet photograph")
            photo=st.file_uploader("Choose a JPG, PNG or WEBP image",type=["jpg","jpeg","png","webp"],help="Uploaded images are processed in memory and not saved.")
            if photo:
                try:
                    image=Image.open(photo)
                    if image.width*image.height>25_000_000:raise ValueError("Image is too large. Use an image under 25 megapixels.")
                    st.image(image,caption=f"Your uploaded image · {image.width} × {image.height} pixels",use_container_width=True)
                except Exception as e:
                    st.error(f"Cannot read image: {e}")
                    photo=None
            else:
                st.markdown('<div class="upload-placeholder"><div class="paw">🐾</div><strong>Your image preview appears here</strong><span>JPG · PNG · WEBP</span></div>',unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown("### Neural network output")
            if photo:
                try:
                    results=classify(image,model)
                    winner,confidence=results[0]
                    st.markdown(f'<div class="prediction-card"><span>MODEL CLASSIFICATION</span><strong>{winner} {"🐈" if winner=="Cat" else "🐕"}</strong><small>Estimated class probability: {confidence:.1%}</small></div>',unsafe_allow_html=True)
                    values=pd.DataFrame(results,columns=["Class","Probability"])
                    fig=alt.Chart(values).mark_bar(cornerRadiusEnd=6).encode(
                        x=alt.X("Probability:Q",scale=alt.Scale(domain=[0,1]),axis=alt.Axis(format="%")),
                        y=alt.Y("Class:N",sort="-x",title=None),
                        color=alt.Color("Class:N",scale=alt.Scale(domain=["Cat","Dog"],range=["#a98be0","#35b7ad"]),legend=None),
                        tooltip=["Class:N",alt.Tooltip("Probability:Q",format=".1%")]
                    ).properties(height=145)
                    st.altair_chart(fig,use_container_width=True)
                    st.caption("Probability is the neural network's output, not a guarantee of correctness. Photos outside the training distribution can be misclassified.")
                except Exception as e:
                    st.error(f"Image classification failed: {e}")
            else:
                st.info("Upload an image to receive a real cat/dog prediction.")
            st.metric("Held-out test accuracy",f"{model['holdout_accuracy']:.1%}")
            st.caption(f"Small MLP classifier · {model['train_samples']:,} training images · {model['test_samples']:,} held-out images.")
with tabs[1]:
    st.markdown("### Sample images from the training archive")
    st.caption("The repository includes images, but no pre-existing trained model. The web application trains its own compact MLP.")
    archive=BASE/"cats_and_dogs.zip"
    if archive.exists():
        try:
            with zipfile.ZipFile(archive) as z:
                allnames=z.namelist()
                for name in ("cats","dogs"):
                    images=sorted(n for n in allnames if n.startswith(f"cats_and_dogs/train/{name}/") and n.lower().endswith(".jpg"))
                    if images:
                        st.markdown(f"**{name.title()}**")
                        columns=st.columns(4)
                        for i,imgpath in enumerate(images[:4]):
                            with columns[i]:
                                st.image(z.read(imgpath),use_container_width=True,caption=f"{name.title()} · dataset image")
        except Exception as e:st.warning(f"Dataset gallery unavailable: {e}")
    else:st.info("Sample images are available in the GitHub dataset archive.")
with tabs[2]:
    st.markdown("### Model methodology")
    a,b,c=st.columns(3)
    a.metric("Classifier","MLP")
    b.metric("Input image","32 × 32 RGB")
    c.metric("Training images",model["train_samples"])
    st.write("This portfolio demo trains a small scikit-learn multilayer perceptron on balanced cat and dog samples from the repository ZIP archive. A stratified holdout is used for the displayed accuracy. The model is not a pretrained CNN; it is a newly trained, lightweight neural classifier created for this demo.")
    st.warning("The demo should not be used for safety-critical tasks or to draw conclusions from ambiguous or out-of-distribution images.")
    st.link_button("View GitHub repository","https://github.com/ayoubennadifb-spec/Cats-dogs-Neural-network-classification")
st.markdown('<div class="footer">CAT / DOG VISION LAB <span>Neural network portfolio demo · inference runs on uploaded image in memory</span></div>',unsafe_allow_html=True)
