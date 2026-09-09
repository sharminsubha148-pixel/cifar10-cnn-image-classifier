import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ১. ওয়েব অ্যাপের ইন্টারফেস
st.set_page_config(page_title="Image Classification App", page_icon="🤖", layout="centered")
st.title("🤖 CIFAR-10 Image Classification Web App")
st.write("আপনার যেকোনো একটি ছবি আপলোড করুন, মডেলটি প্রেডিক্ট করে বলে দেবে এটি কীসের ছবি।")

# ২. কনভার্ট হওয়া নতুন .h5 মডেলটি লোড করা (এতে কোনো জিপ বম্ব ইরর আসবে না)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('cifar10_model.h5')

try:
    model = load_my_model()
    st.sidebar.success("✅ মডেলটি সফলভাবে লোড হয়েছে!")
except Exception as e:
    st.sidebar.error(f"❌ মডেল লোড করতে সমস্যা হয়েছে: {e}")
    st.stop()

# ৩. ১০টি ক্লাসের নাম
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# ৪. ফাইল আপলোডার
uploaded_file = st.file_uploader("একটি ছবি সিলেক্ট করুন (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='আপলোড করা ছবি', use_container_width=True)
    st.write("🔍 মডেলটি ছবিটি পরীক্ষা করছে...")
    
    try:
        # ৫. ইমেজ প্রি-প্রসেসিং
        resized_img = image.resize((32, 32))
        img_array = np.array(resized_img)
        
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
            
        normalized_img = img_array / 255.0
        input_batch = np.expand_dims(normalized_img, axis=0)
        
        # ৬. প্রেডিকশন
        predictions = model.predict(input_batch)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_idx]
        confidence = np.max(predictions[0]) * 100
        
        # ৭. ফলাফল দেখানো
        st.success(f"### 🎉 প্রেডিকশন: **{predicted_class_name.upper()}**")
        st.info(f"💡 মডেলটি এই ব্যাপারে **{confidence:.2f}%** নিশ্চিত।")
        
        st.write("📊 অন্যান্য ক্লাসের সম্ভাবনা:")
        for idx, name in enumerate(class_names):
            st.text(f"{name}: {predictions[0][idx]*100:.2f}%")
            
    except Exception as e:
        st.error(f"⚠️ ছবি প্রসেস করতে সমস্যা হয়েছে: {e}")