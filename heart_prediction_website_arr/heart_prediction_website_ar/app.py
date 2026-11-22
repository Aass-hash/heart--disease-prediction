import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="التنبؤ بمؤشرات أمراض القلب", page_icon="❤️", layout="centered")

# تنسيق عربي من اليمين إلى اليسار
st.markdown("""
<style>
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
}
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # البحث عن ملف النموذج بجانب app.py، ثم في المجلد الأعلى
    app_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(app_dir, "ann_model.joblib"),
        os.path.join(os.path.dirname(app_dir), "ann_model.joblib"),
    ]
    for model_path in possible_paths:
        if os.path.isfile(model_path):
            return joblib.load(model_path)

    searched = "\n".join(possible_paths)
    raise FileNotFoundError(
        "لم يتم العثور على ملف ann_model.joblib. ضع الملف بجانب app.py. "
        f"المسارات التي تم البحث فيها:\n{searched}"
    )

model = load_model()

st.title("❤️ نظام التنبؤ بمؤشرات أمراض القلب")
st.write("أدخل القياسات والبيانات المطلوبة، ثم اضغط على زر **اعرض نتيجة التنبؤ**.")
st.info(
    "هذا المشروع تعليمي ويعرض تصنيف النموذج اعتمادًا على بيانات التدريب فقط. "
    "النتيجة ليست تشخيصًا طبيًا ولا تُستخدم لاتخاذ قرارات علاجية."
)

# القيم الداخلية مطابقة تمامًا للقيم المستخدمة عند تدريب النموذج.
choices = {
    "sex": ["female", "male"],
    "cp": ["asympt", "atyp_angina", "non_anginal", "typ_angina"],
    "fbs": ["f", "t"],
    "restecg": ["left_vent_hyper", "normal", "st_t_wave_abnormality"],
    "exang": ["no", "yes"],
    "slope": ["down", "flat", "up"],
    "thal": ["fixed_defect", "normal", "reversable_defect"]
}

arabic_choices = {
    "sex": {
        "female": "أنثى",
        "male": "ذكر"
    },
    "cp": {
        "asympt": "ألم غير نمطي",
        "atyp_angina": "ذبحة صدرية غير نمطية",
        "non_anginal": "ألم غير ذبحي",
        "typ_angina": "ذبحة صدرية نموذجية"
    },
    "fbs": {
        "f": "لا",
        "t": "نعم"
    },
    "restecg": {
        "left_vent_hyper": "تضخم البطين الأيسر",
        "normal": "طبيعي",
        "st_t_wave_abnormality": "تغير في موجة ST أو T"
    },
    "exang": {
        "no": "لا",
        "yes": "نعم"
    },
    "slope": {
        "down": "هابط",
        "flat": "مستوي",
        "up": "صاعد"
    },
    "thal": {
        "fixed_defect": "عيب ثابت",
        "normal": "طبيعي",
        "reversable_defect": "عيب قابل للعكس"
    }
}

ranges = {
    "age": {"min": 29.0, "max": 77.0, "median": 55.0},
    "trestbps": {"min": 94.0, "max": 200.0, "median": 130.0},
    "chol": {"min": 126.0, "max": 564.0, "median": 240.0},
    "thalach": {"min": 71.0, "max": 202.0, "median": 153.0},
    "oldpeak": {"min": 0.0, "max": 6.2, "median": 0.8},
    "ca": {"min": 0.0, "max": 3.0, "median": 0.0}
}

numeric_labels = {
    "age": "العمر (بالسنوات)",
    "trestbps": "ضغط الدم أثناء الراحة",
    "chol": "مستوى الكوليسترول في الدم",
    "thalach": "أعلى معدل لضربات القلب",
    "oldpeak": "انخفاض مقطع ST أثناء الجهد (Oldpeak)",
    "ca": "عدد الأوعية الدموية الرئيسية الملونة"
}

st.subheader("بيانات الشخص")
with st.form("prediction_form"):
    values = {}
    col1, col2 = st.columns(2)

    for i, (col, info) in enumerate(ranges.items()):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            step = 0.1 if col == "oldpeak" else 1.0
            values[col] = st.number_input(
                numeric_labels[col],
                min_value=float(info["min"]),
                max_value=float(info["max"]),
                value=float(info["median"]),
                step=step
            )

    st.markdown("---")
    st.caption("اختر الإجابة المناسبة لكل بند:")
    col1, col2 = st.columns(2)

    categorical_labels = {
        "sex": "الجنس",
        "cp": "نوع ألم الصدر",
        "fbs": "هل سكر الدم أثناء الصيام أعلى من 120؟",
        "restecg": "نتيجة تخطيط القلب أثناء الراحة",
        "exang": "هل يحدث ألم صدر مع المجهود؟",
        "slope": "ميل مقطع ST أثناء ذروة الجهد",
        "thal": "نتيجة اختبار Thal"
    }

    for i, (col, opts) in enumerate(choices.items()):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            selected_arabic = st.selectbox(
                categorical_labels[col],
                opts,
                format_func=lambda value, key=col: arabic_choices[key][value],
                key=col
            )
            values[col] = selected_arabic

    submitted = st.form_submit_button("اعرض نتيجة التنبؤ", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([values])
    prediction = int(model.predict(input_df)[0])

    st.subheader("نتيجة التنبؤ")
    if prediction == 0:
        st.success("نتيجة النموذج: الفئة (0)")
        st.write(
            "بحسب ترميز مجموعة البيانات المستخدمة في التدريب، "
            "هذه الفئة تقابل التسمية: أقل من 50% (<50)."
        )
    else:
        st.warning("نتيجة النموذج: الفئة (1)")
        st.write(
            "بحسب ترميز مجموعة البيانات المستخدمة في التدريب، "
            "هذه الفئة تقابل التسمية: أكبر من 50% (>50_1)."
        )

    st.markdown(
        "**ماذا تعني النتيجة؟** هي الفئة التي اختارها النموذج بناءً على المدخلات، "
        "وليست تحديدًا لما يعاني منه الشخص أو تشخيصًا لحالته."
    )

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        st.write("احتمالات النموذج لكل فئة (وليست نسبة خطورة طبية):")
        st.write(f"- الفئة (0) — أقل من 50%: **{probs[0] * 100:.1f}%**")
        st.write(f"- الفئة (1) — أكبر من 50%: **{probs[1] * 100:.1f}%**")

    st.caption(
        "ترميز الهدف في ملف التدريب الأصلي: <50 = 0 و >50_1 = 1. "
        "لا تستخدم هذه النتيجة بدل مراجعة الطبيب أو الفحوصات الطبية."
    )
if submitted:
    input_df = pd.DataFrame([values])
    prediction = int(model.predict(input_df)[0])

    st.subheader("❤️ نتيجة تحليل مؤشرات القلب")

    if prediction == 0:
        st.success("🟢 التصنيف: لا توجد مؤشرات على تضيق الشرايين بنسبة 50% أو أكثر")
        st.write("تشير نتيجة النموذج إلى أن الحالة تقع ضمن الفئة الأولى.")
    else:
        st.error("🔴 التصنيف: توجد مؤشرات على تضيق الشرايين بنسبة 50% أو أكثر")
        st.write("تشير نتيجة النموذج إلى أن الحالة تقع ضمن الفئة الثانية.")
    

    st.info(
        "هذه النتيجة تصنيف آلي تعليمي وليست تشخيصًا طبيًا مؤكدًا. "
        "يجب مراجعة الطبيب لتقييم الحالة وإجراء الفحوصات اللازمة."
    )
